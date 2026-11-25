
import base64,json,statistics,sys
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image
from vllm.multimodal.media import image as stock
from vllm.multimodal.hasher import MultiModalHasher
from jpeg_cache import DecodedLRU,MemoryBudget,install,GIB

root=Path(sys.argv[1])
data=root/"erqa-fixed"
samples=[json.loads(row) for row in (data/"samples.jsonl").read_text().splitlines()]
by_size=sorted({i["sha256"]:i for s in samples for i in s["images"]}.values(),
               key=lambda i:i["width"]*i["height"])
chosen=[by_size[(i*529+7)//14] for i in range(15)]
raw=[]
for im in chosen:
    p=(data/im["path"]).resolve()
    assert p.is_relative_to(data.resolve())
    blob=p.read_bytes()
    assert sha256(blob).hexdigest()==im["sha256"]
    raw.append(blob)

def fixture(mode,rotate):
    image=Image.new(mode,(31,27),0)
    exif=Image.Exif()
    if rotate:
        exif[274]=6
    out=BytesIO()
    image.save(out,format="JPEG",exif=exif)
    return out.getvalue()

raw.extend(fixture(mode,rotate) for mode,rotate in (("RGB",True),("CMYK",False),("L",True)))
baseline=stock.ImageMediaIO.load_base64

def check(wanted,got,payload):
    ref,w= wanted.media,got.media
    assert got.original_bytes==payload==wanted.original_bytes
    assert got.io_config==wanted.io_config and ref.size==w.size and ref.mode==w.mode
    assert ref.info==w.info and ref.format==w.format and ref.tobytes()==w.tobytes()
    assert ref.getexif().tobytes()==w.getexif().tobytes()
    assert MultiModalHasher.hash_kwargs("blake3",image=wanted)==MultiModalHasher.hash_kwargs("blake3",image=got)

class Budget:
    def __init__(self,limit):
        self.used=0
        self.limit=limit
    def charge(self,n):
        if self.used+n>self.limit:
            raise ValueError("decoded budget exceeded")
        self.used+=n

cache=DecodedLRU(lambda resident:2*GIB)
assert install(stock,cache)==baseline
for raw_bytes in raw:
    b64=base64.b64encode(raw_bytes)
    original_budget=Budget(2*GIB)
    decoded=baseline(stock.ImageMediaIO(decoded_budget=original_budget),"image/jpeg",b64)
    current_budget=Budget(original_budget.used*2)
    decoder=stock.ImageMediaIO(decoded_budget=current_budget)
    first=decoder.load_base64("image/jpeg",b64)
    second=decoder.load_base64("image/jpeg",b64)
    check(decoded,first,raw_bytes);check(decoded,second,raw_bytes)
    assert current_budget.used==original_budget.used*2
    try:
        decoder.load_base64("image/jpeg",b64)
    except ValueError as exc:
        assert str(exc)=="decoded budget exceeded" and current_budget.used==original_budget.used*2
    else:
        raise AssertionError("hit did not enforce decoded budget")
    first.media.putpixel((0,0),(254,12,14))
    first.media.info["test_changed"]=True
    first.media.close()
    check(decoded,second,raw_bytes)
    check(decoded,stock.ImageMediaIO().load_base64("image/jpeg",b64),raw_bytes)
assert len(cache.cache)==len(raw) and cache.retained<=2*GIB

p=raw[5]
encoded=base64.b64encode(p)
reference=baseline(stock.ImageMediaIO(),"image/jpeg",encoded)
def hit(i):
    copy=stock.ImageMediaIO().load_base64("image/jpeg",encoded)
    check(reference,copy,p)
    copy.media.putpixel((0,0),(i,0,0))
    copy.media.close()
with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(hit,range(12)))
check(reference,stock.ImageMediaIO().load_base64("image/jpeg",encoded),p)

for mode,declared in ((None,"image/jpeg"),("RGB","image/png")):
    src=stock.ImageMediaIO(image_mode=mode)
    original=baseline(src,declared,encoded)
    keep=len(cache.cache)
    check(original,src.load_base64(declared,encoded),p)
    assert len(cache.cache)==keep

for bad in (b"@@@@",base64.b64encode(b"\xff\xd8\xff wrong jpeg")):
    errs=[]
    for target in (baseline,stock.ImageMediaIO.load_base64):
        try:
            target(stock.ImageMediaIO(),"image/jpeg",bad)
        except (ValueError,TypeError) as e:
            errs.append((type(e),__import__("re").sub(r"(?<=_io.BytesIO object at )0x[0-9a-fA-F]+","ADDR",str(e))))
    assert len(errs)==2 and errs[0]==errs[1]

tiny=DecodedLRU(lambda already:1024)
stock.ImageMediaIO.load_base64=baseline
install(stock,tiny)
check(reference,stock.ImageMediaIO().load_base64("image/jpeg",encoded),p)
assert not tiny.cache and tiny.retained==0
cache.clear()
assert not cache.cache and cache.retained==0

def write(path,value):
    path.write_text(str(value)+"\n")

with TemporaryDirectory() as temp:
    base=Path(temp)
    proc=base/"proc"
    (proc/"self").mkdir(parents=True)
    write(proc/"meminfo","MemTotal: 20971520 kB\nMemAvailable: 10485760 kB")
    mount=base/"cgroup"
    child=mount/"child"
    child.mkdir(parents=True)
    write(proc/"self/cgroup","0::/child")
    write(proc/"self/mountinfo",f"1 0 0:1 / {mount} rw - cgroup2 cgroup rw")
    for folder,hard,used in ((child,12*GIB,3*GIB),(mount,8*GIB,6*GIB)):
        write(folder/"memory.max",hard);write(folder/"memory.current",used)
        write(folder/"memory.high","max");write(folder/"memory.events","high 0\nmax 0\noom 0\noom_kill 0")
    mm=MemoryBudget(2*GIB,proc=proc)
    cap=mm(0)
    assert 0<cap<GIB
    write(mount/"memory.events","high 1\nmax 0\noom 0\noom_kill 0")
    assert mm(0)==0
    write(proc/"self/cgroup","15:memory:/child")
    write(proc/"self/mountinfo",f"1 0 0:1 / {mount} rw - cgroup cgroup rw,memory")
    for folder,hard,used in ((child,12*GIB,3*GIB),(mount,8*GIB,6*GIB)):
        write(folder/"memory.limit_in_bytes",hard);write(folder/"memory.usage_in_bytes",used)
        write(folder/"memory.failcnt",0)
        write(folder/"memory.stat",f"hierarchical_memory_limit {hard}")
    mm=MemoryBudget(2*GIB,proc=proc)
    assert mm(0)==cap
    write(mount/"memory.failcnt",1)
    assert mm(0)==0

now=[0.0]
caps=[2*GIB]
lr=DecodedLRU(lambda already:caps[0],clock=lambda:now[0])
stock.ImageMediaIO.load_base64=baseline
install(stock,lr)
stock.ImageMediaIO().load_base64("image/jpeg",encoded)
assert lr.retained>0
caps[0]=0
now[0]=2.0
check(reference,stock.ImageMediaIO().load_base64("image/jpeg",encoded),p)
assert lr.retained==0 and not lr.cache
assert MemoryBudget(2*GIB)(0)==2*GIB
print("JPEG CPU contract: 15 real + 3 codec/EXIF, original BLAKE3, independent mutation/close, parallel hits, bad and oversized fallback, exact charging, cgroup v1/v2 parent pressure and clear")
