
import os
if os.environ.get("CLEF_PRIVATE_JPEG_CACHE_GIB")=="2":
    import atexit, json
    from pathlib import Path
    import importlib.abc
    import importlib.machinery
    import sys
    from jpeg_cache import DecodedLRU,MemoryBudget,install,GIB

    class Loader(importlib.abc.Loader):
        def __init__(self,base):
            self.base=base
        def create_module(self,spec):
            return self.base.create_module(spec)
        def exec_module(self,module):
            self.base.exec_module(module)
            cache=DecodedLRU(MemoryBudget(2*GIB))
            install(module,cache)
            note=os.environ.get("CLEF_PRIVATE_JPEG_CACHE_REPORT")
            if note:
                from jpeg_cache import DECLARED_IMAGE
                root=Path(note)
                pid=os.getpid()
                with (root/f"installed-{pid}.json").open("x") as dst:
                    json.dump({"pid":pid,"capacity_requested":2*GIB,
                               "image_loader_sha256":DECLARED_IMAGE},dst)
                def counts():
                    with (root/f"stats-{pid}.json").open("x") as dst:
                        json.dump({"lookups":cache.lookups,"hits":cache.hits,
                            "peak_weight":cache.peak,"retained":cache.retained},dst)
                atexit.register(counts)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self,name,path=None,target=None):
            if name=="vllm.multimodal.media.image":
                spec=importlib.machinery.PathFinder.find_spec(name,path,target)
                if spec is not None and spec.loader is not None:
                    spec.loader=Loader(spec.loader)
                return spec

    sys.meta_path.insert(0,Finder())
