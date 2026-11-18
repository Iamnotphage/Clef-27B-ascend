#!/usr/bin/env python3
"""CPU projection dispatch trace; no production patch, backend or device timing."""
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import ast,json,hashlib,sys
from pathlib import Path
from unittest.mock import patch
import torch
import torch.nn.functional as F
D=Path(__file__).resolve().parent / 'joint_smallq_dispatch/clef_joint_schema.py'
node=next(n for n in ast.parse(D.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_smallq_cross_attention')
start=next(i for i,n in enumerate(node.body) if isinstance(n,ast.If) and ast.unparse(n.test)=='key is value')
end=next(i for i,n in enumerate(node.body) if isinstance(n,ast.ImportFrom) and n.module=='clef_smallq_kernel')
node.body=node.body[start:end]+[ast.Return(value=ast.Tuple(elts=[ast.Name(id=n,ctx=ast.Load()) for n in ('q','k','v')],ctx=ast.Load()))]
ns={'F':F};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(D),'exec'),ns)
torch.manual_seed(20261119);m=torch.nn.MultiheadAttention(1024,16,batch_first=True).to(torch.bfloat16).eval()
original_linear,original_packed,original_contiguous=F.linear,F._in_projection_packed,torch.Tensor.contiguous
report={'scope':'CPU actual installed torch; D projection-only AST slice, guards/backend excluded; no device timing or precision claim','torch':torch.__version__,'source_D_sha256':hashlib.sha256(D.read_bytes()).hexdigest(),'cases':[]}
with torch.inference_mode():
 for shared,qrows in ((True,1),(False,4)):
  q=torch.randn(1,qrows,1024,dtype=torch.bfloat16);k=torch.randn(1,307,1024,dtype=torch.bfloat16);v=k if shared else k.clone();case={'shared_KV':shared,'Q':qrows,'L':307,'paths':{}};projections={}
  for arm in ('f810_MHA','D_projection_only'):
   linear_calls=[];copies=[];packed_calls=[]
   def linear(x,w,b=None):
    linear_calls.append({'input_shape':list(x.shape),'weight_shape':list(w.shape)})
    return original_linear(x,w,b)
   def packed(q,k,v,*args):
    packed_calls.append({'k_is_v':k is v,'q_is_k':q is k,'q_shape':list(q.shape),'k_shape':list(k.shape)})
    out=original_packed(q,k,v,*args);projections[arm]=out;return out
   def contiguous(x,*args,**kw):
    out=original_contiguous(x,*args,**kw);copies.append({'shape':list(x.shape),'stride':list(x.stride()),'was_contiguous':x.is_contiguous(),'new_storage':out.untyped_storage().data_ptr()!=x.untyped_storage().data_ptr()});return out
   with patch.object(F,'linear',linear),patch.object(F,'_in_projection_packed',packed),patch.object(torch.Tensor,'contiguous',contiguous):
    if arm=='f810_MHA':m(q,k,v,need_weights=False)
    else:ns['_smallq_cross_attention'](m,q,k,v,enabled=True)
   case['paths'][arm]={'linear_calls':linear_calls,'packed_calls':packed_calls,'python_contiguous_sites':copies}
  case['projected_outputs_bitwise_CPU']=all(torch.equal(a,b) for a,b in zip(projections['f810_MHA'],projections['D_projection_only']))
  report['cases'].append(case)
assert 'torch_npu' not in sys.modules and 'triton' not in sys.modules
report['no_device_backends_imported']=True
print(json.dumps(report,indent=2))
