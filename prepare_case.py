from pathlib import Path
import argparse
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('--shape',nargs=4,type=int,default=[2,3,5,40],metavar=('B','M','N','K'))
p.add_argument('--mode',choices=['random','negative','zero'],default='random')
p.add_argument('--dtype',choices=['fp16','bf16'],default='fp16')
p.add_argument('--tx1',type=int,choices=[0,1],default=0)
p.add_argument('--tx2',type=int,choices=[0,1],default=0)
p.add_argument('--seed',type=int,default=2026)
a=p.parse_args();B,M,N,K=a.shape
assert 1<=B<=64 and 1<=M<=8192 and 1<=N<=8192 and 32<=K<=8192 and K%8==0
assert B*M*K<=2**26 and B*N*K<=2**26
rng=np.random.default_rng(a.seed)
if a.dtype=='bf16':
    from ml_dtypes import bfloat16
    storage_dtype=bfloat16;dtype_id=2
else:
    storage_dtype=np.float16;dtype_id=1
if a.mode=='random':
    x1=rng.uniform(-1,1,(B,M,K)).astype(storage_dtype)
    x2=rng.uniform(-1,1,(B,K,N)).astype(storage_dtype)
elif a.mode=='negative':
    x1=np.ones((B,M,K),dtype=storage_dtype)
    x2=np.broadcast_to(-np.arange(N,0,-1,dtype=np.float64)[None,None,:]/K,(B,K,N)).astype(storage_dtype).copy()
else:
    x1=np.zeros((B,M,K),dtype=storage_dtype);x2=np.zeros((B,K,N),dtype=storage_dtype)
# This script is for small integration cases, not large golden generation.
g=(x1.astype(np.float64)@x2.astype(np.float64)).max(axis=-1).sum(axis=-1).astype(np.float32)
Path('input').mkdir(exist_ok=True);Path('output').mkdir(exist_ok=True)
s1=np.ascontiguousarray(x1.transpose(0,2,1) if a.tx1 else x1)
s2=np.ascontiguousarray(x2.transpose(0,2,1) if a.tx2 else x2)
s1.tofile('input/x1.bin');s2.tofile('input/x2.bin');g.tofile('output/golden_y.bin')
Path('output/y.bin').unlink(missing_ok=True)
Path('case_args.txt').write_text(f'{B} {M} {N} {K} {dtype_id} {a.tx1} {a.tx2}\n')
print('args:',Path('case_args.txt').read_text().strip());print('golden:',g)
