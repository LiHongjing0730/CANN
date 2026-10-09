from pathlib import Path
import numpy as np
B=int(Path('case_args.txt').read_text().split()[0])
y=np.fromfile('output/y.bin',dtype=np.float32)
g=np.fromfile('output/golden_y.bin',dtype=np.float32)
assert y.shape==g.shape==(B,),f'shape: {y.shape}, {g.shape}'
assert np.isfinite(y).all() and np.isfinite(g).all()
print('output:',y);print('golden:',g);print('absolute error:',np.abs(y-g))
np.testing.assert_allclose(y,g,rtol=1e-4,atol=1e-4)
print('PASSED:',Path('case_args.txt').read_text().strip())
