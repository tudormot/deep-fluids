
from pathlib import Path
import numpy as np


#test that checks that the brain anatomy is constant, ie does not modify as the tumour grows
data_dir = '/home/tudorm/deep-fluids/data/001/D0.00030_r0.0253_x0.469_y0.421_z0.515'  # a file containing successive brain profiles

p = Path(data_dir)
[x for x in p.iterdir() ]

