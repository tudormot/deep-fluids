
from pathlib import Path
import numpy as np


#test that checks that the brain anatomy is constant, ie does not modify as the tumour grows
data_dir = '/home/tudorm/deep-fluids/data/001/D0.00030_r0.0253_x0.469_y0.421_z0.515'  # a file containing successive brain profiles

p = Path(data_dir)
#files = [x for x in p.iterdir() ]

first_anatomy = np.load(next(p.iterdir()).resolve())['x'][:,:,:,1:]

for file in p.iterdir():
    profile = np.load(file.resolve())
    print('investigating tumor profile with parameters: ' + str(profile['y']) )
    
    profile = profile['x']
    print('shape of profile:'+str(profile.shape))
    anatomy = profile[:,:,:,1:]
    print('shape of anatomy: ' +str (anatomy.shape) )
    profile = np.sum(profile, axis = -1)
    if np.allclose(profile, 1., atol = 1e-3) is True:
        print( "Seems that all 4 dimensions add to 1")
    if np.allclose(np.sum(anatomy,axis= -1),1., atol = 1e-2) is True:
        print( "Seems that only  the anatomy dimensions add to 1")
    if np.allclose(first_anatomy, anatomy) is True:
        print ("at least the brain anatomy stays the same as tumour progresses")
    

    

