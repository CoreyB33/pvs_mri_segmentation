import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from aug_script import t1Patches, MaskPatches
#from val_val import t1_train, mask_train
t1_patch_1 = t1Patches[50,:,:,0]
mask_patch_1 = MaskPatches[50,:,:,0]
matplotlib.image.imsave('t1_patch.png',t1_patch_1,cmap='Greys')
matplotlib.image.imsave('mask_patch.png',mask_patch_1)
#t1_patch_2 = t1_train[2100,:,:,0]
#mask_patch_2 = mask_train[2100,:,:,0]
#matplotlib.image.imsave('t1_patch_2.png',t1_patch_2,cmap='Greys')
#matplotlib.image.imsave('mask_patch_2.png',mask_patch_2)

