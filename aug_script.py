# Test script for data augmentation
#from data_aug.data_aug import *
#from data_aug.bbox_util import *
#import rotation
#from rotation import rotateit
import numpy as np
import os
import sys
import nibabel as nib
from sklearn.utils import shuffle
import random
import copy
import cv2
import pickle as pkl
import matplotlib.pyplot as plt
num_channels = 1
plane = "axial"
patch_size="64x64"
PATCH_SIZE = [int(x) for x in patch_size.split("x")]
DATA_DIR = os.path.join("data", "train_conservative")
t1_names = os.listdir(DATA_DIR)
mask_names = os.listdir(DATA_DIR)
t1_names = [x for x in t1_names if "059_t1" in x]
mask_names = [x for x in mask_names if "pvs" in x]
numatlas = len(t1_names)
patchsize=PATCH_SIZE
patchsize = np.asarray(patchsize, dtype=int)
padsize = np.max(patchsize + 1)
f = 0
for i in range(0, numatlas):
    maskname = mask_names[i]
    maskname = os.path.join(DATA_DIR, maskname)
    temp = nib.load(maskname)
    mask = temp.get_data()
    f = f + np.sum(mask)

max_patch=20000
print("Total number of lesion patches =", f)
total_num_patches = int(np.minimum(max_patch * numatlas, f))
#single_subject_num_patches = total_num_patches // numatlas
print("Allowed total number of patches =", total_num_patches)

doubled_num_patches = total_num_patches * 2
#quad_num_patches = doubled_num_patches * 2
t1_matsize = (doubled_num_patches,patchsize[0], patchsize[1], num_channels)
Mask_matsize = (doubled_num_patches, patchsize[0], patchsize[1],1)
t1Patches = np.zeros(t1_matsize, dtype=np.float16)
MaskPatches = np.zeros(Mask_matsize, dtype=np.float16)
indices = [x for x in range(doubled_num_patches)]
indices = shuffle(indices, random_state=0)
cur_idx = 0
planar_codes = {"axial": (0,1,2), "sagittal": (1,2,0), "coronal": (2,0,1)}
planar_code=planar_codes["axial"]
i=0
t1name=t1_names[i]
t1name=os.path.join(DATA_DIR,t1name)
temp=nib.load(t1name)
t1=temp.get_data()
t1=np.asarray(t1,dtype=np.float16)
maskname = mask_names[i]
maskname = os.path.join(DATA_DIR, maskname)
temp = nib.load(maskname)
mask = temp.get_data()
mask = np.asarray(mask, dtype=np.float16)
t1 = np.transpose(t1, axes=planar_code)
mask = np.transpose(mask, axes=planar_code)
invols = [t1]
patchsize = np.asarray(patchsize, dtype=int)
rng = random.SystemRandom()

mask = np.asarray(mask, dtype=np.float32)
patch_size = np.asarray(patchsize, dtype=int)
dsize = np.floor(patchsize/2).astype(dtype=int)

mask_lesion_indices = np.nonzero(mask)
mask_lesion_indices = np.asarray(mask_lesion_indices, dtype=int)
total_lesion_patches = len(mask_lesion_indices[0])
maxpatch=max_patch
num_patches = np.minimum(maxpatch, total_lesion_patches)
randidx = rng.sample(range(0, total_lesion_patches), num_patches)
# here, 3 corresponds to each axis of the 3D volume
shuffled_mask_lesion_indices = np.ndarray((3, num_patches))
for i in range(0, num_patches):
    for j in range(0, 3):
        shuffled_mask_lesion_indices[j,
                                     i] = mask_lesion_indices[j, randidx[i]]
shuffled_mask_lesion_indices = np.asarray(
        shuffled_mask_lesion_indices, dtype=int)

tmp = copy.deepcopy(invols[0])
tmp[tmp > 0] = 1
tmp[tmp <= 0] = 0
tmp = np.multiply(tmp, 1-mask)

healthy_brain_indices = np.nonzero(tmp)
healthy_brain_indices = np.asarray(healthy_brain_indices, dtype=int)
num_healthy_indices = len(healthy_brain_indices[0])

randidx0 = rng.sample(range(0, num_healthy_indices), num_patches)
# here, 3 corresponds to each axis of the 3D volume
shuffled_healthy_brain_indices = np.ndarray((3, num_patches))
for i in range(0, num_patches):
    for j in range(0, 3):
        shuffled_healthy_brain_indices[j,
                                       i] = healthy_brain_indices[j, randidx0[i]]
shuffled_healthy_brain_indices = np.asarray(
        shuffled_healthy_brain_indices, dtype=int)

newidx = np.concatenate([shuffled_mask_lesion_indices,
                             shuffled_healthy_brain_indices], axis=1)

t1_matsize = (2*num_patches, patchsize[0], patchsize[1], num_channels)
Mask_matsize = (2*num_patches, patchsize[0], patchsize[1],1)
#t1_matsize_unrotated = (2*num_patches, patchsize[0], patchsize[1], num_channels)
#Mask_matsize_unrotated = (2*num_patches, patchsize[0], patchsize[1], 1)
t1Patches = np.ndarray(t1_matsize, dtype=np.float16)
MaskPatches = np.ndarray(Mask_matsize, dtype=np.float16)
#t1Patches_unrotated = np.ndarray(t1_matsize_unrotated, dtype=np.float16)
#MaskPatches_unrotated = np.ndarray(Mask_matsize_unrotated, dtype=np.float16)

for i in range(0, 2*num_patches):
        I = newidx[0, i]
        J = newidx[1, i]
        K = newidx[2, i]
        

        for c in range(num_channels):
            '''
            CTPatches[i, :, :, c] = invols[c][I - dsize[0]: I + dsize[0] + 1,
                                              J - dsize[1]: J + dsize[1] + 1,
                                              K]
            '''

            # trying even-sided patches
            t1Patches[i, :, :, c] = invols[c][I - dsize[0]: I + dsize[0],
                                              J - dsize[1]: J + dsize[1],
                                              K]
            # flairPatches[i, :, :, c] = invols[c][I - dsize[0]: I + dsize[0],
              #                                   J - dsize[1]: J + dsize[1],
               #                                  K]

        '''
        MaskPatches[i, :, :, 0] = mask[I - dsize[0]: I + dsize[0] + 1,
                                       J - dsize[1]:J + dsize[1] + 1,
                                       K]
        '''
        # trying even-sided patches
        # First half of patches have lesions and second half do not (at the center)
        MaskPatches[i, :, :, 0] = mask[I - dsize[0]: I + dsize[0],
                                       J - dsize[1]:J + dsize[1],
                                       K]

# Look at patches using:
        # matplotlib.image.imsave('patch.png', t1Patches[100,:,:,0])
     
#for i in range(0, 2*num_patches):
#        I = newidx[0, i]
#        J = newidx[1, i]
#        K = newidx[2, i]
        

#        for c in range(num_channels):
#            '''
#            CTPatches[i, :, :, c] = invols[c][I - dsize[0]: I + dsize[0] + 1,
#                                              J - dsize[1]: J + dsize[1] + 1,
#                                              K]
#            '''

            # trying even-sided patches
#            t1Patch_unrotated = invols[c][I - dsize[0]: I + dsize[0],
#                                              J - dsize[1]: J + dsize[1],
#                                              K]
#            
#            t1Patch_unrotated=t1Patch_unrotated.astype('float64')
#            t1Patch_rotated=rotateit(t1Patch_unrotated,5)
#            t1Patch_rotated=t1Patch_rotated.astype('float16')
            
#            t1Patches[i+2*num_patches,:,:,c] = t1Patch_rotated
            # flairPatches[i, :, :, c] = invols[c][I - dsize[0]: I + dsize[0],
              #                                   J - dsize[1]: J + dsize[1],
                #                                  K]

        '''
        MaskPatches[i, :, :, 0] = mask[I - dsize[0]: I + dsize[0] + 1,
                                        J - dsize[1]:J + dsize[1] + 1,
                                        K]
        '''
        # trying even-sided patches
        # First half of patches have lesions and second half do not (at the center)
#        MaskPatch_unrotated = mask[I - dsize[0]: I + dsize[0],
#                                        J - dsize[1]:J + dsize[1],
#                                        K]
        
#        MaskPatch_unrotated=MaskPatch_unrotated.astype('float64')
#        MaskPatch_rotated=rotateit(MaskPatch_unrotated,5)
#        MaskPatch_rotated=MaskPatch_rotated.astype('float16')
#        for i in range(0,64):
 #           for j in range(0,64):
 #               if MaskPatch_rotated[i,j]>0.35:
 #                   MaskPatch_rotated[i,j]=1
 #               else:
 #                   MaskPatch_rotated[i,j]=0
 #       
 #       MaskPatches[i+2*num_patches,:,:,0] = MaskPatch_rotated 






































