import os
import numpy as np
import nibabel as nib
from sklearn.utils import shuffle
from tqdm import tqdm
from pad import pad_image
import random
import copy
import math
from time import strftime, time

from scipy.ndimage.interpolation import rotate
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

def PadImage(vol, padsize):
    dim = vol.shape
    padsize = np.asarray(padsize, dtype=int)
    dim2 = dim+2*padsize
    temp = np.zeros(dim2, dtype=float)
    temp[padsize:dim[0]+padsize,
         padsize:dim[1]+padsize,
         padsize:dim[2]+padsize] = vol
    return temp


def get_patches(invols, mask, patchsize, maxpatch, num_channels):
    rng = random.SystemRandom()

    mask = np.asarray(mask, dtype=np.float32)
    patch_size = np.asarray(patchsize, dtype=int)
    dsize = np.floor(patchsize/2).astype(dtype=int)

    # find indices of all lesions in mask volume
    mask_lesion_indices = np.nonzero(mask)
    mask_lesion_indices = np.asarray(mask_lesion_indices, dtype=int)
    total_lesion_patches = len(mask_lesion_indices[0])

    num_patches = np.minimum(maxpatch, total_lesion_patches)
    '''
    print("Number of patches used: {} out of {} (max: {})"
          .format(num_patches,
                  total_lesion_patches,
                  maxpatch))
    '''

    randidx = rng.sample(range(0, total_lesion_patches), num_patches)
    # here, 3 corresponds to each axis of the 3D volume
    shuffled_mask_lesion_indices = np.ndarray((3, num_patches))
    for m in range(0, num_patches):
        for n in range(0, 3):
            shuffled_mask_lesion_indices[n,
                                         m] = mask_lesion_indices[n, randidx[m]]
    shuffled_mask_lesion_indices = np.asarray(
        shuffled_mask_lesion_indices, dtype=int)

    # mask out all lesion indices to get all healthy indices
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
    for m in range(0, num_patches):
        for n in range(0, 3):
            shuffled_healthy_brain_indices[n,
                                           m] = healthy_brain_indices[n, randidx0[m]]
    shuffled_healthy_brain_indices = np.asarray(
        shuffled_healthy_brain_indices, dtype=int)

    newidx = np.concatenate([shuffled_mask_lesion_indices,
                             shuffled_healthy_brain_indices], axis=1)
    #t1_matsize=(4*num...
    t1_matsize = (2*num_patches, patchsize[0], patchsize[1], num_channels)
    #Mask_matsize=(4*num...
    Mask_matsize = (2*num_patches, patchsize[0], patchsize[1], 1)
    t1_matsize_unrotated = (2*num_patches, patchsize[0], patchsize[1],num_channels)
    Mask_matsize_unrotated = (2*num_patches, patchsize[0], patchsize[1], 1)

    t1Patches = np.ndarray(t1_matsize, dtype=np.float16)
    MaskPatches = np.ndarray(Mask_matsize, dtype=np.float16)

    for z in range(0, 2*num_patches):
        I = newidx[0, z]
        J = newidx[1, z]
        K = newidx[2, z]
        

        for c in range(num_channels):

            t1Patches[z, :, :, c] = invols[c][I - dsize[0]: I + dsize[0],
                                              J - dsize[1]: J + dsize[1],
                                              K]

        MaskPatches[z, :, :, 0] = mask[I - dsize[0]: I + dsize[0],
                                       J - dsize[1]:J + dsize[1],
                                       K]
    # Augmentation
    #for i in range(0, 2*num_patches):
     #   I = newidx[0, i]
      #  J = newidx[1, i]
      #  K = newidx[2, i]
      #  a=np.random.uniform(-30,30)
      #  a_rad=a*math.pi/180
        

       # for c in range(num_channels):
       #     '''
       #     CTPatches[i, :, :, c] = invols[c][I - dsize[0]: I + dsize[0] + 1,
       #                                       J - dsize[1]: J + dsize[1] + 1,
       #                                       K]
       #     '''

            ## trying even-sided patches
       #     t1Patch_unrotated = invols[c][I - dsize[0]: I + dsize[0],
       #                                       J - dsize[1]: J + dsize[1],
       #                                       K]
       #     
       #     t1Patch_unrotated=t1Patch_unrotated.astype('float64')
       #     t1Patch_rotated=rotate(t1Patch_unrotated,a,reshape=False,mode='nearest')
       #     t1Patch_rotated=t1Patch_rotated.astype('float16')
       #     
       #     
       #     t1Patches[i+2*num_patches,:,:,c] = t1Patch_rotated
       #     # flairPatches[i, :, :, c] = invols[c][I - dsize[0]: I + dsize[0],
       #       #                                   J - dsize[1]: J + dsize[1],
       #        #                                  K]
    

    #    '''
    #    MaskPatches[i, :, :, 0] = mask[I - dsize[0]: I + dsize[0] + 1,
    #                                   J - dsize[1]:J + dsize[1] + 1,
    #                                   K]
    #    '''
        # trying even-sided patches
        # First half of patches have lesions and second half do not (at the center)
    #    MaskPatch_unrotated = mask[I - dsize[0]: I + dsize[0],
    #                                    J - dsize[1]:J + dsize[1],
    #                                    K]
    #    
    #    MaskPatch_unrotated=MaskPatch_unrotated.astype('float64')
    #    MaskPatch_rotated=rotate(MaskPatch_unrotated,a,reshape=False,mode='nearest')
    #    MaskPatch_rotated=MaskPatch_rotated.astype('float16')
        
        #for m in range(0,48):
         #   for n in range(0,48):
          #      if MaskPatch_rotated[m,n]>0.55:
           #         MaskPatch_rotated[m,n]=1
            #    else:
             #       MaskPatch_rotated[m,n]=0
                        
        
        
    #    MaskPatches[i+2*num_patches,:,:,0] = MaskPatch_rotated 
        
    #for i in range(0, 2*num_patches):
    #    I = newidx[0, i]
    #    J = newidx[1, i]
    #    K = newidx[2, i]
    #    a=np.random.uniform(-30,30)
    #    a_rad=a*math.pi/180
    #    

    #    for c in range(num_channels):
    #        '''
    #        CTPatches[i, :, :, c] = invols[c][I - dsize[0]: I + dsize[0] + 1,
    #                                          J - dsize[1]: J + dsize[1] + 1,
    #                                          K]
    #        '''

            ## trying even-sided patches
     #       t1Patch_unrotated = invols[c][I - dsize[0]: I + dsize[0],
      #                                        J - dsize[1]: J + dsize[1],
      #                                        K]
            
      #      t1Patch_unrotated=t1Patch_unrotated.astype('float64')
      #      t1Patch_rotated=rotate(t1Patch_unrotated,a,reshape=False,mode='nearest')
      #      t1Patch_rotated=t1Patch_rotated.astype('float16')
            
            
      #      t1Patches[i+4*num_patches,:,:,c] = t1Patch_rotated
            # flairPatches[i, :, :, c] = invols[c][I - dsize[0]: I + dsize[0],
              #                                   J - dsize[1]: J + dsize[1],
               #                                  K]

      #  '''
      #  MaskPatches[i, :, :, 0] = mask[I - dsize[0]: I + dsize[0] + 1,
      #                                 J - dsize[1]:J + dsize[1] + 1,
      #                                 K]
      #  '''
        # trying even-sided patches
        # First half of patches have lesions and second half do not (at the center)
      #  MaskPatch_unrotated = mask[I - dsize[0]: I + dsize[0],
      #                                  J - dsize[1]:J + dsize[1],
      #                                  K]
        
      #  MaskPatch_unrotated=MaskPatch_unrotated.astype('float64')
      #  MaskPatch_rotated=rotate(MaskPatch_unrotated,a,reshape=False,mode='nearest')
      #  MaskPatch_rotated=MaskPatch_rotated.astype('float16')
        
        #for m in range(0,48):
         #   for n in range(0,48):
          #      if MaskPatch_rotated[m,n]>0.55:
           #         MaskPatch_rotated[m,n]=1
            #    else:
             #       MaskPatch_rotated[m,n]=0
                        
        
        
      #  MaskPatches[i+4*num_patches,:,:,0] = MaskPatch_rotated
        
    #for i in range(0, 2*num_patches):
    #    I = newidx[0, i]
    #    J = newidx[1, i]
    #    K = newidx[2, i]
    #    a=np.random.uniform(-30,30)
    #    a_rad=a*math.pi/180
        

    #    for c in range(num_channels):
    #        '''
    #        CTPatches[i, :, :, c] = invols[c][I - dsize[0]: I + dsize[0] + 1,
    #                                          J - dsize[1]: J + dsize[1] + 1,
    #                                          K]
    #        '''

            ## trying even-sided patches
    #        t1Patch_unrotated = invols[c][I - dsize[0]: I + dsize[0],
    #                                          J - dsize[1]: J + dsize[1],
    #                                          K]
            
    #        t1Patch_unrotated=t1Patch_unrotated.astype('float64')
    #        t1Patch_rotated=rotate(t1Patch_unrotated,a,reshape=False,mode='nearest')
    #        t1Patch_rotated=t1Patch_rotated.astype('float16')
            
            
    #        t1Patches[i+6*num_patches,:,:,c] = t1Patch_rotated
            # flairPatches[i, :, :, c] = invols[c][I - dsize[0]: I + dsize[0],
              #                                   J - dsize[1]: J + dsize[1],
               #                                  K]

    #    '''
    #    MaskPatches[i, :, :, 0] = mask[I - dsize[0]: I + dsize[0] + 1,
    #                                   J - dsize[1]:J + dsize[1] + 1,
    #                                   K]
    #    '''
        # trying even-sided patches
        # First half of patches have lesions and second half do not (at the center)
    #    MaskPatch_unrotated = mask[I - dsize[0]: I + dsize[0],
    #                                    J - dsize[1]:J + dsize[1],
    #                                    K]
        
    #    MaskPatch_unrotated=MaskPatch_unrotated.astype('float64')
    #    MaskPatch_rotated=rotate(MaskPatch_unrotated,a,reshape=False,mode='nearest')
    #    MaskPatch_rotated=MaskPatch_rotated.astype('float16')
        
        #for m in range(0,48):
         #   for n in range(0,48):
          #      if MaskPatch_rotated[m,n]>0.55:
           #         MaskPatch_rotated[m,n]=1
            #    else:
             #       MaskPatch_rotated[m,n]=0
                        
        
        
    #    MaskPatches[i+6*num_patches,:,:,0] = MaskPatch_rotated

    t1Patches = np.asarray(t1Patches, dtype=np.float16)
    MaskPatches = np.asarray(MaskPatches, dtype=np.float16)

    return t1Patches, MaskPatches


def CreatePatchesForTraining(atlasdir, plane, patchsize, max_patch=150000, num_channels=1):
    '''
    Params:
        - TODO
        - healthy: bool, False if not going over the healthy dataset, true otherwise
    '''

    # get filenames
    # Adding in flair patch
    t1_names = os.listdir(atlasdir)
    flair_names = os.listdir(atlasdir)
    mask_names = os.listdir(atlasdir)

    t1_names = [x for x in t1_names if "t1_masked" in x]
    flair_names = [x for x in flair_names if "rflair" in x]
    mask_names = [x for x in mask_names if "pvs" in x]

    t1_names.sort()
    flair_names.sort()
    mask_names.sort()
    
    

    numatlas = len(t1_names)

    patchsize = np.asarray(patchsize, dtype=int)
    padsize = np.max(patchsize + 1)# / 2

    # calculate total number of voxels for all images to pre-allocate array
    f = 0
    for i in range(0, numatlas):
        maskname = mask_names[i]
        maskname = os.path.join(atlasdir, maskname)
        temp = nib.load(maskname)
        mask = temp.get_data()
        f = f + np.sum(mask)

    print("Total number of lesion patches =", f)
    total_num_patches = int(np.minimum(max_patch * numatlas, f))
    single_subject_num_patches = total_num_patches // numatlas
    print("Allowed total number of patches =", total_num_patches)

    # note here we double the size of the tensors to allow for healthy patches too
    doubled_num_patches = total_num_patches * 2
    #quad_num_patches = doubled_num_patches * 2
    #oct_num_patches = quad_num_patches * 2
    if plane == "axial":
        #t1_matsize = (oct_num_patches,
        #              patchsize[0], patchsize[1], num_channels)
        t1_matsize = (doubled_num_patches,
                      patchsize[0], patchsize[1], num_channels)
        #Mask_matsize = (oct_num_patches, patchsize[0], patchsize[1],1)
        Mask_matsize = (doubled_num_patches, patchsize[0], patchsize[1], 1)
        flair_matsize = (doubled_num_patches, patchsize[0], patchsize[1], num_channels)
    elif plane == "sagittal":
        t1_matsize = (doubled_num_patches,
                      patchsize[0], 16, num_channels)
        Mask_matsize = (doubled_num_patches, patchsize[0], 16, 1)
        flair_matsize = (doubled_num_patches, patchsize[0], 16, num_channels)
    elif plane == "coronal":
        t1_matsize = (doubled_num_patches,
                      16, patchsize[1], num_channels)
        Mask_matsize = (doubled_num_patches, 16, patchsize[1], 1)
        flair_matsize = (doubled_num_patches, 16, patchsize[1], num_channels)


    t1Patches = np.zeros(t1_matsize, dtype=np.float16)
    flairPatches = np.zeros(flair_matsize, dtype=np.float16)
    MaskPatches = np.zeros(Mask_matsize, dtype=np.float16)

    indices = [x for x in range(doubled_num_patches)]
    #indices = [x for x in range(oct_num_patches)]
    #indices = shuffle(indices, random_state=0)
    cur_idx = 0

    # interpret plane
    planar_codes = {"axial": (0, 1, 2),
                    "sagittal": (1, 2, 0),
                    "coronal": (2, 0, 1)}
    planar_code = planar_codes[plane]

    for i in tqdm(range(0, numatlas)):
        t1name = t1_names[i]
        t1name = os.path.join(atlasdir, t1name)
        flairname = flair_names[i]
        flairname = os.path.join(atlasdir, flairname)
        cur_idx2=0

        temp = nib.load(t1name)
        t1 = temp.get_data()
        t1 = np.asarray(t1, dtype=np.float16)
        
        temp = nib.load(flairname)
        flair = temp.get_data()
        flair = np.asarray(flair, dtype=np.float16)

        maskname = mask_names[i]
        maskname = os.path.join(atlasdir, maskname)
        temp = nib.load(maskname)
        mask = temp.get_data()
        mask = np.asarray(mask, dtype=np.float16)

        # here, need to ensure that the CT and mask tensors
        # are padded out to larger than the size of the requested
        # patches, to allow for patches to be gathered from edges
        t1 = PadImage(t1, padsize)
        flair = PadImage(flair, padsize)
        mask = PadImage(mask, padsize)

        t1 = np.transpose(t1, axes=planar_code)
        flair = np.transpose(flair, axes=planar_code)
        mask = np.transpose(mask, axes=planar_code)

        invols = [t1]  # can handle multichannel here

        # adjusting patch size after transpose
        if planar_code != planar_codes["axial"]:
            if t1.shape[0] < t1.shape[1]:
                patchsize = (t1.shape[0]//4, patchsize[1])
            if t1.shape[1] < t1.shape[0]:
                patchsize = (patchsize[0], t1.shape[1]//4)
        patchsize = np.asarray(patchsize, dtype=int)
        
        #n_split=5
 
        #for train_index,val_index in KFold(n_split).split(invols):
        #    t1_train,t1_val=X[train_index],X[val_index]
        #    mask_train,mask_val=Y[train_index],Y[val_index]

        t1PatchesA, MaskPatchesA = get_patches(invols,
                                               mask,
                                               patchsize,
                                               single_subject_num_patches,
                                               num_channels,)

        t1PatchesA = np.asarray(t1PatchesA, dtype=np.float16)
        MaskPatchesA = np.asarray(MaskPatchesA, dtype=np.float16)

        for t1_patch, mask_patch in zip(t1PatchesA, MaskPatchesA):
            t1Patches[indices[cur_idx], :, :, :] = t1_patch
            MaskPatches[indices[cur_idx], :, :, :] = mask_patch
            cur_idx += 1
            cur_idx2 += 1
            
        n_split=5
        les_per_im[i]=cur_idx2

    return (t1Patches, MaskPatches, les_per_im)
