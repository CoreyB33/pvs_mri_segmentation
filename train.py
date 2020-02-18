import json
import numpy as np
import os
from subprocess import Popen, PIPE
import sys
from sklearn.model_selection import KFold

import utils
import patch_ops
import tensorflow as tf


from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from multi_gpu import ModelMGPU
from losses import *
from unet import unet

os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":

    results = utils.parse_args("train")

    NUM_GPUS = 1
    if results.GPUID == None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif results.GPUID == -1:
        # find maximum number of available GPUs
        call = "nvidia-smi --list-gpus"
        pipe = Popen(call, shell=True, stdout=PIPE).stdout
        available_gpus = pipe.read().decode().splitlines()
        NUM_GPUS = len(available_gpus)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(results.GPUID)

    num_channels = results.num_channels
    plane = results.plane
    num_epochs = 1000000
    num_patches = results.num_patches
    batch_size = results.batch_size
    model = results.model
    model_architecture = "unet"
    start_time = utils.now()
    experiment_details = start_time + "_" + model_architecture + "_" +\
            results.experiment_details
    loss = results.loss
    learning_rate = 1e-4

    utils.save_args_to_csv(results, os.path.join("results", experiment_details))

    WEIGHT_DIR = os.path.join("models", "weights", experiment_details)
    TB_LOG_DIR = os.path.join("models", "tensorboard", start_time)

    MODEL_NAME = model_architecture + "_model_" + experiment_details
    MODEL_PATH = os.path.join(WEIGHT_DIR, MODEL_NAME + ".json")

    HISTORY_PATH = os.path.join(WEIGHT_DIR, MODEL_NAME + "_history.json")

    # files and paths
    TRAIN_DIR = results.SRC_DIR

    for d in [WEIGHT_DIR, TB_LOG_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    PATCH_SIZE = [int(x) for x in results.patch_size.split("x")]

    ######### MODEL AND CALLBACKS #########
    if not model:
        model = unet(model_path=MODEL_PATH,
                          num_channels=num_channels,
                          loss=dice_coef_loss,
                          ds=2,
                          lr=learning_rate,
                          num_gpus=NUM_GPUS,
                          verbose=1,)
    else:
        print("Continuing training with", model)
        model = load_model(model, custom_objects=custom_losses)

    monitor = "val_dice_coef"

    # checkpoints
    checkpoint_filename = str(start_time) +\
        "_epoch_{epoch:04d}_" +\
        monitor+"_{"+monitor+":.4f}_weights.hdf5"

    checkpoint_filename = os.path.join(WEIGHT_DIR, checkpoint_filename)
    checkpoint = ModelCheckpoint(checkpoint_filename,
                                 monitor='val_loss',
                                 save_best_only=True,
                                 mode='auto',
                                 verbose=0,)

    # tensorboard
    tb = TensorBoard(log_dir=TB_LOG_DIR)

    # early stopping
    es = EarlyStopping(monitor="val_loss",
                       min_delta=1e-4,
                       patience=10,
                       verbose=1,
                       mode='auto')

    callbacks_list = [checkpoint, tb, es]

    ######### DATA IMPORT #########
    DATA_DIR = os.path.join("data", "train")
    
    t1_patches, mask_patches, les_per_im = patch_ops.CreatePatchesForTraining(
        atlasdir=DATA_DIR,
        plane=plane,
        patchsize=PATCH_SIZE,
        max_patch=num_patches,
        num_channels=num_channels)
    
    les_per_im=les_per_im.astype('int32')

    print("Individual patch dimensions:", t1_patches[0].shape)
    print("Num patches:", len(t1_patches))
    print("t1_patches shape: {}\nmask_patches shape: {}".format(
        t1_patches.shape, mask_patches.shape))
    
    t1im1 = t1Patches[np.arange(0,les_per_im[0]-1),:,:,:]
    base=les_per_im[0]
    t1im2 = t1Patches[np.arange(base,base+les_per_im[1]-1),:,:,:]
    base=base+les_per_im[1]-1
    t1im3 = t1Patches[np.arange(base,base+les_per_im[2]-1),:,:,:]
    base=base+les_per_im[2]-1
    t1im4 = t1Patches[np.arange(base,base+les_per_im[3]-1),:,:,:]
    base=base+les_per_im[3]-1
    t1im5 = t1Patches[np.arange(base,base+les_per_im[4]-1),:,:,:]
    base=base+les_per_im[4]-1
    t1im6 = t1Patches[np.arange(base,base+les_per_im[5]-1),:,:,:]
    base=base+les_per_im[5]-1
    t1im7 = t1Patches[np.arange(base,base+les_per_im[6]-1),:,:,:]
    base=base+les_per_im[6]-1
    t1im8 = t1Patches[np.arange(base,base+les_per_im[7]-1),:,:,:]
    base=base+les_per_im[7]-1
    t1im9 = t1Patches[np.arange(base,base+les_per_im[8]-1),:,:,:]
    base=base+les_per_im[8]-1
    t1im10 = t1Patches[np.arange(base,base+les_per_im[9]-1),:,:,:]
    base=base+les_per_im[9]-1
    t1im11 = t1Patches[np.arange(base,base+les_per_im[10]-1),:,:,:]
    base=base+les_per_im[10]-1
    t1im12 = t1Patches[np.arange(base,base+les_per_im[11]-1),:,:,:]
    base=base+les_per_im[11]-1
    t1im13 = t1Patches[np.arange(base,base+les_per_im[12]-1),:,:,:]
    base=base+les_per_im[12]-1
    t1im14 = t1Patches[np.arange(base,base+les_per_im[13]-1),:,:,:]
    base=base+les_per_im[13]-1
    t1im15 = t1Patches[np.arange(base,base+les_per_im[14]-1),:,:,:]
    base=base+les_per_im[14]-1
    t1im16 = t1Patches[np.arange(base,base+les_per_im[15]-1),:,:,:]
    base=base+les_per_im[15]-1
    t1im17 = t1Patches[np.arange(base,base+les_per_im[16]-1),:,:,:]
    base=base+les_per_im[16]-1
    t1im18 = t1Patches[np.arange(base,base+les_per_im[17]-1),:,:,:]
    base=base+les_per_im[17]-1
    t1im19 = t1Patches[np.arange(base,base+les_per_im[18]-1),:,:,:]
    base=base+les_per_im[18]-1
    t1im20 = t1Patches[np.arange(base,base+les_per_im[19]-1),:,:,:]
    base=base+les_per_im[19]-1
    t1im21 = t1Patches[np.arange(base,base+les_per_im[20]-1),:,:,:]
    base=base+les_per_im[20]-1
    t1im22 = t1Patches[np.arange(base,base+les_per_im[21]-1),:,:,:]
    base=base+les_per_im[21]-1
    t1im23 = t1Patches[np.arange(base,base+les_per_im[22]-1),:,:,:]
    base=base+les_per_im[22]-1
    t1im24 = t1Patches[np.arange(base,base+les_per_im[23]-1),:,:,:]
    base=base+les_per_im[23]-1
    t1im25 = t1Patches[np.arange(base,base+les_per_im[24]-1),:,:,:]
    base=base+les_per_im[24]-1
    t1im26 = t1Patches[np.arange(base,base+les_per_im[25]-1),:,:,:]
    base=base+les_per_im[25]-1
    t1im27 = t1Patches[np.arange(base,base+les_per_im[26]-1),:,:,:]
    base=base+les_per_im[26]-1
    t1im28 = t1Patches[np.arange(base,base+les_per_im[27]-1),:,:,:]
    base=base+les_per_im[27]-1
    t1im29 = t1Patches[np.arange(base,base+les_per_im[28]-1),:,:,:]
    base=base+les_per_im[28]-1
    t1im30 = t1Patches[np.arange(base,base+les_per_im[29]-1),:,:,:]
    base=base+les_per_im[29]-1
    t1im31 = t1Patches[np.arange(base,base+les_per_im[30]-1),:,:,:]
    base=base+les_per_im[30]-1
    t1im32 = t1Patches[np.arange(base,base+les_per_im[31]-1),:,:,:]
    base=base+les_per_im[31]-1
    t1im33 = t1Patches[np.arange(base,base+les_per_im[32]-1),:,:,:]
    base=base+les_per_im[32]-1
    t1im34 = t1Patches[np.arange(base,base+les_per_im[33]-1),:,:,:]
    base=base+les_per_im[33]-1
    t1im35 = t1Patches[np.arange(base,base+les_per_im[34]-1),:,:,:]
    base=base+les_per_im[34]-1
    t1im36 = t1Patches[np.arange(base,base+les_per_im[35]-1),:,:,:]
    base=base+les_per_im[35]-1
    t1im37 = t1Patches[np.arange(base,base+les_per_im[36]-1),:,:,:]
    base=base+les_per_im[36]-1
    t1im38 = t1Patches[np.arange(base,base+les_per_im[37]-1),:,:,:]
    base=base+les_per_im[37]-1
    t1im39 = t1Patches[np.arange(base,base+les_per_im[38]-1),:,:,:]
    base=base+les_per_im[38]-1
    t1im40 = t1Patches[np.arange(base,base+les_per_im[39]-1),:,:,:]
    t1images=(t1im1,t1im2,t1im3,t1im4,t1im5,t1im6,t1im7,t1im8,t1im9,t1im10,t1im11,t1im12,t1im13,t1im14,t1im15,t1im16,t1im17,t1im18,t1im19,t1im20,t1im21,t1im22,t1im23,t1im24,t1im25,t1im26,t1im27,t1im28,t1im29,t1im30,t1im31,t1im32,t1im33,t1im34,t1im35,t1im36,t1im37,t1im38,t1im39,t1im40)

    maskim1 = MaskPatches[np.arange(0,les_per_im[0]-1),:,:,:]
    base2=les_per_im[0]
    maskim2 = MaskPatches[np.arange(base2,base2+les_per_im[1]-1),:,:,:]
    base2=base2+les_per_im[1]-1
    maskim3 = MaskPatches[np.arange(base2,base2+les_per_im[2]-1),:,:,:]
    base2=base2+les_per_im[2]-1
    maskim4 = MaskPatches[np.arange(base2,base2+les_per_im[3]-1),:,:,:]
    base2=base2+les_per_im[3]-1
    maskim5 = MaskPatches[np.arange(base2,base2+les_per_im[4]-1),:,:,:]
    base2=base2+les_per_im[4]-1
    maskim6 = MaskPatches[np.arange(base2,base2+les_per_im[5]-1),:,:,:]
    base2=base2+les_per_im[5]-1
    maskim7 = MaskPatches[np.arange(base2,base2+les_per_im[6]-1),:,:,:]
    base2=base2+les_per_im[6]-1
    maskim8 = MaskPatches[np.arange(base2,base2+les_per_im[7]-1),:,:,:]
    base2=base2+les_per_im[7]-1
    maskim9 = MaskPatches[np.arange(base2,base2+les_per_im[8]-1),:,:,:]
    base2=base2+les_per_im[8]-1
    maskim10 = MaskPatches[np.arange(base2,base2+les_per_im[9]-1),:,:,:]
    base2=base2+les_per_im[9]-1
    maskim11 = MaskPatches[np.arange(base2,base2+les_per_im[10]-1),:,:,:]
    base2=base2+les_per_im[10]-1
    maskim12 = MaskPatches[np.arange(base2,base2+les_per_im[11]-1),:,:,:]
    base2=base2+les_per_im[11]-1
    maskim13 = MaskPatches[np.arange(base2,base2+les_per_im[12]-1),:,:,:]
    base2=base2+les_per_im[12]-1
    maskim14 = MaskPatches[np.arange(base2,base2+les_per_im[13]-1),:,:,:]
    base2=base2+les_per_im[13]-1
    maskim15 = MaskPatches[np.arange(base2,base2+les_per_im[14]-1),:,:,:]
    base2=base2+les_per_im[14]-1
    maskim16 = MaskPatches[np.arange(base2,base2+les_per_im[15]-1),:,:,:]
    base2=base2+les_per_im[15]-1
    maskim17 = MaskPatches[np.arange(base2,base2+les_per_im[16]-1),:,:,:]
    base2=base2+les_per_im[16]-1
    maskim18 = MaskPatches[np.arange(base2,base2+les_per_im[17]-1),:,:,:]
    base2=base2+les_per_im[17]-1
    maskim19 = MaskPatches[np.arange(base2,base2+les_per_im[18]-1),:,:,:]
    base2=base2+les_per_im[18]-1
    maskim20 = MaskPatches[np.arange(base2,base2+les_per_im[19]-1),:,:,:]
    base2=base2+les_per_im[19]-1
    maskim21 = MaskPatches[np.arange(base2,base2+les_per_im[20]-1),:,:,:]
    base2=base2+les_per_im[20]-1
    maskim22 = MaskPatches[np.arange(base2,base2+les_per_im[21]-1),:,:,:]
    base2=base2+les_per_im[21]-1
    maskim23 = MaskPatches[np.arange(base2,base2+les_per_im[22]-1),:,:,:]
    base2=base2+les_per_im[22]-1
    maskim24 = MaskPatches[np.arange(base2,base2+les_per_im[23]-1),:,:,:]
    base2=base2+les_per_im[23]-1
    maskim25 = MaskPatches[np.arange(base2,base2+les_per_im[24]-1),:,:,:]
    base2=base2+les_per_im[24]-1
    maskim26 = MaskPatches[np.arange(base2,base2+les_per_im[25]-1),:,:,:]
    base2=base2+les_per_im[25]-1
    maskim27 = MaskPatches[np.arange(base2,base2+les_per_im[26]-1),:,:,:]
    base2=base2+les_per_im[26]-1
    maskim28 = MaskPatches[np.arange(base2,base2+les_per_im[27]-1),:,:,:]
    base2=base2+les_per_im[27]-1
    maskim29 = MaskPatches[np.arange(base2,base2+les_per_im[28]-1),:,:,:]
    base2=base2+les_per_im[28]-1
    maskim30 = MaskPatches[np.arange(base2,base2+les_per_im[29]-1),:,:,:]
    base2=base2+les_per_im[29]-1
    maskim31 = MaskPatches[np.arange(base2,base2+les_per_im[30]-1),:,:,:]
    base2=base2+les_per_im[30]-1
    maskim32 = MaskPatches[np.arange(base2,base2+les_per_im[31]-1),:,:,:]
    base2=base2+les_per_im[31]-1
    maskim33 = MaskPatches[np.arange(base2,base2+les_per_im[32]-1),:,:,:]
    base2=base2+les_per_im[32]-1
    maskim34 = MaskPatches[np.arange(base2,base2+les_per_im[33]-1),:,:,:]
    base2=base2+les_per_im[33]-1
    maskim35 = MaskPatches[np.arange(base2,base2+les_per_im[34]-1),:,:,:]
    base2=base2+les_per_im[34]-1
    maskim36 = MaskPatches[np.arange(base2,base2+les_per_im[35]-1),:,:,:]
    base2=base2+les_per_im[35]-1
    maskim37 = MaskPatches[np.arange(base2,base2+les_per_im[36]-1),:,:,:]
    base2=base2+les_per_im[36]-1
    maskim38 = MaskPatches[np.arange(base2,base2+les_per_im[37]-1),:,:,:]
    base2=base2+les_per_im[37]-1
    maskim39 = MaskPatches[np.arange(base2,base2+les_per_im[38]-1),:,:,:]
    base2=base2+les_per_im[38]-1
    maskim40 = MaskPatches[np.arange(base2,base2+les_per_im[39]-1),:,:,:]
    maskimages=(maskim1,maskim2,maskim3,maskim4,maskim5,maskim6,maskim7,maskim8,maskim9,maskim10,maskim11,maskim12,maskim13,maskim14,maskim15,maskim16,maskim17,maskim18,maskim19,maskim20,maskim21,maskim22,maskim23,maskim24,maskim25,maskim26,maskim27,maskim28,maskim29,maskim30,maskim31,maskim32,maskim33,maskim34,maskim35,maskim36,maskim37,maskim38,maskim39,maskim40)

    
    # need to find a way to store the amount of indices in given image
     
    for train_index,val_index in KFold(n_split).split(t1images):
        f=train_index[0]
        b=train_index[-1]+1
        f2=val_index[0]
        b2=val_index[-1]+1
        t1_train,t1_val=t1images[f:b],t1images[f2:b2]
        mask_train,mask_val=maskimages[f:b],maskimages[f2:b2]
  
    # Below two lines can be removed
 # model=create_model()
 # model.fit(x_train, y_train,epochs=20)
  
 # print('Model evaluation ',model.evaluate(x_test,y_test))

        ######### TRAINING #########
        history = model.fit(t1_patches,
                            mask_patches,
                            batch_size=batch_size,
                            epochs=num_epochs,
                            verbose=1,
                            validation_split=0.2,
                            callbacks=callbacks_list,)

        with open(HISTORY_PATH, 'w') as f:
            json.dump(history.history, f)

    K.clear_session()
