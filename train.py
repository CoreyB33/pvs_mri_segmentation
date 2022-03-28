import json
import numpy as np
import os
from subprocess import Popen, PIPE
import sys
from sklearn.model_selection import KFold

import utils
import patch_ops
import tensorflow as tf
import sklearn
import matplotlib.pyplot as plt


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
    
    METRICS = [ 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      dice_coef,]

    ######### MODEL AND CALLBACKS #########
    if not model:
        model = unet(model_path=MODEL_PATH,
                          num_channels=num_channels,
                          loss=dice_coef_loss,
                          ds=2,
                          lr=learning_rate,
                          num_gpus=NUM_GPUS,
                          verbose=1,
                          metrics=METRICS,)
    else:
        print("Continuing training with", model)
        model = load_model(model, custom_objects=custom_losses)

    monitor = "loss"

    # checkpoints
    checkpoint_filename = str(start_time) +\
        "_epoch_{epoch:04d}_" +\
        monitor+"_{"+monitor+":.4f}_weights.hdf5"

    checkpoint_filename = os.path.join(WEIGHT_DIR, checkpoint_filename)
    checkpoint = ModelCheckpoint(checkpoint_filename,
                                 monitor='loss',
                                 save_best_only=True,
                                 mode='auto',
                                 verbose=0,
                                 metrics=METRICS,)

    # tensorboard
    tb = TensorBoard(log_dir=TB_LOG_DIR)

    # early stopping
    es = EarlyStopping(monitor="loss",
                       min_delta=1e-4,
                       patience=10,
                       verbose=1,
                       mode='auto')

    callbacks_list = [checkpoint, tb, es]

    ######### DATA IMPORT #########
    DATA_DIR = os.path.join("data", "train_conservative_sensitivity_normalized")
    
    t1_patches, mask_patches = patch_ops.CreatePatchesForTraining(
        atlasdir=DATA_DIR,
        plane=plane,
        patchsize=PATCH_SIZE,
        max_patch=num_patches,
        num_channels=num_channels)

    print("Individual patch dimensions:", t1_patches[0].shape)
    print("Num patches:", len(t1_patches))
    print("t1_patches shape: {}\nmask_patches shape: {}".format(
        t1_patches.shape, mask_patches.shape))


    ######### TRAINING #########
    history = model.fit(t1_patches,
                        mask_patches,
                        batch_size=batch_size,
                        epochs=num_epochs,
                        verbose=1,
                        validation_split=0.2,
                        callbacks=callbacks_list,)

    with open(HISTORY_PATH, 'w') as f:
        json.dump(str(history.history), f)
        
    #def plot_roc(name, labels, predictions, **kwargs):
    #    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

#        plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
 #       plt.xlabel('False positives [%]')
  #      plt.ylabel('True positives [%]')
   #     plt.xlim([-0.5,20])
    #    plt.ylim([80,100.5])
     #   plt.grid(True)
      #  ax = plt.gca()
       # ax.set_aspect('equal')
        
    #plot_roc("Train Baseline", y_true, y_pred, color=colors[0])
    #print(history)
    #tp=history.history['tp']
    #tn=history.history['tn']
    #fp=history.history['fp']
    #fn=history.history['fn']
    #print(tp[1])
    #sensitivity=[]
    #specificity=[]
    
    #for i in range(len(tp)):
    #    sens=tp[i]/(tp[i]+fn[i])
    #    spec=tn[i]/(tn[i]+fp[i])
    #    spec_x=1-spec
    #    sensitivity.append(sens)
    #    specificity.append(spec_x)
    
    #print(sensitivity)
    #print(specificity)
    #sensitivity=history.history['tp']/(history.history['tp']+history.history['fn'])
    #print(sensitivity)   
    #train_labels=history.history
    #plt.plot(specificity, sensitivity)
    #plt.show()
    
    #file1 = open("myfile.txt","w")
    #file1.writelines('sensitivity:')
    #file1.writelines(str(sensitivity))
    #file1.writelines('100-specificity:')
    #file1.writelines(str(specificity))
    #file1.close()

    K.clear_session()
