'''
Author: Samuel Remedios

Use of this script involves:
    - set SRC_DIR to point to the directory holding all images
    - ensure this script sits at the top level in directory, alongside data/

Input images should simply be the raw CT scans.

'''
import os
import numpy as np
import nibabel as nib
import shutil
import utils
#from utils import preprocess
from save_figures import *
from apply_model import apply_model_3d
from pad import pad_image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from losses import *

os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'

if __name__ == "__main__":

    ######################## COMMAND LINE ARGUMENTS ########################
    results = utils.parse_args("multiseg")
    num_channels = results.num_channels

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

    model_filename = results.weights

    experiment_name = model_filename.split(os.sep)[-2]
    utils.save_args_to_csv(results, os.path.join("results", experiment_name))

    ######################## FOLDER SETUP ########################
    SKULLSTRIP_SCRIPT_PATH = os.path.join("utils", "CT_BET.sh")

    DATA_DIR = results.DATA_DIR

    PREPROCESSING_DIR = os.path.join(DATA_DIR, "preprocessed")
    SEG_ROOT_DIR = os.path.join(DATA_DIR, "segmentations")
    STATS_DIR = os.path.join("results", experiment_name)
    FIGURES_DIR = os.path.join("results", experiment_name, "figures")
    SEG_DIR = os.path.join(SEG_ROOT_DIR, experiment_name)
    REORIENT_DIR = os.path.join(SEG_DIR, "reoriented")
    TMPDIR = os.path.join(
        PREPROCESSING_DIR, "tmp_intermediate_preprocessing_steps")

    for d in [PREPROCESSING_DIR,
              SEG_ROOT_DIR,
              STATS_DIR,
              SEG_DIR,
              REORIENT_DIR,
              FIGURES_DIR,
              TMPDIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    # Stats file
    stat_filename = "result_" + experiment_name + ".csv"
    STATS_FILE = os.path.join(STATS_DIR, stat_filename)
    DICE_METRICS_FILE = os.path.join(
        STATS_DIR, "detailed_dice_" + experiment_name + ".csv")

    ######################## LOAD MODEL ########################
    model = load_model(model_filename,
                             custom_objects=custom_losses)

    ######################## PREPROCESSING ########################

    filenames = [x for x in os.listdir(DATA_DIR)
                 if not os.path.isdir((os.path.join(DATA_DIR, x)))]
                 
    filenames.sort()
    
    #preprocess.preprocess_dir(DATA_DIR,
     #                         PREPROCESSING_DIR,
      #                        SKULLSTRIP_SCRIPT_PATH)

    ######################## SEGMENT FILE ########################

    filenames = [x for x in os.listdir(DATA_DIR)
                 if not os.path.isdir((os.path.join(DATA_DIR, x)))]
    filenames_t1 = [x for x in filenames if "t1_wm_masked" in x]
    filenames_flair = [x for x in filenames if "output_wmh" in x]
    
    filenames_t1.sort()
    filenames_flair.sort()

    for filename_t1, filename_flair in zip(filenames_t1, filenames_flair):
        # load nifti file data
        nii_obj_t1 = nib.load(os.path.join(DATA_DIR, filename_t1))
        nii_img_t1 = nii_obj_t1.get_data()
        header = nii_obj_t1.header
        affine = nii_obj_t1.affine
        
        nii_obj_flair = nib.load(os.path.join(DATA_DIR, filename_flair))
        nii_img_flair = nii_obj_flair.get_data()
        header_flair = nii_obj_flair.header
        affine_flair = nii_obj_flair.affine

        # reshape to account for implicit "1" channel
        #nii_img = np.reshape(nii_img, nii_img.shape + (1,))
        
        nii_img = np.zeros((256,256,170,2), dtype=np.float16)
        nii_img[:,:,:,0] = nii_img_t1
        nii_img[:,:,:,1] = nii_img_flair
        
        nii_img = pad_image(nii_img)
        
        seg_output=os.path.join(SEG_DIR, "seg_output_" + filename_t1)
        seg_output_nii_obj=nib.Nifti1Image(nii_img[:,:,:,0], affine=affine, header=header)
        nib.save(seg_output_nii_obj, seg_output)
        
        seg_output=os.path.join(SEG_DIR, "seg_output_" + filename_flair)
        seg_output_nii_obj=nib.Nifti1Image(nii_img[:,:,:,1], affine=affine, header=header)
        nib.save(seg_output_nii_obj, seg_output)

        # segment
        segmented_img = apply_model_3d(nii_img, model)
        print("T1 Output shape: {}".format(nii_img.shape))
        print("Segmented Output shape: {}".format(segmented_img.shape))
        
        
        # save resultant image
        segmented_filename = os.path.join(SEG_DIR, filename_t1)
        segmented_nii_obj = nib.Nifti1Image(
            segmented_img, affine=affine, header=header)
        nib.save(segmented_nii_obj, segmented_filename)
        
        

        # Reorient back to original before comparisons
        #print("Reorienting...")
        #utils.reorient(filename, DATA_DIR, SEG_DIR)

        # get probability volumes and threshold image
        print("Thresholding...")
        utils.threshold(filename_t1, SEG_DIR, SEG_DIR, 0.5)

    if os.path.exists(TMPDIR):
        shutil.rmtree(TMPDIR)

    K.clear_session()
