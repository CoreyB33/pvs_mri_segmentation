import os
import numpy as np
import nibabel as nib
from subprocess import Popen, PIPE

from sklearn import metrics
import utils
#from utils import preprocess
import save_figures
import apply_model
import pad
from save_figures import *
from apply_model import apply_model_3d
from apply_model import apply_model
from apply_model import apply_triplanar_models
from pad import pad_image
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import losses
from losses import *

os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'

if __name__ == "__main__":

    ######################## COMMAND LINE ARGUMENTS ########################
    results = utils.parse_args("validate")
    DATA_DIR = results.VAL_DIR
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

    thresh = results.threshold
    if DATA_DIR.split(os.sep)[1] == "test":
        dir_tag = open("host_id.cfg").read().split()[
            0] + "_" + DATA_DIR.split(os.sep)[1]
    else:
        dir_tag = DATA_DIR.split(os.sep)[1]
    experiment_name = os.path.basename(model_filename)[:os.path.basename(model_filename)
                                                       .find("_weights")] + "_" + dir_tag

    utils.save_args_to_csv(results, os.path.join("results", experiment_name))

    ######################## PREPROCESS TESTING DATA ########################
    SKULLSTRIP_SCRIPT_PATH = os.path.join("utils", "CT_BET.sh")

    PREPROCESSING_DIR = os.path.join(DATA_DIR, "preprocessed")
    SEG_ROOT_DIR = os.path.join(DATA_DIR, "segmentations")
    STATS_DIR = os.path.join("results", experiment_name)
    FIGURES_DIR = os.path.join("results", experiment_name, "figures")
    SEG_DIR = os.path.join(SEG_ROOT_DIR, experiment_name)
    REORIENT_DIR = os.path.join(SEG_DIR, "reoriented")

    for d in [PREPROCESSING_DIR, SEG_ROOT_DIR, STATS_DIR, SEG_DIR, REORIENT_DIR, FIGURES_DIR]:
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

    # preprocess.preprocess_dir(DATA_DIR,
      #                        PREPROCESSING_DIR,
       #                       SKULLSTRIP_SCRIPT_PATH)

    ######################## SEGMENT FILES ########################
    filenames = [x for x in os.listdir(DATA_DIR)
                 if not os.path.isdir(os.path.join(DATA_DIR, x))]
    masks = [x for x in filenames if "pvs" in x]
    # Using 4D file instead of just t1, if using just t1, use "t1" instead of "multi"
    filenames_t1 = [x for x in filenames if "t1_wm_masked" in x]
    filenames_flair = [x for x in filenames if "output_wmh" in x]
    

    filenames_t1.sort()
    filenames_flair.sort()
    masks.sort()

    if len(filenames_t1) != len(masks):
        print("Error, file missing. #t1:{}, #masks:{}".format(
            len(filenames_t1), len(masks)))

    print("Using model:", model_filename)

    # used only for printing result
    mean_dice = 0
    pred_vols = []
    gt_vols = []

    roc_aucs = []
    precision_scores = []

    for filename_t1, filename_flair, mask in zip(filenames_t1, filenames_flair, masks):
        # load nifti file data
        nii_obj_t1 = nib.load(os.path.join(DATA_DIR, filename_t1))
        nii_img_t1 = nii_obj_t1.get_data()
        header = nii_obj_t1.header
        affine = nii_obj_t1.affine
        
        nii_obj_flair = nib.load(os.path.join(DATA_DIR, filename_flair))
        nii_img_flair = nii_obj_flair.get_data()
        header_flair = nii_obj_flair.header
        affine_flair = nii_obj_flair.affine

        #print("nii image shape = {}".format(nii_img.shape))
        
        # load mask file data
        mask_obj = nib.load(os.path.join(DATA_DIR, mask))
        mask_img = mask_obj.get_data()

        # pad and reshape to account for implicit "1" channel
        # Uncomment if using just t1
        #nii_img = np.reshape(nii_img, nii_img.shape + (1,))
        
        nii_img = np.zeros((256,256,176,2), dtype=np.float16)
        
        
        orig_shape = nii_img.shape

        print("nii img shape from test.py= {}".format(nii_img.shape))
        
        # if the mask is larger, pad to hardcoded value 
        if mask_img.shape[0] > nii_img_t1.shape[0] or mask_img.shape[1] > nii_img_t1.shape[1]:
            TARGET_DIMS = (656,656,96)
            nii_img_t1 = pad_image(nii_img_t1, target_dims=TARGET_DIMS)
            nii_img_flair = pad_image(nii_img_flair, target_dims=TARGET_DIMS)
            mask_img = pad_image(mask_img, target_dims=TARGET_DIMS)
        else: # otherwise pad normally
            nii_img_t1 = pad_image(nii_img_t1)
            nii_img_flair = pad_image(nii_img_flair)
            mask_img = pad_image(mask_img, target_dims=nii_img.shape[:3])
        
        nii_img[:,:,:,0] = nii_img_t1
        nii_img[:,:,:,1] = nii_img_flair
        
        print("nii_img shape: {}".format(nii_img.shape))
        print("mask_img shape: {}".format(mask_img.shape))  
            
        test_input=os.path.join(SEG_DIR, "test_input" + filename_t1)
        test_input_nii_obj=nib.Nifti1Image(nii_img, affine=affine, header=header)
        nib.save(test_input_nii_obj, test_input)
        
        test_input_mask=os.path.join(SEG_DIR, "test_input" + mask)
        test_input_mask_nii_obj=nib.Nifti1Image(mask_img, affine=affine, header=header)
        nib.save(test_input_mask_nii_obj, test_input_mask)
        
        # segment (everything looks good up to here)
        segmented_img = apply_model_3d(nii_img, model)
        pred_shape = segmented_img.shape

        print("segmented_img hi shape: {}".format(segmented_img.shape))

        # create nii obj
        segmented_nii_obj = nib.Nifti1Image(
            segmented_img, affine=affine, header=header)
        
        nib.save(segmented_nii_obj, os.path.join(SEG_DIR, "seg_test_" + filename_t1))
        
        # Consider commenting out everything after this and creating custom script to edit images (delete anything outside of brain tissue, and anything lying in white matter lesion)
        # Then calculate dice

        mask_obj = nib.Nifti1Image(
            mask_img, affine=mask_obj.affine, header=mask_obj.header)
        
        # write statistics to file
        print("Collecting stats...")
        cur_vol_dice, cur_slices_dice, cur_vol, cur_vol_gt = utils.write_stats(filename_t1,
                                                                               segmented_nii_obj,
                                                                               mask_obj,
                                                                               STATS_FILE,
                                                                               thresh,)


        # crop off the padding if necessary
        diff_num_slices = int(np.abs(pred_shape[-1]-orig_shape[-2])/2)
        print("Mask Img: {}".format(mask_img.shape))
        print("Orig shape: {}".format(orig_shape))
        print("Pred shape: {}".format(pred_shape))
        print("Slice diff: {}".format(diff_num_slices))

        if int(np.abs(pred_shape[-1] - orig_shape[-2])) > 0:
            diff_num_slices = int(np.abs(pred_shape[-1]-orig_shape[-2])/2)
            print("New shape: {}".format(segmented_img[:, :, diff_num_slices:-diff_num_slices].shape
                                         ))
            segmented_img = segmented_img[:, :,
                                          diff_num_slices:-diff_num_slices]

        print("Segmented_img shape: {}".format(segmented_img.shape))
        # save resultant image
        segmented_filename = os.path.join(SEG_DIR, filename_t1)
        segmented_nii_obj = nib.Nifti1Image(
            segmented_img, affine=affine, header=header)
        nib.save(segmented_nii_obj, segmented_filename)

        utils.write_dice_scores(filename_t1, cur_vol_dice,
                                cur_slices_dice, DICE_METRICS_FILE)

        mean_dice += cur_vol_dice
        pred_vols.append(cur_vol)
        gt_vols.append(cur_vol_gt)

        # Reorient back to original before comparisons
        # print("Reorienting...")
        # utils.reorient(filename, DATA_DIR, SEG_DIR)

        # get probability volumes and threshold image
        print("Thresholding...")
        utils.threshold(filename_t1, SEG_DIR, SEG_DIR, thresh)

    mean_dice = mean_dice / len(filenames_t1)
    pred_vols = np.array(pred_vols)
    gt_vols = np.array(gt_vols)
    corr = np.corrcoef(pred_vols, gt_vols)[0, 1]
    print("*** Segmentation complete. ***")
    print("Mean DICE: {:.3f}".format(mean_dice))
    print("Volume Correlation: {:.3f}".format(corr))

    # save these two numbers to file
    metrics_path = os.path.join(STATS_DIR, "metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("Dice: {:.4f}\nVolume Correlation: {:.4f}\n".format(
            mean_dice,
            corr))

    K.clear_session()
