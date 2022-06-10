### TEST FILE ###
import glob
import os
import tensorflow as tf
import numpy as np
import nibabel as nib
import torchio as tio
import json
import torch as t
import torch.nn as nn
from matplotlib import cm
from matplotlib import image as im
from cachetools import func
from keras.models import load_model
import yaml
from matplotlib import pyplot as plt
import scipy.stats as st
from loss_functions import gen_dice_coef, gen_dice_coef_loss, gen_dice_fhnw_coef, gen_dice_fhnw_coef_loss, \
    dropout_loss, dice_coef_git_all, dice_coef_git
from keras import backend as K
from skimage import color
from skimage import io

def reshape(img, lab):
    img = tf.expand_dims(input=img, axis=3)
    lab = tf.expand_dims(input=lab, axis=3)
    lab = np.squeeze(lab, axis=3)
    lab = tf.one_hot(lab, 3)
    img = tf.expand_dims(input=img, axis=0)
    return img, lab


def pad_with_zeros(img, lab):
    final_shape = (imshape[0], imshape[1], imshape[2])
    x = imshape[0]
    y = imshape[1]
    z = imshape[2]
    if (img.shape[0] < imshape[0]):
        x = img.shape[0]
    if (img.shape[1] < imshape[1]):
        y = img.shape[1]
    if (img.shape[2] < imshape[2]):
        z = img.shape[2]
    img_zero = np.zeros(final_shape)
    #print(img.shape)
    img_zero[0:x, 0:y, 0:z] = img[0:x, 0:y, 0:z]
    lab_zero = np.zeros(final_shape)
    #print(lab.shape)
    lab_zero[0:x, 0:y, 0:z] = lab[0:x, 0:y, 0:z]
    return img_zero, lab_zero


def numpy_reader(path):
    data = np.load(path).as_type(np.float32)

    affine = np.eye(4)

    return data, affine

def dice_pair_calc(pred_stk, nb_samples):
    dice = []
    for i in range(nb_samples - 1):
            for j in range(nb_samples-i):
                if j!=0:
                    dice.append(dice_coef_git(pred_stk[i,:], pred_stk[j+i,:]))
    dice_mean = np.mean(np.asarray(dice))
    return dice_mean

def dice_within_calc(pred_stk,mean,nb_samples):
    dice = []
    for i in range(nb_samples):
         dice.append(dice_coef_git(mean[:,:,:,:], pred_stk[i,:,:,:,:]))
    dice_mean = np.mean(np.asarray(dice))
    return dice_mean


def uncertainty_predictions(pred_stack, nb):
    var = tf.math.reduce_variance(pred_stack, axis=0)
    mean_var = tf.math.reduce_mean(var)
    pred_std = np.std(pred_stack, axis=0)
    mean_std = np.mean(pred_std)
    dice_p = dice_pair_calc(pred_stack, 5)
    pred_mean = np.mean(pred_stack, axis=0)
    dice_w = dice_within_calc(pred_stack, pred_mean, 5)

    return mean_var, mean_std, dice_w, dice_p



############### LOAD DATA  ###################


with open("/media/m-ssd2/jaye/fetal_lung_segmentation/Code/3d_UNet/3d_model/config_test_3d.yaml", "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


# CHOOSE WHICH LABEL SHOULD BE PREDICTED 
label_id = config["label"]

imshape = (config["width"], config["height"], config["depth"])

pred_path = config["prediction_path"]

test_imgs = config["test_imgs"]
test_segs = config["test_segs"]

ipaths = sorted(glob.glob(test_imgs))
spaths = sorted(glob.glob(test_segs))

dice_array = []
dice_p_array = []
dice_w_array = []
mean_var_arr = []
mean_std_arr = []
bs_l_arr = []
bs_r_arr = []
cv_arr = []
cv_arr = []
volumes = []
prec_arr = []

path = config["saved_model_path"]
if os.path.exists(path):
    model = load_model(path, compile=False)
    model.summary()
else:
    print('Failed to load existing model at: {}'.format(path))

n_testimages = len(ipaths)
for i in range(n_testimages):
    mean_var = []
    mean_std = []
    dice_p_arr = []
    dice_w_arr =  []
    bs_l =  []
    bs_r =  []
    cv_st = []

    img_file = glob.glob(ipaths[i])[0]
    img = nib.load(img_file).get_fdata()
    lab_file = glob.glob(spaths[i])[0]
    lab = nib.load(lab_file).get_fdata()

    subject_a = tio.Subject(
        img_=tio.ScalarImage(img_file, ),
        label_=tio.LabelMap(lab_file, )
    )
    """
    IF PATCHING IS BIGGER THAN CERTAIN IMAGE DIMENSION PAD WITH ZEROS

    transform = tio.CropOrPad((256, 256, 256))
    subject_a = transform(subject_a)
    """

    lab = subject_a['label_'][tio.DATA]
    lb = tf.one_hot(lab,3)
    units = np.count_nonzero(K.flatten(lb[...,label_id]))
    volume_lb = units * 0.8 * 0.8 * 0.8

    sampler = tio.GridSampler(subject=subject_a, patch_size=64)
    patch_loader = t.utils.data.DataLoader(sampler, batch_size=1)
    aggregator = tio.inference.GridAggregator(sampler)
    nbr_patches = len(sampler)
    patch_counter = 0

    with t.no_grad():

        for patches_batch in patch_loader:
            img = patches_batch['img_'][tio.DATA]
            lab = patches_batch['label_'][tio.DATA]
            locations = patches_batch[tio.LOCATION]
            img = tf.squeeze(img, axis=0)
            lab = tf.squeeze(lab, axis=0)
            img = tf.squeeze(img, axis=0)
            lab = tf.squeeze(lab, axis=0)
            img = np.asarray(img)
            lab = np.asarray(lab)
            img, lab = reshape(img, lab)

            # PREDICTION
            pred = model(img, training=False)
            pred = np.where(pred < 0.5, 0, 1)

            # DROPOUT SAMPLES
            pred_stack = []
            for q in range(5):
                pred = model(img, training=False)
                pred_stack.append(pred)
            pred_stack = np.asarray(pred_stack)

            # UNCERTAINTY COMPUTATIONS
            var, std, dice_w, dice_pair = uncertainty_predictions(pred_stack, 5)
            mean_var.append(var)
            mean_std.append(std)
            dice_p_arr.append(dice_pair)
            dice_w_arr.append(dice_w)

            pred = tf.expand_dims(pred, axis=0)
            outputs = np.asarray(pred)
            aggregator.add_batch(t.from_numpy(outputs[:,:,:,:,:,label_id]), locations)
            patch_counter += 1
    output_tensor = aggregator.get_output_tensor()
    output_tensor = tf.squeeze(output_tensor, axis= 0)

    output_save = np.float32(output_tensor)
    output_nifti = nib.Nifti1Image(output_save, np.eye(4))
    nib.save(output_nifti, str(pred_path) + "pred_" + str(label_id) +"_"+ str(i*nbr_patches + patch_counter) +"_.nii.gz")
   

    # VOLUME CALCULATION LEFT/RIGHT LUNG
    units = np.count_nonzero(K.flatten(output_tensor))
    volume = units * 0.8 * 0.8 * 0.8
    volumes.append(volume_lb)
    diff = abs((volume_lb - volume))
    prec = (100 * diff)/volume_lb
    prec_arr.append(prec)

    dice_p_array.append(np.mean(dice_p_arr))
    dice_w_array.append(np.mean(dice_w_arr))
    mean_var_arr.append(np.mean(mean_var))
    mean_std_arr.append(np.mean(mean_std))
    dice_array.append(float(dice_coef_git_all(lb[...,label_id], tf.cast(output_tensor, tf.float32))))

    

with open("dice_values.txt", 'w') as f:
    for item in dice_array:
        f.write("%s\n" % float(item)) 

with open("mean_var.txt", 'w') as f:
    for item in mean_var_arr:
        f.write("%s\n" % float(item)) 
 
with open("mean_std.txt", 'w') as f:
    for item in mean_std_arr:
        f.write("%s\n" % float(item)) 


with open("dice_pair.txt", 'w') as f:
    for item in dice_p_array:
        f.write("%s\n" % float(item))


with open("dice_within.txt", 'w') as f:
    for item in dice_w_array:
        f.write("%s\n" % float(item))






