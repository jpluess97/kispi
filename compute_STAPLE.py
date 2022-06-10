import nibabel as nib
import numpy as np
import os
import glob
import SimpleITK as sitk 
import tensorflow as tf
import PIL.Image
from loss_functions import gen_dice_coef, gen_dice_coef_loss, gen_dice_fhnw_coef, gen_dice_fhnw_coef_loss, \
    dropout_loss, dice_coef_git_all, dice_coef_git



"""
THE COMPUTATION OF THE STAPLE HAS TO BE DONE SEPARATELY FOR EACH LABEL THE FOLLOWING CALCULATIONS ARE DONE FOR TWO LABELS AND ONE BACKGROUND CLASS
"""
# avoids tensorflow uses all GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)




### Fill In current_dir
training_dir = '/media/m-ssd2/jaye/fetal_lung_segmentation/Code/training_ganz/'

scan_ids = []

n_annotators = 3

annotator_names = ["JG", "RK", "MZ"]



for file in os.listdir(training_dir):
    if 'label' not in file:
        scan_ids.append(file)


volumes = []
count = 0

for scan in sorted(scan_ids):
    print(scan)
    count += 1
    seg_stack_lab1 = []
    seg_stack_lab2 = []
    seg_stack = []
    stack_annot = []
    stack_oh = []
    img_file = glob.glob(training_dir + scan)[0]
    img = nib.load(img_file).get_fdata()
    for annotators in range(n_annotators):
        label_file = glob.glob(training_dir + scan[:-7] + "-label_" + annotator_names[annotators] + "_proc.nii.gz")[0]
        label = nib.load(label_file).get_fdata()
        label_ = nib.load(label_file)
        label__ = np.copy(label)
        seg_one_hot = tf.one_hot(label__, 3)
        seg_ =  np.copy(seg_one_hot[:,:,:,0:2])
        index_one = tf.argmax(seg_, axis=3)
        index_one =np.asarray(index_one)
        seg_b = seg_one_hot[:, :, :, 0]
        seg_r = seg_one_hot[:, :, :, 2]
        seg_b = tf.expand_dims(input=seg_b, axis=3)
        seg_r = tf.expand_dims(input=seg_r, axis=3)
        seg_single = np.concatenate((seg_b, seg_r), axis=3)
        index_two = tf.argmax(seg_single, axis=3)
        index_two = np.asarray(index_two)
        seg_one = sitk.GetImageFromArray(index_one.astype(np.int16))
        seg_two = sitk.GetImageFromArray(index_two.astype(np.int16))
        seg_stack_lab1.append(seg_one)
        stack_annot.append(label)
        stack_oh.append(seg_one_hot)
   
    
    seg_stack = [seg_stack_lab1, seg_stack_lab2]

    indexes_lab = []
    indexes_lab = np.where(label > 0)
    # SEPARATE STAPLE COMPUTATIONS FOR EACH LABEL
    STAPLE_seg_one = sitk.STAPLE(seg_stack_lab1, 1.0)
    STAPLE_seg_two = sitk.STAPLE(seg_stack_lab2, 2.0)
    label_1 = sitk.GetArrayFromImage(STAPLE_seg_one)
    label_1 = np.where(label_1 < 0.5, 0, 1)
    indexes_1 = []
    indexes_1 = np.where(label_1 > 0)
    label_2 = sitk.GetArrayFromImage(STAPLE_seg_two)
    label_2 = np.where(label_2 < 0.1, 0, 2)
    indexes_2 = []
    indexes_2 = np.where(label_2 > 0)
    # CONCATENATE BOTH LABELS
    lab = np.add(label_1, label_2)
    indexes = []
    indexes = np.where(lab > 0)
    # SAVE STAPLE AS NIFTI FILE
    ni_img = nib.Nifti1Image(lab, label_.affine)
    nib.save(ni_img, scan)
    # VOLUME CALCULATIONS
    lab_oh = tf.one_hot(lab,3)
    units = np.count_nonzero(lab_oh[...,2])
    volume = units * 0.8 * 0.8 * 0.8
    volumes.append(volume)



