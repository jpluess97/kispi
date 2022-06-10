import glob
import random
import torchio as tio
import nibabel as nib
import numpy as np
from medpy.io import load
import tensorflow as tf
from numpy.lib import stride_tricks
from keras import backend as K
from volumentations import *
import imageio
import torchio as tio
import torch as t
from albumentations import RandomRotate90, HorizontalFlip, Compose, RandomCrop, RandomBrightnessContrast, RandomGamma, \
    Rotate, ElasticTransform
from albumentations.augmentations.transforms import GaussNoise, GlassBlur, GaussianBlur, Sharpen
from matplotlib import pyplot as plt, cm
from skimage.util import random_noise

"""
DATALOADER IS USED TO LOAD BATCHES OF DATA INTO THE NETWORK
IT CONTAINS FUNCTIONS FOR A FULLY IMPLEMENTED KERAS TRAINING LOOP AS WELL AS A FUNCTION FOR THE CUSTOM TRAINING LOOP IMPLEMENTED IN THE TRAINING FILE. 
"""


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, annot_paths, imshape=(64, 64, 64), n_classes=3,
                 batch_size=2, shuffle=True, augment_affine=False, probability_affine=0.5, salt=True, lung="left",
                 separation=False):

        self.image_paths = image_paths
        self.annot_paths = annot_paths
        self.batch_size = batch_size
        self.imshape = imshape
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment_affine = augment_affine
        self.probability_affine = probability_affine
        self.on_epoch_end()
        self.salt_ = salt
        self.side = lung
        self.separation = separation

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        image_paths = [self.image_paths[k] for k in indexes]
        annot_paths = [self.annot_paths[k] for k in indexes]

        X, y = self.__data_generation(image_paths, annot_paths)
        print(X.shape)
        return X, y

    def cutup(data, block, stride):
        sh = np.array(data.shape)
        block = np.asanyarray(block)
        stride = np.asanyarray(stride)
        nbl = (sh - block) // stride + 1
        strides = np.r_[data.strides * stride, data.strides]
        dims = np.r_[nbl, block]
        data_ = stride_tricks.as_strided(data, strides=strides, shape=dims)
        return data_

"""
DATA AUGMENTATION
"""

    def elastic(self):
        return Compose([
            ElasticTransform((0.25), interpolation=2, p=0.1),
            RandomRotate90((2), p=0.5)
        ], p=1.0)

    def randombrightnesscontrast(self):
        return Compose([
            RandomBrightnessContrast(p=0.5),
            RandomRotate90((2), p=0.2)
        ], p=1.0)

    def gaussian_noise(self):
        return Compose([
            GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            RandomRotate90((2), p=0.2)
        ], p=1.0)

    def gaussian_blurr(self):
        return Compose([
            GaussianBlur(blur_limit=(3, 7), sigma_limit=0, always_apply=False, p=0.5),
            RandomRotate90((2), p=0.2)
        ], p=1.0)

    def sharpen(self):
        return Compose([
            Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=False, p=0.5) ,
            RandomRotate90((3), p=0.2)
        ], p=1.0)

    def aug_elastic(self, img, lab):
        aug = self.elastic()
        data = {'image': img, 'mask': lab}
        aug_data = aug(**data)
        img, lab = aug_data['image'], aug_data['mask']
        return img, lab

    def aug_randombrightness(self, img, lab):
        aug = self.randombrightnesscontrast()
        data = {'image': img, "mask": lab}
        aug_data = aug(**data)
        img, lab= aug_data['image'], aug_data['mask']
        return img, lab

    def aug_gaussian_noise(self, img, lab):
        aug = self.gaussian_noise()
        data = {'image': img, 'mask': lab}
        aug_data = aug(**data)
        img, lab = aug_data['image'], aug_data['mask']
        return img, lab

    def aug_gaussian_blur(self, img, lab):
        aug = self.gaussian_noise()
        data = {'image': img, 'mask': lab}
        aug_data = aug(**data)
        img, lab = aug_data['image'], aug_data['mask']
        return img, lab
    def aug_sharpen(self, img, lab ):
        aug = self.sharpen()
        data = {'image': img, 'mask': lab}
        aug_data = aug(**data)
        img, lab = aug_data['image'], aug_data['mask']
        return img, lab

    def augmentation(self, img, lab):
        aug = self.elastic()
        data = {'image': img, 'mask': lab}
        aug_data = aug(**data)
        img, lab = aug_data['image'], aug_data['mask']
        return img, lab

    def augmentation_flip(self, img, lab):
        aug = self.elastic()
        data = {'image': img, 'mask': lab}
        aug_data = aug(**data)
        img, lab = aug_data['image'], aug_data['mask']
        return img, lab


    def reshape(self, img, lab):
        img = tf.expand_dims(input=img, axis=3)
        lab = tf.expand_dims(input=lab, axis=3)
        lab = np.squeeze(lab, axis=3)
        lab = tf.one_hot(lab, 3)
        return img, lab

    def pad_with_zeros(self, img, lab):
        final_shape = (self.imshape[0], self.imshape[1], self.imshape[2])
        x = self.imshape[0]
        y = self.imshape[1]
        z = self.imshape[2]
        if (img.shape[0] < self.imshape[0]):
            x = img.shape[0]
        if (img.shape[1] < self.imshape[1]):
            y = img.shape[1]
        if (img.shape[2] < self.imshape[2]):
            z = img.shape[2]
        img_zero = np.zeros(final_shape)
        img_zero[0:x, 0:y, 0:z] = img[0:x, 0:y, 0:z]
        lab_zero = np.zeros(final_shape)
        lab_zero[0:x, 0:y, 0:z] = lab[0:x, 0:y, 0:z]
        return img_zero, lab_zero

    def pad_with_zeros_patch(self, img, lab):
        final_shape = (256, 256, 256)
        x = 256
        y = 256
        z = 256
        if (img.shape[0] < 256):
            x = img.shape[0]
        if (img.shape[1] < 256):
            y = img.shape[1]
        if (img.shape[2] < 256):
            z = img.shape[2]
        img_zero = np.zeros(final_shape)
        img_zero[0:x, 0:y, 0:z] = img[0:x, 0:y, 0:z]
        lab_zero = np.zeros(final_shape)
        lab_zero[0:x, 0:y, 0:z] = lab[0:x, 0:y, 0:z]
        return img_zero, lab_zero

"""
THIS FUNCTION CAN BE USED WHEN YOU USE THE FULLY IMPLEMENTED TRAINING LOOP FROM KERAS
"""
    def __data_generation(self, image_paths, annot_paths):
        n_data_aug = 5
        count = 0
        nb_patch = 48
        X = np.empty((self.batch_size * n_data_aug * nb_patch, self.imshape[0], self.imshape[1], self.imshape[2], 1),
                     dtype=np.float32)
        Y = np.empty(
            (self.batch_size * n_data_aug * nb_patch, self.imshape[0], self.imshape[1], self.imshape[2], self.n_classes),
            dtype=np.float32)
        for i, (im_path, annot_path) in enumerate(zip(image_paths, annot_paths)):
            img_file = glob.glob(im_path)[0]
            img_ = nib.load(img_file).get_fdata()
            lab_file = glob.glob(annot_path)[0]
            lab_ = nib.load(lab_file).get_fdata()

            img_, lab_ = self.pad_with_zeros_patch(img_, lab_)
            
            if (K.sum(K.flatten(lab_))!=0):
                subject_a = tio.Subject(
                    img_=tio.ScalarImage(img_file, ),
                    label_=tio.LabelMap(lab_file, ))
                # IF SOME IMAGE DIMENSION ARE SMALLER THAN THE PATCH SIZE, PAD WITH ZEROS
                #transform = tio.CropOrPad((256, 256, 256))
                #subject_a = transform(subject_a)

                sampler = tio.GridSampler(subject=subject_a, patch_size=64)
                nbr_batches = len(sampler)

                for j, patch in enumerate(sampler):
                    img = patch['img_'][tio.DATA]
                    lab = patch['label_'][tio.DATA]

                    img = tf.squeeze(img, axis=0)
                    lab = tf.squeeze(lab, axis=0)
                    img = np.asarray(img)
                    lab = np.asarray(lab)

                    img_aug_el, lab_aug_el = self.aug_elastic(img, lab)
                    img_aug_gb, lab_aug_gb = self.aug_gaussian_blur(img, lab)
                    img_aug_gn, lab_aug_gn = self.aug_gaussian_noise(img, lab)
                    img_aug_sh, lab_aug_sh = self.aug_sharpen(img, lab)

                    img, lab = self.reshape(img, lab)
                    img_aug_el, lab_aug_el = self.reshape(img_aug_el, lab_aug_el)
                    img_aug_gb, lab_aug_gb = self.reshape(img_aug_gb, lab_aug_gb)
                    img_aug_gn, lab_aug_gn = self.reshape(img_aug_gn, lab_aug_gn)
                    img_aug_sh, lab_aug_sh = self.reshape(img_aug_sh, lab_aug_sh)

                    if (K.sum(K.flatten(lab[...,1:]))!=0):

                        X[count,] = img
                        Y[count,] = lab
                        count += 1
                    
                        X[count,] = img_aug_el
                        Y[count,] = lab_aug_el

                        count += 1
                        X[count,] = img_aug_gb
                        Y[count,] = lab_aug_gb

                        count += 1
                        X[count,] = img_aug_gn
                        Y[count,] = lab_aug_gn

                        count += 1
                        X[count,] = img_aug_sh
                        Y[count,] = lab_aug_sh

                        count += 1
        return X, Y


"""
THIS FUNCTION IS USED FOR THE CUSTOM TRAINING LOOP
IT CONTAINS PATCHING, DATA AUGMENTATION AND DISCARDES PATCHES WITH NO LABEL FOR BETTER TRAINING ACCURACY
"""
    def data_load_patch(self, image_paths, annot_paths, batchsize =1):
        n_data_aug = 5
        count = 0
        nb_patch = 48
        X = np.empty((self.batch_size * n_data_aug * nb_patch, self.imshape[0], self.imshape[1], self.imshape[2], 1),
                     dtype=np.float32)
        Y = np.empty(
            (self.batch_size * n_data_aug * nb_patch, self.imshape[0], self.imshape[1], self.imshape[2], self.n_classes),
            dtype=np.float32)
        for i, (im_path, annot_path) in enumerate(zip(image_paths, annot_paths)):
            img_file = glob.glob(im_path)[0]
            img_ = nib.load(img_file).get_fdata()
            lab_file = glob.glob(annot_path)[0]
            lab_ = nib.load(lab_file).get_fdata()

            img_, lab_ = self.pad_with_zeros_patch(img_, lab_)
            
            if (K.sum(K.flatten(lab_))!=0):
                subject_a = tio.Subject(
                    img_=tio.ScalarImage(img_file, ),
                    label_=tio.LabelMap(lab_file, ))
                # IF SOME IMAGE DIMENSION ARE SMALLER THAN THE PATCH SIZE, PAD WITH ZEROS
                #transform = tio.CropOrPad((256, 256, 256))
                #subject_a = transform(subject_a)

                sampler = tio.GridSampler(subject=subject_a, patch_size=64)
                nbr_batches = len(sampler)

                for j, patch in enumerate(sampler):
                    img = patch['img_'][tio.DATA]
                    lab = patch['label_'][tio.DATA]

                    img = tf.squeeze(img, axis=0)
                    lab = tf.squeeze(lab, axis=0)
                    img = np.asarray(img)
                    lab = np.asarray(lab)

                    img_aug_el, lab_aug_el = self.aug_elastic(img, lab)
                    img_aug_gb, lab_aug_gb = self.aug_gaussian_blur(img, lab)
                    img_aug_gn, lab_aug_gn = self.aug_gaussian_noise(img, lab)
                    img_aug_sh, lab_aug_sh = self.aug_sharpen(img, lab)

                    img, lab = self.reshape(img, lab)
                    img_aug_el, lab_aug_el = self.reshape(img_aug_el, lab_aug_el)
                    img_aug_gb, lab_aug_gb = self.reshape(img_aug_gb, lab_aug_gb)
                    img_aug_gn, lab_aug_gn = self.reshape(img_aug_gn, lab_aug_gn)
                    img_aug_sh, lab_aug_sh = self.reshape(img_aug_sh, lab_aug_sh)

                    if (K.sum(K.flatten(lab[...,1:]))!=0):

                        X[count,] = img
                        Y[count,] = lab
                        count += 1
                    
                        X[count,] = img_aug_el
                        Y[count,] = lab_aug_el

                        count += 1
                        X[count,] = img_aug_gb
                        Y[count,] = lab_aug_gb

                        count += 1
                        X[count,] = img_aug_gn
                        Y[count,] = lab_aug_gn

                        count += 1
                        X[count,] = img_aug_sh
                        Y[count,] = lab_aug_sh

                        count += 1
        return X, Y, count
