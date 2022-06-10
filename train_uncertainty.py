import sys

import time
import nibabel as nib
import tensorflow as tf
import numpy as np
import glob
import os
import random
from keras.models import load_model
from keras import backend as K

from models import  unet_3d_inst_norm, unet_jaye_3d_ch, unet_3d_gaussian_do, unet_3d_do_innerlayer
from loss_functions import dropout_loss, dice_coef_3cat_loss, dice_loss_git,  weighted_cross_entropyloss, \
    dice_coef_git, dice_loss_git_all
from utils import get_curr_time
from dataloader_uncertainty import DataGenerator
from callbacks import TB_Image
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
import yaml
from custom_callback import CustomReduceLRoP

# avoids tensorflow uses all GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


############### LOAD DATA  ###################
with open("/media/m-ssd2/jaye/fetal_lung_segmentation/Code/3d_UNet/3d_model/config_uncertainty.yaml", "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

if config["augmentation_affine"]:
    augmentation = "augmentation"
else:
    augmentation = "no_augmentation"

save_freq = 5
model_weights = "/Users/aplu/Documents/semesterarbeit2022/Code/data_separation/3d_model/saved_models_uncertainty/"

training_data_path    = '/media/m-ssd2/jaye/fetal_lung_segmentation/nnUnet/nnUNet_raw_data_base/nnUNet_raw_data/Task601_lung/'
validation_data_path  = '/media/m-ssd2/jaye/fetal_lung_segmentation/nnUnet/nnUNet_raw_data_base/nnUNet_raw_data/Task601_lung/'
tensorboard_data_path = '/media/m-ssd2/jaye/fetal_lung_segmentation/Code/3d_UNet/3d_model/tb_data/*'
checkpoint_path       = config["save_model_path"]

model_name = os.path.join(config["model_name"] +
                          '_' + config["orientation"] +
                          '_' + config["loss"] +
                          '_' + str(config["learning_rate"]) +
                          '_' + augmentation +
                          '_' + 'epochs' + str(config["num_epochs"]) +
                          '_' + 'bs' + str(config["batch_size"]) +
                          '_' + get_curr_time())

logs_path = os.path.join(checkpoint_path,
                         "logs",
                         model_name)

model_save_file = os.path.join(checkpoint_path,
                               model_name + "_patches.model")





training_images_path = os.path.join(training_data_path + 'imagesTr/*')
training_labels_path = os.path.join(training_data_path + 'labelsTr/*')

validation_images_path = os.path.join(training_data_path + 'imagesTs/*')
validation_labels_path = os.path.join(training_data_path + 'labelsTs/*')

tr_ipaths = sorted(glob.glob(training_images_path))
tr_lpaths = sorted(glob.glob(training_labels_path))

val_ipaths = sorted(glob.glob(validation_images_path))
val_lpaths = sorted(glob.glob(validation_labels_path))

train_dg = DataGenerator(image_paths=tr_ipaths, annot_paths=tr_lpaths,
                         batch_size=1,
                         augment_affine=config["augmentation_affine"],
                         probability_affine=config["probability_affine"],
                         lung=config["lung"],
                         separation=config["train_separation"])

val_dg = DataGenerator(image_paths=val_ipaths, annot_paths=val_lpaths,
                       batch_size=1,
                       augment_affine=False,
                       lung=config["lung"],
                       separation=config["train_separation"])

""" LOAD IMAGE CONFIGURATION FOR NN"""

### LOAD MODEL PARAMETERS ###

opt_SGD = tf.keras.optimizers.SGD(learning_rate=0.01,
                                  momentum=0.99, nesterov=True)
if config["loss"] == "dice_background":
    loss = dice_loss_git_all
    metrics = [dice_coef_git_all]
    if config["monitor"] == "validation loss":
        monitor = 'val_loss'
    elif config["monitor"] == "validation accuracy":
        monitor = 'val_gen_dice_fhnw_coef'

elif config["loss"] == "dice_foreground":
    loss = dice_loss_git
    metrics = [dice_coef_git]

    if config["monitor"] == "validation loss":
        monitor = 'val_loss'
    elif config["monitor"] == "validation accuracy":
        monitor = 'val_gen_dice_coef'

# METRICS

train_accuracy_metric = tf.keras.metrics.CategoricalCrossentropy( name='train_accuracy')
val_accuracy_metric = tf.keras.metrics.CategoricalCrossentropy( name='val_accuracy')

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

### MODEL ###
imshape = (64,64,64, 1)
model = unet_3d_inst_norm(input_shape=imshape, model_name=config["model_name"], n_classes=3)
"""
path = "/media/m-ssd2/jaye/fetal_lung_segmentation/Code/3d_UNet/3d_model/saved_models_uncertainty/110_n.model"
if os.path.exists(path):
    model = load_model(path, compile=False)
    model.summary()
else:
    print('Failed to load existing model at: {}'.format(path))
"""
model.compile(optimizer = opt_SGD, loss = loss, metrics = metrics)



## Custom Modification:

reduce_rl_plateau = CustomReduceLRoP(factor= 0.7, patience=20, 
                              verbose=1, 
                              optim_lr=opt_SGD.learning_rate, 
                              reduce_lin=False)


##### LOAD TRAINING PARAMETERS ######
load_weights_from_epoch = -1
load_weights_from_epoch = -1
step_size = 5
EPOCHS = 300
BS = 1
BS_aug = 3
numUpdates = np.int32(55)
nbr_dropout_samples = 5
indexes = np.arange(55)
alpha = 3
shuffle = True
loss_monitor = []
loss_monitor_all =[]
dice_val = []
do_monitor = []
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        #print(x.shape)
        pred = model(x, training=True)      
        pred_stack = [] 
              
        pred_stack_r = [] 
        for i in range(3):
            pred_do = model(x, training=True)
            pred_stack.append(pred_do[...,1:])
            

        loss_do = dropout_loss(y, pred_stack)
        
        loss_dice = dice_loss_git(y, pred)
        
        dice = dice_loss_git_all(y, pred)

        loss_value = loss_dice + alpha * loss_do
        
        
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_accuracy_metric.update_state(y, pred)
    return loss_value, dice

@tf.function
def test_step(x, y):
    val_logits = model(x, training=False)
    val_accuracy_metric.update_state(y, val_logits)
    
    return float(dice_loss_git(y, val_logits))

### TRAIN NETWORK ###
reduce_rl_plateau.on_train_begin()

for epoch in range(0, EPOCHS):
    print("[INFO] starting epoch {}/{}...".format(
        epoch + 1, EPOCHS), end="")
    sys.stdout.flush()
    start_time = time.time()
    if shuffle == True:
        np.random.shuffle(indexes)
    

    for step in range(0,numUpdates):
        start = step * BS
        end = start + BS
        batch_indexes = indexes[start:end]
        img_batch = [train_dg.image_paths[k] for k in batch_indexes]
        lab_batch = [train_dg.annot_paths[k] for k in batch_indexes]
        batch = [img_batch, lab_batch]
        image, label, count = train_dg.data_load_patch(img_batch, lab_batch, batchsize=BS)
        indexes_sm = np.arange(np.int32(count))
        np.random.shuffle(indexes_sm)
        dice_arr_m = []
        for step_sm in range(0, np.int32(count/BS_aug)):
            start_sm = step_sm * BS_aug
            end_sm = start_sm + BS_aug
            batch_indexes_sm = indexes_sm[start_sm:end_sm]
            img_batch_sm = [image[k,...] for k in batch_indexes_sm]
            lab_batch_sm = [label[k,...] for k in batch_indexes_sm]
            img_batch_sm = np.asarray(img_batch_sm)
            lab_batch_sm = np.asarray(lab_batch_sm)
            loss_value, dice = train_step(img_batch_sm, lab_batch_sm)
            dice_arr_m.append(float(loss_value))
            if step_sm % 10 == 0:
                print(float(loss_value))
                print(float(dice))
         
        loss_monitor_all.append(dice)
        loss_monitor.append(loss_value)
        #do_monitor.append(loss_do)
    train_accuracy_metric.result() 
    train_accuracy_metric.reset_states()  
    if epoch % 10 == 0:
       save_= os.path.join(checkpoint_path,
                               model_name + "__" + str(epoch)+   "_NORMAL_2.model")
       print(save_)
       print("SAVING MODEL")
       model.save(save_)



    # VALIDATION
    indexes_val = np.arange(5)
    np.random.shuffle(indexes_val)
    val_indexes= indexes_val[0:1]
    print(val_indexes)
    val_img_batch = [val_dg.image_paths[k] for k in val_indexes]
    val_lab_batch = [val_dg.annot_paths[k] for k in val_indexes]
    val_image, val_label, count_ = val_dg.data_load_patch(val_img_batch, val_lab_batch, batchsize = BS)
    loss_val = test_step(val_image[0:2], val_label[0:2])
    print("val loss")
    print(float(loss_val))
    dice_val.append(float(loss_val))
  
    val_acc = val_accuracy_metric.result()
    val_accuracy_metric.reset_states()
    reduce_rl_plateau.on_epoch_end(epoch, 1-loss_val)

    with open('dice_monitor.txt', 'w') as f:
        for item in loss_monitor:
            f.write("%s\n" % float(item))



    
"""   
    if epoch % save_freq == 0:
        model.save_weights(filepath=model_weights + "epoch-{}".format(epoch), save_format="tf")
"""  
with open('dice_only.txt', 'w') as f:
    for item in loss_monitor:
        f.write("%s\n" % float(item))
with open('dice_all.txt', 'w') as f:
    for item in loss_monitor_all:
        f.write("%s\n" % float(item))
with open('dice_val.txt', 'w') as f:
    for item in dice_val:
        f.write("%s\n" % float(item))

with open('do_monitor.txt', 'w') as f:
    for item in do_monitor:
        f.write("%s\n" % float(item))

model.save(model_save_file)

