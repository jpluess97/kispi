import os, sys
import nibabel as nib
import pandas as pd
import random
from datetime import datetime
random.seed(1812)

# base_path: change to NOT aligned!
base_path = os.path.normpath("/media/m-ssd2/feta_projects/feta_base/feta_2.1_training")
testing_path = os.path.normpath("/media/m-ssd2/feta_projects/feta_base/feta_2.1_validation")
metainfo_path = "/media/m-ssd2/feta_projects/feta_base/feta_2.1_training/participants.tsv"
templates_path = os.path.normpath("/media/m-proc/andras/Fetlas/CRL")
rel_dir = os.path.normpath("sub-{0:03d}/anat")
data_file = "sub-{0:03}_rec-{type}_T2w.nii.gz"
seg_file = "sub-{0:03d}_rec-{type}_dseg.nii.gz"


def get_data_path(i: int) -> str:
    assert i > 0 and i <= 80 
    path = os.path.join(base_path, rel_dir.format(i), data_file.format(i, type = "mial"))
    if os.path.exists(path):
        pass
    else:
        path = os.path.join(base_path, rel_dir.format(i), data_file.format(i, type = "irtk"))
    return path

def get_seg_path(i: int) -> str:
    assert i > 0 and i <= 80
    path = os.path.join(base_path, rel_dir.format(i), seg_file.format(i, type = "mial"))
    if os.path.exists(path):
        pass
    else:
        path = os.path.join(base_path, rel_dir.format(i), seg_file.format(i, type = "irtk"))
    return path 

def get_testing_data_path(i: int) -> str:
    assert i > 900 and i <= 940 
    path = os.path.join(testing_path, rel_dir.format(i), data_file.format(i, type = "mial"))
    if os.path.exists(path):
        pass
    else:
        path = os.path.join(testing_path, rel_dir.format(i), data_file.format(i, type = "irtk"))
    return path

def get_testing_seg_path(i: int) -> str:
    assert i > 900 and i <= 940
    path = os.path.join(testing_path, rel_dir.format(i), seg_file.format(i, type = "mial"))
    if os.path.exists(path):
        pass
    else:
        path = os.path.join(testing_path, rel_dir.format(i), seg_file.format(i, type = "irtk"))
    return path 


def get_data_paths(ids: list) -> list:
    return [get_data_path(i) for i in ids] 

def get_seg_paths(ids: list) -> list:
    return [get_seg_path(i) for i in ids] 


def load_dnifty(i: int):
    return nib.load(get_data_path(i))

def load_snifty(i: int):
    return nib.load(get_seg_path(i))

def load_data(i: int):
    return nib.load(get_data_path(i)).get_fdata()

def load_seg(i: int):
    return nib.load(get_seg_path(i)).get_fdata()

def load_all(i: int):
    return load_data(i), load_seg(i)

###

def number_from_id(id_):
    n = ""
    for x in list(id_):
        if x.isdigit():
            n += str(x)
    return n

def get_data_paths_split(splits: list, nimages = None, all_images = False):
    '''
    nimages: number of images per split
    splits: [(20,25), (26, 29), (30, 35)]
    '''
    
    def get_ids(data):        
        ids = data["participant_id"].apply(number_from_id)
        ids_numeric = list(map(int, ids))
        return ids_numeric
    
    metainfo = pd.read_csv(metainfo_path, sep='\t')
    
    # assert that every number in split is in the GA range
    path_list = []
    for split in splits:
        first = split[0]
        last = split[1]
    
        data = metainfo[((metainfo["Gestational age"] >= first) & (metainfo["Gestational age"] < last))]
        all_paths = get_data_paths(get_ids(data))
        
        if all_images:
            path_list.append(all_paths)
        else:
            assert nimages != None
            sample_paths = random.sample(all_paths, nimages)
            path_list.append(sample_paths)
        
    if len(path_list) > 1:
        return path_list
    else:
        return path_list[0]
    

def get_seg_paths_split(splits: list, nimages = None, all_images = False):
    '''
    nimages: number of images per split
    splits: [(20,25), (26, 29), (30, 35)]
    '''
    
    def get_ids(data):        
        ids = data["participant_id"].apply(number_from_id)
        ids_numeric = list(map(int, ids))
        return ids_numeric
    
    metainfo = pd.read_csv(metainfo_path, sep='\t')
    
    # assert that every number in split is in the GA range
    path_list = []
    for split in splits:
        first = split[0]
        last = split[1]
    
        data = metainfo[((metainfo["Gestational age"] >= first) & (metainfo["Gestational age"] < last))]
        all_paths = get_seg_paths(get_ids(data))
        
        if all_images:
            path_list.append(all_paths)
        else:
            assert nimages != None
            sample_paths = random.sample(all_paths, nimages)
            path_list.append(sample_paths)
        
    if len(path_list) > 1:
        return path_list
    else:
        return path_list[0]

def get_curr_time():
    return datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
