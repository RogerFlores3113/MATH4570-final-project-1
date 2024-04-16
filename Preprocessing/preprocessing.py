
import numpy as np
import os
import random
import pandas as pd
from PIL import Image


DATA_ROOT = "data"

DATA_FOLDERS = ["epidural",
                "intraparenchymal",
                "itroventricular",
                "multi",
                "normal",
                "subarachnoid",
                "subdural"]

DATA_SUBFOLDERS = ["brain_bone_window", 
                    "brain_window", 
                    "max_contrast_window", 
                    "subdural_window"]

def load_labels(path_to_labels="../../hemorrhage-labels.csv"):
    df = pd.read_csv(path_to_labels, index_col="Image")
    return df.T.to_dict("dict")



def image_path_generator(folder_path, seed=0):
    all_files = []
    # walk through all files and collect paths
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.jpg'):
                image_path = os.path.join(root, file)
                all_files.append(image_path)
    
    # we randomize the files according to our seed
    random.seed(seed)
    random.shuffle(all_files)

    # yields:  img_path
    for image_path in all_files:
        yield image_path




def image_generator(folder_path, seed=0):
    all_files = []
    # Walk through all files and collect paths
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.jpg'):
                image_path = os.path.join(root, file)
                all_files.append((image_path, file))
    
    # we randomize the files according to our seed
    random.seed(seed)
    random.shuffle(all_files)

    # yields:  img, seed
    for image_path, file_id in all_files:
        image = Image.open(image_path)
        image_array = np.array(image)
        yield image_array, file_id.split(".")[0]





def batch_generator(folder_path, batch_size=9, seed=0):
    batch = []
    ids = []
    for image_data, file_id in image_generator(folder_path, seed=seed):
        batch.append(image_data)
        ids.append(file_id)
        if len(batch) == batch_size:
            yield np.array(batch), ids
            batch = []
            ids = []
    
    # to clear out the remainder if our batch size doesn't match total size perfectly!
    if batch:
        yield np.array(batch), ids
