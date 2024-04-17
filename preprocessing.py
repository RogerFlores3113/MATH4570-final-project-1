
import numpy as np
import os
import random
import pandas as pd
from PIL import Image
import ast
import json


DATA_ROOT = "renders"

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


SEGMENTATION_FILES = list(sorted([
                      "Results_Epidural Hemorrhage Detection_2020-11-16_21.31.26.148.csv",
                      "Results_Intraparenchymal Hemorrhage Detection_2020-11-16_21.39.31.268.csv",
                      "Results_Multiple Hemorrhage Detection_2020-11-16_21.36.24.018.csv",
                      "Results_Subarachnoid Hemorrhage Detection_2020-11-16_21.36.18.668.csv",
                      "Results_Subdural Hemorrhage Detection_2020-11-16_21.35.48.040.csv",
                      "Results_Subdural Hemorrhage Detection_2020-11-16_21.37.19.745.csv"
                      ]))

SEGMENTATION_NAMES = list(sorted(["epidural",
                      "intraparenchymal",
                      "multiple",
                      "subarachnoid",
                      "subdural-1",
                      "subdural-2"]))


def load_labels(path_to_folder="../../Hemorrhage Segmentation Project"):
    path = path_to_folder + "/" if path_to_folder[-1] != "/" else path_to_folder

    df = pd.read_csv(path + "hemorrhage-labels.csv", index_col="Image")
    dict = df.T.to_dict("dict")

        # if only_classification:
        #     return dict
        
        # for filename, parent  in zip(SEGMENTATION_FILES, SEGMENTATION_NAMES): 
        #     dat1 = load_segmentation(path + filename)
        #     merge_dicts(dict, dat1, parent)
        #     print(f"loaded {parent} segmentation data")

    return dict

    
def merge_dicts(dict1, dict2, parent):
    for key in dict2:
        if key in dict1:
            # Add dict2s key-value pairs under the parent_key in dict1
            dict1[key][parent] = dict2[key]
        else:
            # If key doesn't exist in dict1, add it
            dict1[key] = {parent: dict2[key]}
    return dict1


# def load_segmentation(path_to_folder):

#     df = pd.read_csv(path_to_folder)

#     df["Origin"] = df["Origin"].str.replace(".jpg", "", regex=False)
#     df.set_index("Origin", inplace=True)

#     if df.columns.duplicated().any():
#         # Handle duplicate column names. Here we add a suffix to the duplicates
#         df.columns = pd.io.parsers.base_parser.ParserBase({'names': df.columns})._maybe_dedup_names(df.columns)


#     return df.T.to_dict("dict")



def load_segmentation(path_to_folder):
    path = path_to_folder + "/" if path_to_folder[-1] != "/" else path_to_folder

    dfs = {}
    def json_loader(s):
        if isinstance(s, str):
            return json.loads(s)
        else:
            return s

    for filename, parent  in zip(SEGMENTATION_FILES, SEGMENTATION_NAMES): 
            df = pd.read_csv(path + filename)
            df["Origin"] = df["Origin"].str.replace(".jpg", "", regex=False)
            df.set_index("Origin", inplace=True)
            
            df["Correct Label"] = df["Correct Label"].str.replace("\'", "\"", regex=False)
            df['Correct Label'] = df['Correct Label'].apply(json_loader)
            df["Majority Label"] = df["Majority Label"].str.replace("\'", "\"", regex=False)
            df['Majority Label'] = df['Majority Label'].apply(json_loader)


            dfs[parent] = df.copy()

            print(f"loaded {parent} segmentation data")

    return dfs


def image_path_generator(folder_path, seed=0):
    all_files = []
    # walk through all files and collect paths
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".jpg"):
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
            if file.lower().endswith(".jpg"):
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
        

