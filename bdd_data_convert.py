

"""
Converts the BDD100K data to KiTTi format 
The format of BDD100K can be read at bdd_100k_format.md

The KiTTi dataset expects a file for each image, where each line is a bounding box of a given class
class|trunc|occluded|observation_angle|bbox left|bbox top| bbox right|bbox bottom| and some others that are not of interest.
See kitti_format.md for full explanation.
An example line can be :
Truck 0.00 0 -1.57 599.41 156.40 629.75 189.25 2.85 2.63 12.34 0.47 1.49 69.44 -1.56
"""

import json
import os
import pandas as pd
LABEL_DIR = "DATA/bdd100k/labels/"
OUT_DIR = "DATA/bdd100k/kitti_labels/train" # Where to write the converted Kitti labels
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
train_filepath = LABEL_DIR + "bdd100k_labels_images_train.json"
val_filepath = LABEL_DIR + "bdd100k_labels_images_val.json"

CATEGORIES = ['Person', 'traffic sign',
              'traffic light', 'car', 'bike', 'truck']


def read_json(filepath):
    """ Reads the json file at the given filepath and returns a dict representation of the file """
    with open(filepath, 'r') as f:
        return json.loads(f.read())


def convert_to_kitti(json_dict):
    for entry in json_dict:
        write_kitti_file(entry)


def write_kitti_file(json_entry):
    """ 
    Extracts all values of interest from the bdd json entry and returns a dataframe where
    each row is a bounding box for an object in the image.
    """
    
    name = json_entry["name"]
    filename = name.split('.')[0] + ".txt"
    filepath = os.path.join(OUT_DIR, filename)
    labels = json_entry["labels"]
    kitti_string = ""
    # Each bounding box in the image
    for label in labels:
        try:
            category = label["category"]
            if category not in CATEGORIES:
                continue
            truncated = label["attributes"]["truncated"]
            # Since occluded is true/false in BDD but [0-3] in Kitti, map all true/false to 1/0.
            occluded = int(label["attributes"]["occluded"])
            # Assume observation angle is 0 for all entries, since this is not included in BDD100K
            angle = 0
            x1, y1, x2, y2 = label["box2d"].values()
            bbox_id = label["id"]
        except KeyError as e:
            print("Keyerror when extracting label for {}, key: {}".format(name, e))
            continue
        kitti_string += "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(
            category, truncated, occluded, angle, x1, y1, x2, y2, bbox_id, 0, 0, 0, 0, 0, 0, 0)

    with open(filepath, 'w') as f:
        print("Writing ", filepath, "to file")
        f.write(kitti_string)
    

def create_ref_file(IMG_DIR, GT_DIR, out_filename):
    """Creates the reference files needed by the modules to perform training. 
    Each reference file (val or train) consists of two entries per line, where the first
    is a filepath to an image, and the second is a filepath to the ground truth (.txt file for 
    bboxes or image for masks). 
    
    Note: The method assumes that the directories contains nothing else than the image and label files,
    and that there is a one-to-one match between images and labels. 
    Arguments:
        IMG_DIR -- Filepath to directory where all images are
        GT_DIR -- Filepath to directory where ground truths are. 
        out_filename -- What the resulting reference file will be named
    """

    images = sorted(os.listdir(IMG_DIR))
    labels = sorted(os.listdir(GT_DIR))
    with open(out_filename, 'w') as f:
        for img in images: #, label in zip(images, labels):
            img_name = img.split('.')[0]
            if any(img_name in label for label in labels):
		print(img_name, label)
                img_path = os.path.join(IMG_DIR, img)
                label_path = os.path.join(GT_DIR, label)
                f.write("{} {}\n".format(img_path, label_path))
	    else:
		print(img_name)
		raise Exception

def make_bbox_ref_file(IMG_DIR, KITTI_LABELS_DIR, out_filename):
    """ Loop through all kitti labels and find the corresponding image"""

    images = set(os.listdir(IMG_DIR))
    pairs = []
    for label in os.listdir(KITTI_LABELS_DIR):
        img_id = label.split(".")[0]
        img_name = img_id + ".png"
        if img_name in images:
            img_path = os.path.join(IMG_DIR, img_name)
            label_path = os.path.join(KITTI_LABELS_DIR, label)
            pairs.append(img_path, label_path)
    
    with open(out_filename, 'w') as f:
        entries = "\n".join(["{} {}".format(img, label) for img,label in pairs])
        f.write(entries)


if __name__ == "__main__":
    #json_dict = read_json(train_filepath)
    #convert_to_kitti(json_dict)
    IMG_DIR = "DATA/bdd100k/seg/images/train"
    KITTI_LABELS = "DATA/bdd100k/kitti_labels/train"

    create_ref_file(IMG_DIR, KITTI_LABELS, "bdd_train.txt")
    #make_bbox_ref_file(IMG_DIR, KITTI_LABELS, "kitti_test")
