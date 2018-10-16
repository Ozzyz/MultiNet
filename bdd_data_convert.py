

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
import numpy as np
import logging
from matplotlib.path import Path
from matplotlib import patches
import matplotlib.pyplot as plt
import cv2 
import argparse

#LABEL_DIR = "DATA/bdd100k/labels/"
#OUT_DIR = "DATA/bdd100k/kitti_labels/val" # Where to write the converted Kitti labels
#SEG_OUT_DIR = "DATA/bdd100k/new_seg/train"
#IMG_DIR = "DATA/bdd100k/images/100k/train"

#if not os.path.exists(OUT_DIR):
#    os.makedirs(OUT_DIR)
#
#if not os.path.exists(SEG_OUT_DIR):
#    os.makedirs(SEG_OUT_DIR)

#train_filepath = LABEL_DIR + "bdd100k_labels_images_train.json"
#val_filepath = LABEL_DIR + "bdd100k_labels_images_val.json"

OBJ_CATEGORIES = ['Person', 'traffic sign',
              'traffic light', 'car', 'bike', 'truck']

ROAD_CATEGORIES = ['drivable area']

def read_json(filepath):
    """ Reads the json file at the given filepath and returns a dict representation of the file """
    with open(filepath, 'r') as f:
        return json.loads(f.read())


def write_all_images_and_labels(json_dict, LABEL_OUT_DIR, SEG_OUT_DIR, SRC_IMG_DIR, show=False):
    """Writes all labels and segmented images specified in the given dict 
    to LABEL_OUT_DIR for the kitti-formated labels and SEG_OUT_DIR for the segmented images."""
    for entry in json_dict:
        write_label_and_segment(entry, LABEL_OUT_DIR, SEG_OUT_DIR,SRC_IMG_DIR, show=show)


def write_label_and_segment(json_entry, LABEL_OUT_DIR, SEG_OUT_DIR, SRC_IMG_DIR, show=False):
    """ 
    Extracts all values of interest from the bdd json entry and writes bounding box 
    info to file, as well as writing segmented road images.
    """
    
    name = json_entry["name"]
    filename = name.split('.')[0] + ".txt"
    filepath = os.path.join(LABEL_OUT_DIR, filename)
    out_path = os.path.join(SEG_OUT_DIR, name)
    labels = json_entry["labels"]
    kitti_string = ""
    logging.info(f"Reading json entry for file {name}")
    contains_drivable = False
    contains_object = False

    categories_in_image = []

    img_path = os.path.join(SRC_IMG_DIR, name)
    img = cv2.imread(img_path)
    seg_image = np.zeros_like(img)
    

    # Each bounding box or drivable area in the image
    for label in labels:
        category = label["category"]
        categories_in_image.append(category)
        if category in OBJ_CATEGORIES:
            kitti_string += extract_bboxes(label)
            contains_object = True
        if category in ROAD_CATEGORIES:
            partial_seg_image = extract_and_draw_drivable_area(seg_image, label, show=True)
            seg_image += partial_seg_image
            contains_drivable = True

    if show:
        cv2.imshow('img', seg_image)
        cv2.waitKey(0)
    
    logging.info(f"Wriging {out_path}.")
    cv2.imwrite(out_path, seg_image)
    if not (contains_drivable and contains_object):
        logging.warn(f"{name} : contains_drivable: {contains_drivable}, contains_object: {contains_object}")
    logging.info(f"Categories in image {name}: {categories_in_image}")

    with open(filepath, 'w') as f:
        logging.info(f"Writing {filepath}")
        f.write(kitti_string)
    

def extract_and_draw_drivable_area(img, label, seg_color = [255, 0, 255], show=False):
    """ Extracts the poly2d information from the drivable area entry and returns a segmented image.
        Each poly2d-entry contains a vertices-entry which is a list of lists of nodes. 
        It also contains a 'type'-entry which is a string like 'LLLCCCL' where L in the i'th entry
        means that node i should be drawn as a cubic Bézier curve (C) or a line (L). 
    """
    LINE_TYPES = {'L' : Path.LINETO, 'C': Path.CURVE4}
    polygons = label["poly2d"]

    #logging.info(f"Writing polygons on image {img_path}")
    for polygon in polygons:
        # TODO: Find out how to write custom curves instead of just line
        draw_codes = [LINE_TYPES[code] for code in polygon["types"]]        
        draw_closed = polygon["closed"]

        nodes = [node for node in polygon["vertices"]]
        if draw_closed:
            nodes.append(nodes[0])
        nodes = np.array(nodes, dtype=np.int32)

        nodes = nodes.reshape((-1, 1, 2))
        cv2.fillPoly(img, [nodes], seg_color)
    
    return img


def extract_bboxes(label):
    """ Returns a string in kitti-format with the extracted entries from the label dict """
    try:
        category = label["category"]
        truncated = label["attributes"]["truncated"]
        # Since occluded is true/false in BDD but [0-3] in Kitti, map all true/false to 1/0.
        occluded = int(label["attributes"]["occluded"])
        # Assume observation angle is 0 for all entries, since this is not included in BDD100K
        angle = 0
        x1, y1, x2, y2 = label["box2d"].values()
        bbox_id = label["id"]
        return "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(
                category, truncated, occluded, angle, x1, y1, x2, y2, bbox_id, 0, 0, 0, 0, 0, 0, 0)
    except KeyError as e:
        logging.warn("Keyerror when extracting label for {}, key: {}".format(category, e))
    return ""


def create_label_ref_file(IMG_DIR, LABEL_DIR, out_filename):
    """Creates the reference files needed by the modules to perform training. 
    Each reference file (val or train) consists of two entries per line, where the first
    is a filepath to an image, and the second is a filepath to the ground truth (.txt file for 
    bboxes or image for masks). 
    
    Note: The method assumes that the directories contains nothing else than the image and label files,
    and that there is a one-to-one match between images and labels. 
    Arguments:
        IMG_DIR -- Filepath to directory where all images are
        LABEL_DIR -- Filepath to directory where ground truths are. 
        out_filename -- What the resulting reference file will be named
    """

    images = sorted(os.listdir(IMG_DIR))
    labels = sorted(os.listdir(LABEL_DIR))
    with open(out_filename, 'w') as f:
        for img in images: #, label in zip(images, labels):
            img_name = img.split('.')[0]
            label = img_name + ".txt"
            if any(img_name in label for label in labels):
                img_path = os.path.join(IMG_DIR, img)
                label_path = os.path.join(LABEL_DIR, label)
                f.write(f"{img_path} {label_path}\n")
            else:
                print(img_name)
                raise Exception

def create_seg_ref_file(IMG_DIR, SEG_DIR, out_filename):
    """ Creates the reference files needed by the modules to perform training. 
    Each reference file (val or train) consists of two entries per line, where the first
    is a filepath to an image, and the second is a filepath to the ground truth (.txt file for 
    bboxes or image for masks). 
    
    Note: The method assumes that the directories contains nothing else than the image and label files,
    and that there is a one-to-one match between images and labels. 
    Arguments:
        IMG_DIR -- Filepath to directory where all images are
        SEG_DIR -- Filepath to directory where ground truths are. 
        out_filename -- What the resulting reference file will be named
    """
    images = os.listdir(IMG_DIR)
    seg_images = os.listdir(SEG_DIR)

    with open(out_filename, 'w') as f:
        for image in images:
            img_id = image.split('.')[0]
            if not any(img_id in seg_img for seg_img in seg_images):
                logging.warn(f"Could not find segmented image for image id {img_id}")
                continue
            img_path = os.path.join(IMG_DIR, image)
            seg_path = os.path.join(SEG_DIR, img_id+".jpg")
            f.write(f"{img_path} {seg_path}\n")
    

def make_bbox_ref_file(IMG_DIR, KITTI_LABELS_DIR, out_filename):
    """ Loop through all kitti labels and find the corresponding image.
        Then, create a file with name {out_filename} where each row 
        is of format {img_path} {label_path}
    """

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_img_dir', default="DATA/bdd100k/images/100k/val", type=str, help="The directory of source images")
    parser.add_argument('--json_path', default="DATA/bdd100k/labels/bdd100k_labels_images_val.json", type=str, help="The filepath to the json annotation file")

    parser.add_argument('--seg_out_dir', default="DATA/bdd100k/new_seg/val", type=str, help="Where the segmented images will be stored")
    parser.add_argument('--labels_out_dir', default='DATA/bdd100k/kitti_labels/val', type=str, help="Where the bboxes for each image will be stored")
    parser.add_argument('--seg_out_file', default='img_seg_val.txt', type=str, help="The filename of the file that links images and segmented images")
    parser.add_argument('--labels_out_file', default='img_bbox_val.txt', type=str, help="The filename of the file that links images and label (.txt) files")

    return parser.parse_args()

def maybe_create_dir(*dirs):
    """ Creates the directories if they do not exist """
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logging.info(f"Creating dir {dir_path}")
        else:
            logging.info(f"{dir_path} already exists - skipping.")


if __name__ == "__main__":
    args = parse_args()
    # Create all segmented images and labels
    json_dict = read_json(args.json_path)
    LABEL_OUT_DIR, SEG_OUT_DIR, SRC_IMG_DIR = args.labels_out_dir, args.seg_out_dir, args.src_img_dir
    maybe_create_dir(LABEL_OUT_DIR, SEG_OUT_DIR, SRC_IMG_DIR)
    write_all_images_and_labels(json_dict, LABEL_OUT_DIR, SEG_OUT_DIR, SRC_IMG_DIR, show=False)
    # Create link files that connects input (img) and output (bboxes or segmentation) 
    create_label_ref_file(SRC_IMG_DIR, LABEL_OUT_DIR, args.labels_out_file)
    create_seg_ref_file(SRC_IMG_DIR, SEG_OUT_DIR, args.seg_out_file)
