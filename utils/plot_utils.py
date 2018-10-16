



import cv2
import os
import logging
import argparse

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def extract_bboxes(filename):
    bboxes = []
    with open(filename, 'r') as f:
        rows = f.readlines()
        for row in rows:
            row = row.split(' ')
            if len(row) == 17:
                label = row[0] + ' ' + row[1]
                (x1, y1, x2, y2) = [int(float(x)) for x in row[5:9]]
            else:
                label = row[0]
                (x1, y1, x2, y2) = [int(float(x)) for x in row[4:8]]
            print(f"Reading file {filename} with label {label} ({x1}, {y1}, {x2}, {y2})")
            bboxes.append((label, (x1,y1,x2,y2)))
    return bboxes 


def draw_bboxes(img, bboxes):
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    COLORS = [RED, GREEN, BLUE]
    for i, bbox in enumerate(bboxes):
        label, (x1, y1, x2, y2) = bbox
        color = COLORS[i % len(COLORS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3) 

def plot_overlays(image_id, LABEL_DIR, IMAGE_DIR, SEG_DIR):
    """ Plots the image with road segmentation and bounding boxes overlayed """
    
    label_path = os.path.join(LABEL_DIR, image_id + ".txt")
    orig_img_path = os.path.join(IMAGE_DIR, image_id) + ".jpg"
    seg_img_path = os.path.join(SEG_DIR, image_id + ".png")
    
    orig_img = cv2.imread(orig_img_path)
    seg_img = cv2.imread(seg_img_path)
    bboxes = extract_bboxes(label_path)
    
    overlayed = cv2.addWeighted(orig_img, 0.7, seg_img, 0.3, 0)
    
    draw_bboxes(overlayed, bboxes)

    cv2.imshow('overlayed', overlayed)
    cv2.waitKey(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_id', default="b1c66a42-6f7d68ca", type=str, help="The id of the image to be displayed")    
    parser.add_argument('--img_dir', default="DATA/bdd100k/images/100k/val", type=str, help="The directory of source images")    
    parser.add_argument('--seg_dir', default="DATA/bdd100k/new_seg/val", type=str, help="Where the segmented images will be stored")
    parser.add_argument('--labels_dir', default='DATA/bdd100k/kitti_labels/val', type=str, help="Where the bboxes for each image will be stored")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    plot_overlays(args.img_id, args.labels_dir, args.img_dir, args.seg_dir)

