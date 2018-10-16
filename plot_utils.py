



import cv2


def extract_bboxes(filename):
    bboxes = []
    with open(filename, 'r') as f:
        rows = f.readlines()
        for row in rows:
            label, (x1, y1, x2, y2) = row[0], row[4:8]
            bboxes.append((label, (x1,y1,x2,y2))
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
    orig_img_path = os.path.join(IMAGE_DIR, image_id) + ".jpeg"
    seg_img_path = os.path.join(SEG_DIR, image_id + ".png")
    
    orig_img = cv2.imread(orig_img_path)
    seg_img = cv2.imread(seg_img_path)
    bboxes = extract_bboxes(label_path)
    
    overlayed = cv2.addWeighted(orig_img, 0.7, seg_img, 0.3, 0)
    
    bbox_img = draw_bboxes(overlayed, bboxes)

    cv2.imshow(bbox_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    plot_overlays('c16e5581-3e798354', 'DATA/val', 'DATA/images/val', 'DATA/new_seg/val')
