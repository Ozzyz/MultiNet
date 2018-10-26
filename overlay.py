from matplotlib import image, pyplot
import numpy as np
import os
import time

#Takes in an image folder and a segmentation mask folder, layers the mask over the image for each match and displays it
def overlay(img_dir, seg_dir):
    files = os.listdir(img_dir)
    i = 0
    for file in os.listdir(img_dir):
        seg_file = file.replace('.jpg', '.png')  # Mask are saved as png rather than jpg
        img_path = os.path.join(img_dir, file)
        seg_path = os.path.join(seg_dir, seg_file)
        try:
            # Try loading the segment mask first as we know the image should exist
            seg = image.imread(seg_path)
            img = image.imread(img_path)

            # Normalize images
            seg = seg/np.max(seg)
            img = img/np.max(img)

            # Layer the images and normalize result
            overlay = img + seg
            overlay /= np.max(overlay)

            # Display overlay for a set time
            pyplot.imshow(overlay)
            pyplot.pause(1)

        except FileNotFoundError:
            print(seg_file + ' not found in seg')

        i += 1
        print(str(i) + '/' + str(len(files)))

if __name__ == "__main__":
    img_dir = os.path.join('DATA', 'bdd100k', 'cropped_images', '100k', 'val')
    seg_dir = os.path.join('DATA', 'bdd100k', 'cropped_seg', 'val')
    overlay(img_dir, seg_dir)
