from skimage.io import imread, imsave
import numpy as np
import os

data_path = os.path.join('DATA', 'seg', 'color_labels')

for root, dirs, files in os.walk(data_path, topdown=True):
    i = 0
    for name in files:
            path = os.path.join(root, name)

            img = imread(path)
            out = np.zeros((img.shape[0], img.shape[1], 3), int)

            for x in range(img.shape[0]):
                for y in range(img.shape[1]):
                    if np.array_equal(img[x][y][:3], [128, 64, 128]):
                        out[x][y] = [255, 0, 255]
                    else:
                        out[x][y] = [255, 0, 0]
            
            out_path = path.replace('color_labels', 'kitti_labels')
            imsave(out_path, out)
            i += 1
            print(i, '/', len(files))