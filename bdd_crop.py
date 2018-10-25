from PIL import Image
import numpy as np
import os

def crop_images(data_path, replacement):
    for root, dirs, files in os.walk(data_path, topdown=True):
        i = 0
        for name in files:
                path = os.path.join(root, name)
                img = Image.open(path)
                
                width, height = img.size
                new_width, new_height = 1248, 384

                left = (width - new_width)//2
                top = (height - new_height)//2
                right = (width + new_width)//2
                bottom = (height + new_height)//2

                crop = img.crop((left, top, right, bottom))

                out_path = path.replace(replacement[0], replacement[1])
                crop.save(out_path)

                i += 1
                print(i, '/', len(files))

if __name__ == "__main__":
    img_path = os.path.join('DATA', 'bdd100k', 'images', '100k')
    crop_images(img_path, ['images', 'cropped_images'])
    seg_path = os.path.join('DATA', 'bdd100k', 'new_seg')
    crop_images(seg_path, ['new_seg', 'cropped_seg'])
