import numpy as np
import cv2 as cv
import image_slicer
import os
from pathlib import Path
import math
from BoundBoxes import getPoints
from PIL import Image

def draw_cv(img, x1, y1, x2, y2):
    # Create a black image
    #img = np.zeros((512,512,3), np.uint8)
    #img = cv.UMat(img).get()

    # Draw a diagonal blue line with thickness of 5 px
    #cv.line(img,(0,0),(511,511),(255,0,0),5)
    cv_img = cv.imread(img)
    cv.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cv.imshow(img,cv_img)
    cv.waitKey(0)  # 0==wait forever


# splits image into k boxes; k must be even
def split_img(directory, img_name, k):
    img = directory + img_name
    tiles = image_slicer.slice(img, k, save=False)
    new_path = Path(directory + img_name.split('.')[0])

    if new_path.exists():
        image_slicer.save_tiles(tiles, prefix='', directory=new_path)
    else:
        os.mkdir(new_path)
        image_slicer.save_tiles(tiles, prefix='', directory=new_path)
    return new_path


def get_slice(pixel, img_size, k):
    img_w, img_h = img_size
    slice_w, slice_h = int(img_w / math.sqrt(k)), int(img_h / math.sqrt(k))
    w, h = pixel

    row_coord = math.ceil(w / slice_w)
    col_coord = math.ceil(h / slice_h)
    return f'_0{row_coord}_0{col_coord}.png'

def draw_points(new_path, img_size, k):
    for pixel, p1, p2 in getPoints(None):
        file_name = get_slice(pixel, img_size, k)
        print(file_name)
        print(p1, p2)
        x1, y1 = p1
        x2, y2 = p2
        draw_cv(f'{new_path}\\{file_name}', x1, y1, x2, y2)



def main():
    path = 'D:\\Research\\ObjectDetection\\WallinData\\wallin\\blaine_harbor\\blaine_June19\\'  # \\pics\\'
    img_name = 'Blaine_June19_2019_flt3_5_P4P_ortho.tif'  # 'motherboard.jpg'
    img_size = 34326, 17976
    Image.MAX_IMAGE_PIXELS = None
    new_path = split_img(path, img_name, 9)
    os.chdir(path)
    k = 9
    draw_points(new_path, img_size, k)
