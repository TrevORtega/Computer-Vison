import numpy as np
import shapefile as shp
import cv2
import imageio
import torch
from PIL import Image
from pathlib import Path
from IPython.display import Image as I
import torchvision
import torchvision.transforms.functional as F
import torchvision.transforms as T
import os, sys, math, random
sys.path.append('../')
from models.utils import spatial_transforms, geometric_transforms, make_keypoint_mask

# In[17]:


class Keypoint_Dataset(torch.utils.data.Dataset):
    def __init__(self, tif, shp, data_path, transforms = None, data = None):
        self.tif = tif
        self.shp = shp
        self.data_path = data_path
        self.transforms = transforms
        
        # (latitude, longitude, meter to pixel conversion ratio) gotten from .tfw file
        self.coords = self.getRealCoords(self.tif[:-3] + 'tfw')
        
        self.avg_pic_size = imageio.imread(Path(data_path, os.listdir(data_path)[0])).shape
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        if data is None:
            # generates a list of tuples of format: ({tile for data point}, {bbox for data point})
            self.data = self.gen_data()
        else:
            # we imported data from somewhere else
            self.data = data
    
    #item is (path, bbox). We open image as numpy array, then convert it to 3 channel with cv2, then turn into tensor
    def __getitem__(self, index):
        path, pixels = self.data[index]
        
        img_arr = imageio.imread(path)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGRA2BGR)
        try:
            mask = make_keypoint_mask(img_arr, pixels)
        except:
            print('img:', img_arr.shape, 'pixels:', pixels)
            raise
        og_img = None
        if self.transforms: 
            img_arr, mask = geometric_transforms(img_arr, mask)
            img_arr, og_img, mask = spatial_transforms(img_arr, self.transforms, mask=mask)
        else:    
            img_arr = torchvision.transforms.ToTensor()(img_arr)
        #return (img_arr, mask, og_img) 
        return (img_arr, mask)
    
    def __len__(self):
        return len(self.data)


# In[18]:


    # Finding the specific tile that belongs to the given pixel inside a bigger image
    def get_tile(self, pixel):
        pixel_w, pixel_h = pixel
        h, w, _ = self.avg_pic_size
        
        # divide pixel length by slice length to find its specific slice
        column, pixel_w = [abs(int(x)) for x in divmod(pixel_w, w)]
        row, pixel_h = [abs(int(x)) for x in divmod(pixel_h, h)]
        # tile format is _{row}_{column}.tif
        # return path to tile, and new pixel
        return Path(self.data_path, f'_{row:02d}_{column:02d}.tif'), (pixel_w, pixel_h) 


    # In[20]:


    # shp data points are given in latitude and longitude coordinates, we must convert them to meters, then to pixel distances
    def getPixel(self, lat2, lon2):
        lat1, lon1, meters2pixels = self.coords
        x = abs((lat2 - lat1) * meters2pixels)

        y = abs((lon2 - lon1) * meters2pixels)

        return x, y


    # In[21]:


    # gets pixel values according to a given shape file and coordinate locations, and pairs them with a respective tile
    def get_points_and_tiles(self, sf):
        """
        targets format : {
                path_2_img1 : num_objects_in_img1,
                path_2_img2 : num_objects_in_img2
                         }
        """
        targets = {}
        for shape in sf.iterShapes():
            point = shape.points[0]
            width, height = point
            pixel = self.getPixel(width, height)
            tile, pixel = self.get_tile(pixel)
            
            if tile in targets:
                targets[tile].append(pixel)
            else:
                targets[tile] = [pixel]

        return targets 

    def format_data(self, targets):
        data = []
        for img, pixels in targets.items():
            data.append((img, pixels))
        return data

    def add_random_tiles(self, targets):
        ratio = 0.001
        directory = os.listdir(self.data_path)
        num_random_pics = int(len(directory) * (1-ratio))
        for i in range(num_random_pics):
            path = self.data_path + '/' + random.choice(directory)
            if path not in targets:
                targets[path] = []
        return targets


    def gen_data(self):
        tfw = self.tif[:-3] + 'tfw'
        sf = shp.Reader(self.shp)
        targets = self.get_points_and_tiles(sf)
        targets = self.add_random_tiles(targets)
        return self.format_data(targets)


    # gets the spacial coordinates of the top left corner of the image (.tif)
    # These coordinates are in a tfw file with the same name as the image file
    def getRealCoords(self, tfw):
        x , y, meters2pixels = None, None, None
        with open(tfw) as f:
            # 5th line should be x coord, 6th is y
            lines = f.readlines()
            meters2pixels = 1 / float(lines[0])
            x = float(lines[4][:-2])
            y = float(lines[5][:-2])


            return (x, y, meters2pixels)
