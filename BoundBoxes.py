import cv2
import os
import shapefile
from pathlib import Path

# Width, Height in pixels of the average bounding box
BOX_DIM = 200,200#150, 150
points = [(50,50),(100,100),(1900,1045),(1800,900)]
# Width / Height of image
IMG_DIM_RATIO = 36136 / 68262


# takes in a point from a shapefile, then yields coordinates for a cv2 rectangle
def getPoints(shp, dbf):
    box_radius = int(BOX_DIM[0] / 2)
    sf = shapefile.Reader(shp, dbf=dbf)
    for shape in sf.iterShapes():
        point = shape.points[0]
        width, height = point
        x1 = width - box_radius
        y1 = height + box_radius
        x2 = width + box_radius
        y2 = height - box_radius
        # Normalize so each coordinate is between 0 and 1
        #adj_point = width * IMG_DIM_RATIO, height * IMG_DIM_RATIO
        yield point, (x1, y1), (x2, y2)
# Draw rectangle with cv2
def drawRectangle(path, name, p1, p2):
    print(path)
    img = cv2.imread(path)
    cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
    cv2.imwrite(name, img)


# Draw and save bounding boxes around each point in image
def drawBoxes(folder_path, img_name, shp_name):
    output_name = img_name[:-3] + 'txt'
    img_path = folder_path + img_name
    shp_path = folder_path + shp_name
    print('path exist?:', Path(img_path).exists())
    for point in getPoints(shp_path):
        point1, point2 = point
        drawRectangle(img_path, output_name, point1, point2)

    return img_path, output_name


if __name__ == "__main__":
    folder_path = os.getcwd() + '\\WallinData\\wallin\\blaine_harbor\\blaine_June19\\'
    img_name = 'Blaine_June19_2019_flt1_2_ortho.tif'
    shp_name = 'cormorant_6_19.shp'
    dbf_name = 'cormorant_6_19.dbf'
    #img, out_name = drawBoxes(folder_path, img_name, shp_name)
    #cv2.imshow(img_name, img)
    for point in getPoints(folder_path + shp_name, folder_path + dbf_name):
        print('box:', point)

