from read_roi import read_roi_file
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

width = 84
height = 128

ROIFilename = '15_ADC'

# read roi file
roi_object = read_roi_file(ROIFilename + '.roi')
roi_temp = roi_object[ROIFilename]

# make polygon
polygon = []
for x, y in zip(roi_temp['x'], roi_temp['y']):
    polygon.append(x)
    polygon.append(y)

# make mask
img = Image.new('L', (width, height), 0)
ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
mask = np.array(img)

# display
plt.imshow(mask, cmap='gray')
plt.show()
