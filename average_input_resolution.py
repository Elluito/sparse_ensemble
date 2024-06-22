import PIL
from PIL import Image
import os

widths = []
heights = []

for img in os.listdir(""):
    img_path = os.path.join("")  # Making image file path
    im = Image.open(img_path)
    widths.append(im.size[0])
    heights.append(im.size[1])

AVG_HEIGHT = round(sum(heights) / len(heights))

AVG_WIDTH = round(sum(widths) / len(widths))

print("Average Height: {}".format(AVG_HEIGHT))

print("Average Width: {}".format(AVG_WIDTH))
