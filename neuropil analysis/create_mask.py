from numpy import asarray
import os
import pandas as pd
from PIL import Image
path = '/media/careylab/Samsung_T5/TM RAW FILES/split ipsi fast/MC8855/2021_04_05/Neuropil_analysis/'
mask_vessel = Image.open(os.path.join(path,'Mask_vessel.png'))
mask_vessel = asarray(mask_vessel)>0

mask = Image.open(os.path.join(path,'Mask_shape.png'))
mask = asarray(mask)
mask[mask_vessel] = 0
mask = Image.fromarray(mask)
mask.save("Mask.png", format="png")


