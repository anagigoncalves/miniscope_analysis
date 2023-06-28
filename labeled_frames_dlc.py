import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

project_path = 'C:\\Users\\Ana\\Documents\\PhD\\Projects\\DeepLocoTM_MdV_Miniscopes_ClosedLoop\\tmLOCO-dd-2019-08-29\\labeled-data\\'

animal_folder = 'MC16946_60_25_0.15_0.15_tied_1_2'
img_name = 'img13229.png'
csv_name = 'CollectedData_dd.csv'
#load image
im = plt.imread(os.path.join(project_path, animal_folder, img_name))
plt.imshow(im)

#load csv
labeled_data = pd.read_csv(os.path.join(project_path, animal_folder, csv_name), header=1)
img_path = os.path.join('labeled-data', animal_folder, img_name)
coord_row = np.where(labeled_data.iloc[:, 0] == img_path)[0][0]
coords = labeled_data.iloc[coord_row, 1::2]
