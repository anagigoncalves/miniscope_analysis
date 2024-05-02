import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Input data
save_path = 'J:\\Thesis\\for figures\\'
path_session_data = 'J:\\Miniscope processed files'
animals = ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
session_data = pd.read_excel(os.path.join(path_session_data, 'session_data_tied_S1.xlsx'))
protocol = 'tied baseline S1'
# for the order ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
fov_coords = np.array([[6.12, 0.5],
                     [6.24, 1],
                     [6.64, 1],
                     [6.48, 1.5],
                     [6.48, 1.7]]) #AP, ML

os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

cmap = plt.get_cmap('magma')
color_animals = [cmap(i) for i in np.linspace(0, 1, 6)]
def get_colors_plot(animal_name, color_animals):
    if animal_name=='MC8855':
        color_plot = color_animals[0]
    if animal_name=='MC9194':
        color_plot = color_animals[1]
    if animal_name=='MC10221':
        color_plot = color_animals[2]
    if animal_name=='MC9513':
        color_plot = color_animals[3]
    if animal_name=='MC9226':
        color_plot = color_animals[4]
    return color_plot

#Get coordinates for all ROIs
roi_coordinates = []
animal_id = []
for count_a, animal in enumerate(animals):
    session_data_idx = np.where(session_data['animal'] == animal)[0][0]
    ses_info = session_data.iloc[session_data_idx, :]
    date = ses_info[3]
    path = os.path.join(path_session_data, 'TM RAW FILES', ses_info[0], ses_info[1], date + '\\')
    mscope = miniscope_session_class.miniscope_session(path)
    path_loco = os.path.join(path_session_data, 'TM TRACKING FILES',
                             ses_info[0] + ' ' + ses_info[2] + ' ' + date.split('_')[-1] + date.split('_')[-2] +
                             date.split('_')[-3][2:] + '\\')
    loco = locomotion_class.loco_class(path_loco)
    session = loco.get_session_id()
    # Compute ROI coordinates
    coord_ext = np.load(os.path.join(mscope.path, 'processed files', 'coord_ext.npy'), allow_pickle=True)
    centroid_ext = mscope.get_roi_centroids(coord_ext)
    centroid_ext_arr = np.array(centroid_ext)
    #Flip coords horizontally and vertically because image in miniscope is flipped
    centroid_ext_flip = np.zeros(np.shape(centroid_ext_arr))
    centroid_ext_flip[:, 1] = 1000-centroid_ext_arr[:, 0]
    centroid_ext_flip[:, 0] = 1000-centroid_ext_arr[:, 1]
    #Need to swap again, because now ML and AP are swapped
    #Adjust for the FOV coordinates to get global coordinates
    centroid_ext_swap = np.array(centroid_ext_flip)[:, [1, 0]] 
    fov_coord = fov_coords[count_a]
    fov_corner = np.array([fov_coord[1] - 0.5, fov_coord[0] - 0.5]) #ML is the centroid[:, 0] and AP the centroid[:, 1]
    centroid_dist_corner = (np.array(centroid_ext_swap) * 0.001) + fov_corner
    roi_coordinates.extend(centroid_dist_corner)
    for i in range(len(centroid_dist_corner)):
        animal_id.append(get_colors_plot(animal, color_animals))
roi_coordinates_arr = np.array(roi_coordinates)

fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
sc = ax.scatter(roi_coordinates_arr[:, 0], roi_coordinates_arr[:, 1], c=animal_id, s=15)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.gca().invert_yaxis()
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_ylabel('AP coordinate (mm)', fontsize=20)
ax.set_xlabel('ML coordinate (mm)', fontsize=20)
plt.savefig(os.path.join(save_path, 'roilocation_'+protocol.replace(' ', '_')), dpi=256)
plt.savefig(os.path.join(save_path, 'roilocation_'+protocol.replace(' ', '_')+'.svg'), dpi=256)