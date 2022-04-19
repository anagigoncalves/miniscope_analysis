from neuropil_class import Neuropil
import numpy as np
import matplotlib.pyplot as plt

path = 'D:/TM RAW FILES/split ipsi fast/MC8855/2021_04_05/Registered video/'
trial = 1

neuropil = Neuropil(path, trial)
# neuropil.grid_division()
cluster_image = neuropil.neuropil_clustering(save = False)
plt.imshow(cluster_image, aspect='auto', cmap='hsv')
plt.show()

# neuropil.Ca_events_neuropil()