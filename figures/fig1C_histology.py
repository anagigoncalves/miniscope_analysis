import matplotlib.pyplot as plt
import matplotlib.colors as mpcol
import numpy as np

from vedo import embedWindow  # for more explanations about these two lines checks the notebooks workflow example
embedWindow(None)

import brainrender
from brainrender import Scene
from brainrender.actors import Points

brainrender.settings.BACKGROUND_COLOR = [1, 1, 1]  # change rendering background color
brainrender.settings.WHOLE_SCREEN = (False)   # make the rendering window be smaller
brainrender.settings.SHOW_AXES = False  # turn off the axes display
brainrender.settings.SHADER_STYLE = 'plastic'

scene = Scene(atlas_name='allen_mouse_100um')
res = 100
# cb = scene.add_brain_region("VERM", alpha=0.5, color='blue')
# Animal colors
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


# Add to scene
animal_order = ['MC8855', 'MC9194', 'MC10221', 'MC9226', 'MC9513']
ipts_all = np.array([[6.12, 0, -0.5],
                     [6.24, 0, -1],
                     [6.48, 0, -1.7],
                     [6.64, 0, -1],
                     [6.48, 0, -1.5],]) #AP, DV, ML
# from looking at Paxinos here are the coordinates - should load paxino's mesh
# ipts_all = np.array([[6.12, 0, -1],
#                      [6.24, 0, -0.5],
#                      [6.64, 0, -2],
#                      [6.12, 0, -1.5],
#                      [6.48, 0, -1],]) #AP, DV, ML
color_hex = []
for c in animal_order:
    color_hex.append(mpcol.rgb2hex(get_colors_plot(c, color_animals)))

ipts_all_transform = (ipts_all*1000)+np.tile(np.array([5400, 0, 5700]), (np.shape(ipts_all)[0], 1))
scene.add(Points(ipts_all_transform, name=animal_order, colors=color_hex, radius=300, alpha=0.75))

# render
scale = 2
scene.content
scene.render(camera='top', zoom=0.995)
# scene.screenshot(name="miniscope_histology_locations", scale=scale)
# scene.close()

#CITE
# https://www.sciencedirect.com/science/article/pii/S0092867420304025#cebib0010
# https://elifesciences.org/articles/65751
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8137149/
