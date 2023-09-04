# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry as geometry
from shapely.ops import cascaded_union, polygonize
from scipy.spatial import Delaunay
from descartes import PolygonPatch
import pandas as pd
from itertools import chain
import warnings
warnings.filterwarnings('ignore')


def alpha_shape(points, only_outer, alpha):
 """
 Compute the alpha shape (concave hull) of a set
 of points.

 param points: Iterable container of points.
 param alpha: alpha value to influence the
     gooeyness of the border. Smaller numbers
     don't fall inward as much as larger numbers.
     Too large, and you lose everything!
 only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
 """
 assert points.shape[0] > 3, "Need at least four points"
 def add_edge(edges, i, j):
  """
  Add an edge between the i-th and j-th points,
  if not in the list already
  """
  if (i, j) in edges or (j, i) in edges:
   # already added
   assert (j, i) in edges, "Can't go twice over same directed edge right?"
   if only_outer:
    # if both neighboring triangles are in shape, it's not a boundary edge
    edges.remove((j, i))
   return
  edges.add((i, j))
  # edge_points.append(points[i, j])

 tri = Delaunay(points)
 edges = set()
 edge_points = []
 # loop over triangles:
 # ia, ib, ic = indices of corner points of the
 # triangle
 for ia, ib, ic in tri.vertices:
  pa = points[ia]
  pb = points[ib]
  pc = points[ic]

  # Lengths of sides of triangle
  a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
  b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
  c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)

  # Semiperimeter of triangle
  s = (a + b + c) / 2.0

  # Area of triangle by Heron's formula
  area = np.sqrt(s * (s - a) * (s - b) * (s - c))
  circum_r = np.divide(a * b * c, (4.0 * area))

  # Here's the radius filter.
  # print circum_r
  if circum_r < (1.0 / alpha):
   add_edge(edges, ia, ib)
   add_edge(edges, ib, ic)
   add_edge(edges, ic, ia)
 return edges

def find_edges_with(i, edge_set):
  i_first = [j for (x, j) in edge_set if x == i]
  i_second = [j for (j, x) in edge_set if x == i]
  return i_first, i_second


def stitch_boundaries(edges):
 edge_set = edges.copy()
 boundary_lst = []
 while len(edge_set) > 0:
  boundary = []
  edge0 = edge_set.pop()
  boundary.append(edge0)
  last_edge = edge0
  while len(edge_set) > 0:
   i, j = last_edge
   j_first, j_second = find_edges_with(j, edge_set)
   if j_first:
    edge_set.remove((j, j_first[0]))
    edge_with_j = (j, j_first[0])
    boundary.append(edge_with_j)
    last_edge = edge_with_j
   elif j_second:
    edge_set.remove((j_second[0], j))
    edge_with_j = (j, j_second[0])  # flip edge rep
    boundary.append(edge_with_j)
    last_edge = edge_with_j

   if edge0[0] == last_edge[1]:
    break

  boundary_lst.append(boundary)
 return boundary_lst

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel('J:\\Miniscope processed files\\session_data_MC8855.xlsx')
# for s in range(len(session_data)):
s = 0
ses_info = session_data.iloc[s, :]
date = ses_info[3]
# path inputs
path = os.path.join(path_session_data, 'TM RAW FILES', ses_info[0], ses_info[1], date + '\\')
path_loco = os.path.join(path_session_data, 'TM TRACKING FILES', ses_info[0] + ' ' + ses_info[2] + ' ' + date.split('_')[-1]+date.split('_')[-2]+date.split('_')[-3][2:] + '\\')
session_type = path.split('\\')[-4].split(' ')[0]
mscope = miniscope_session_class.miniscope_session(path)
loco = locomotion_class.loco_class(path_loco)

# Session data and inputs
animal = mscope.get_animal_id()
session = loco.get_session_id()
traces_type = 'raw'
[df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace, coord_ext, reg_th, reg_bad_frames, trials,
 clusters_rois, colors_cluster, colors_session, idx_roi_cluster_ordered, ref_image, frames_dFF] = mscope.load_processed_files()

cluster = 1
idx_cluster = np.where(idx_roi_cluster_ordered == cluster)[0]
rois_coordinates_cluster = np.array(list(chain.from_iterable(coord_ext[idx_cluster])))

edges = alpha_shape(rois_coordinates_cluster, 1, alpha=0.4)
plt.figure()
plt.scatter(rois_coordinates_cluster[:, 0],rois_coordinates_cluster[:, 1], color='blue')
edges_coordinates = []
for i, j in edges:
 edges_coordinates.append([rois_coordinates_cluster[[i, j], 0], rois_coordinates_cluster[[i, j], 1]])
edges_coordinates_array = np.array(edges_coordinates)
plt.scatter(edges_coordinates_array[:, 0], edges_coordinates_array[:, 1], color='black')

