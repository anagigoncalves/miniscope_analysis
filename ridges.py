from mimetypes import init
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
a = [ 126,  249,  324,  467,  632,  698,  763,  803,  908,  987, 1042, 1192, 1292, 1356, 1474, 1592, 1680, 1757]
fa = 0.0950018
wa = np.floor(1/fa)
b = [ 126,  260,  318,  467,  632,  693,  795,  907,  987, 1048, 1188,1292, 1357, 1471, 1584, 1671, 1757]
fb = 0.08620368
wb = np.floor(1/fb)
c = [ 128,  309,  472,  685,  789,  906,  984, 1051, 1185, 1296, 1470,1584, 1758]
fc = 0.07822035
wc = np.floor(1/fc)
init_intervals = []
for i in c:
    interval = [-wc+i,wc+i]
    init_intervals.append(interval)
init_intervals = np.array(init_intervals)

matrix = np.zeros((3, 1759))
matrix[0][a]=1
matrix[1][b]=1
matrix[2][c]=1

plt.imshow(matrix, aspect='auto')
plt.show()



    
