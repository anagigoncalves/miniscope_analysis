import numpy as np
import matplotlib.pyplot as p; p.rcParams['toolbar'] = 'None';
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def opa(a, b, scale=True):
    aT = a.mean(0)
    bT = b.mean(0)
    A = a - aT 
    B = b - bT
    if scale:
        # Calculate the Euclidean norm for each column
        aS = np.sum(A * A)**.5
        bS = np.sum(B * B)**.5
        # Normalize the columns by dividing by their respective norms
        A /= aS
        B /= bS
    else:
        # Set scaling factors to 1 if scaling is not performed
        aS = 1.0
        bS = 1.0
    U, _, V = np.linalg.svd(np.dot(B.T, A))
    aR = np.dot(U, V)
    if np.linalg.det(aR) < 0:
        V[1] *= -1
        aR = np.dot(U, V)
    if scale:
        aS = aS / bS
        aT -= (bT.dot(aR) * aS)
    else:
        aT -= bT.dot(aR)
    aD = (np.sum((A - B.dot(aR))**2) / len(a))**.5
    return aR, aS, aT, aD
        
def gpa(v, n=-1):
    if n < 0:
        p = avg(v)
    else:
        p = v[n]
    l = len(v)
    r, s, t, d = np.ndarray((4, l), object)
    for i in range(l):
        r[i], s[i], t[i], d[i] = opa(p, v[i]) 
    return r, s, t, d

def avg(v):
    v_= np.copy(v)
    l = len(v_) 
    R, S, T = [list(np.zeros(l)) for _ in range(3)]
    for i, j in np.ndindex(l, l):
        r, s, t, _ = opa(v_[i], v_[j]) 
        R[j] += np.arccos(min(1, max(-1, np.trace(r[:1])))) * np.sign(r[1][0]) 
        S[j] += s 
        T[j] += t 
    for i in range(l):
        a = R[i] / l
        r = [np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]
        v_[i] = v_[i].dot(r) * (S[i] / l) + (T[i] / l) 
    return v_.mean(0)


a = np.array(avg_trajectory['8'])
b = np.array(avg_trajectory['1'])
aR, aS, aT, aD = opa(a, b, scale = False)
aligned_a = (a - aT) * aS @ aR.T
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(a[0], a[1], a[2], c='crimson', linewidth = 2.5, label='a', alpha = 0.3)
ax.plot(b[0], b[1], b[2], c='black', linewidth = 2.5, label='b')
ax.plot(aligned_a[0], aligned_a[1], aligned_a[2], c='crimson', linewidth = 2.5, label='Aligned a')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.legend()
plt.show()


def get_rotation_angle(a, b):
    cross_covariance_matrix = np.dot(a, b.T)
    u, _, vh = np.linalg.svd(cross_covariance_matrix)
    rotation_matrix = np.dot(vh.T, u.T)
    rotation_angle_rad = np.arccos((np.trace(rotation_matrix) - 1) / 2)
    rotation_angle_deg = np.degrees(rotation_angle_rad)
    axis = (rotation_matrix - rotation_matrix.T) / (2 * np.sin(rotation_angle_rad))
    return rotation_angle_deg
