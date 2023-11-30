# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 16:35:58 2023

@author: User
"""

# Average_firing_rate is a matrix (time x paws, ROIs)
pca = PCA(n_components=9)
temporal_factors = pca.fit_transform(Average_firing_rate) # Matrix size (time x paws, n_components)
neuron_factors = pca.components_ # Matrix of weights of size (n_components, ROIs)