# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = 'J:\\Titer analysis\\session files\\'
days = ['08', '10', '14', '17', '20', '23', '27']
dils = ['dilution_1_to_10', 'dilution_1_to_50', 'dilution_1_to_100']

# fr_ref = 2
# isi_vector = np.ones(10000)*(1/fr_ref)+np.random.uniform(low=0, high=0.01, size=10000)*np.random.choice(np.array([1, -1]), 1)
# isi_series = pd.Series(isi_vector)
# fr_ref_comp = 1/np.nanmean(isi_vector)
# cv_ref = np.std(isi_vector)/np.mean(isi_vector)
# cv2_ref = np.nanmean(2*(isi_series.diff().abs())/isi_series.rolling(2).sum())

roi_nr_days = np.zeros((len(dils),len(days)))
fr_days = []
cv_days = []
cv2_days = []
perc90_days = []
skew_days = []
maxF_days = []
for count_dil, dil in enumerate(dils):
    fr = []
    cv = []
    cv2 = []
    perc90 = []
    skew = []
    maxF = []
    for count_d, d in enumerate(days):
        roi_nr_days[count_dil, count_d] = np.load(os.path.join(path, dil + '_day_' + d + '_roinr.npy'))
        fr.append(np.load(os.path.join(path, dil + '_day_' + d + '_fr.npy')))
        cv.append(np.load(os.path.join(path, dil + '_day_' + d + '_cv.npy')))
        cv2.append(np.load(os.path.join(path, dil + '_day_' + d + '_cv2.npy')))
        perc90.append(np.load(os.path.join(path, dil + '_day_' + d + '_perc90.npy')))
        skew.append(np.load(os.path.join(path, dil + '_day_' + d + '_skew.npy')))
        maxF.append(np.load(os.path.join(path, dil + '_day_' + d + '_maxF.npy')))
    fr_days.append(fr)
    cv_days.append(cv)
    cv2_days.append(cv2)
    perc90_days.append(perc90)
    skew_days.append(skew)
    maxF_days.append(maxF)

fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
for count_dil, dil in enumerate(dils):
    plt.scatter(np.arange(len(days)), roi_nr_days[count_dil, :]/np.nanmax(roi_nr_days[count_dil, :]), s=80, label=dil)
    plt.plot(roi_nr_days[count_dil, :]/np.nanmax(roi_nr_days[count_dil, :]), label=dil)
ax.set_xlabel('Days post surgery', fontsize=20)
ax.set_ylabel('Number of ROIs\n(normalized)', fontsize=20)
#ax.legend(dils, frameon=False, fontsize=20)
ax.set_xticks(np.arange(len(days)))
ax.set_xticklabels(days)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=20)
#plt.savefig('D:\\Titer analysis\\roi_nr', dpi=256)
plt.savefig('J:\\Thesis\\figuresChapter2\\titer_analysis_roi_nr', dpi=256)
plt.savefig('J:\\Thesis\\figuresChapter2\\titer_analysis_roi_nr.svg', dpi=256)

fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
fr_days_mean = np.zeros((len(dils),len(days)))
fr_days_std = np.zeros((len(dils),len(days)))
for count_dil, dil in enumerate(dils):
    for d in range(len(days)):
    #     plt.scatter(np.ones(len(fr_days[d]))*d+np.random.uniform(low=0,high=0.5,size=len(fr_days[d]))-0.25, fr_days[d], s=10, color='black')
        fr_days_mean[count_dil, d] = np.mean(fr_days[count_dil][d])
        fr_days_std[count_dil, d] = np.std(fr_days[count_dil][d])
# plt.axhline(y=fr_ref_comp, color='red', linewidth=2)
for count_dil, dil in enumerate(dils):
    plt.plot(np.arange(len(days)), fr_days_mean[count_dil, :], linewidth=2, label=dil)
    plt.fill_between(np.arange(len(days)), fr_days_mean[count_dil, :]-fr_days_std[count_dil, :], fr_days_mean[count_dil, :]+fr_days_std[count_dil, :], alpha=0.3)
ax.set_xlabel('Days post surgery', fontsize=20)
ax.set_ylabel('Calcium event\nrate (Hz)', fontsize=20)
#ax.legend(dils, frameon=False, fontsize=16)
ax.set_xticks(np.arange(len(days)))
ax.set_xticklabels(days)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=20)
#plt.savefig('D:\\Titer analysis\\fr', dpi=256)
plt.savefig('J:\\Thesis\\figuresChapter2\\titer_analysis_fr', dpi=256)
plt.savefig('J:\\Thesis\\figuresChapter2\\titer_analysis_fr.svg', dpi=256)

fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
cv_days_mean = np.zeros((len(dils),len(days)))
cv_days_std = np.zeros((len(dils),len(days)))
for count_dil, dil in enumerate(dils):
    for d in range(len(days)):
        cv_days_mean[count_dil, d] = np.mean(cv_days[count_dil][d])
        cv_days_std[count_dil, d] = np.std(cv_days[count_dil][d])
for count_dil, dil in enumerate(dils):
    plt.plot(np.arange(len(days)), cv_days_mean[count_dil, :], linewidth=2, label=dil)
    plt.fill_between(np.arange(len(days)), cv_days_mean[count_dil, :]-cv_days_std[count_dil, :], cv_days_mean[count_dil, :]+cv_days_std[count_dil, :], alpha=0.3)
ax.set_xlabel('Days post surgery', fontsize=16)
ax.set_ylabel('Coefficient of variation', fontsize=16)
#ax.legend(dils, frameon=False, fontsize=16)
ax.set_xticks(np.arange(len(days)))
ax.set_xticklabels(days)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=16)
#plt.savefig('D:\\Titer analysis\\cv', dpi=256)

fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
cv2_days_mean = np.zeros((len(dils),len(days)))
cv2_days_std = np.zeros((len(dils),len(days)))
for count_dil, dil in enumerate(dils):
    for d in range(len(days)):
        cv2_days_mean[count_dil, d] = np.mean(cv2_days[count_dil][d])
        cv2_days_std[count_dil, d] = np.std(cv2_days[count_dil][d])
for count_dil, dil in enumerate(dils):
    plt.plot(np.arange(len(days)), cv2_days_mean[count_dil, :], linewidth=2, label=dil)
    plt.fill_between(np.arange(len(days)), cv2_days_mean[count_dil, :]-cv2_days_std[count_dil, :], cv2_days_mean[count_dil, :]+cv2_days_std[count_dil, :], alpha=0.3)
ax.set_xlabel('Days post surgery', fontsize=16)
ax.set_ylabel('CV2', fontsize=16)
#ax.legend(dils, frameon=False, fontsize=16)
ax.set_xticks(np.arange(len(days)))
ax.set_xticklabels(days)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=16)
#plt.savefig('D:\\Titer analysis\\cv2', dpi=256)

fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
perc90_days_mean = np.zeros((len(dils),len(days)))
perc90_days_std = np.zeros((len(dils),len(days)))
for count_dil, dil in enumerate(dils):
    for d in range(len(days)):
        perc90_days_mean[count_dil, d] = np.mean(perc90_days[count_dil][d])
        perc90_days_std[count_dil, d] = np.std(perc90_days[count_dil][d])
for count_dil, dil in enumerate(dils):
    plt.plot(np.arange(len(days)), perc90_days_mean[count_dil, :], linewidth=2, label=dil)
    plt.fill_between(np.arange(len(days)), perc90_days_mean[count_dil, :]-perc90_days_std[count_dil, :], perc90_days_mean[count_dil, :]+perc90_days_std[count_dil, :], alpha=0.3)
ax.set_xlabel('Days post surgery', fontsize=20)
ax.set_ylabel('90 percentile of\ninter-event interval (s)', fontsize=20)
#ax.legend(dils, frameon=False, fontsize=16)
ax.set_xticks(np.arange(len(days)))
ax.set_xticklabels(days)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=20)
#plt.savefig('D:\\Titer analysis\\perc90', dpi=256)
plt.savefig('J:\\Thesis\\figuresChapter2\\titer_analysis_isi_perc90', dpi=256)
plt.savefig('J:\\Thesis\\figuresChapter2\\titer_analysis_isi_perc90.svg', dpi=256)

fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
skew_days_mean = np.zeros((len(dils),len(days)))
skew_days_std = np.zeros((len(dils),len(days)))
for count_dil, dil in enumerate(dils):
    for d in range(len(days)):
        skew_days_mean[count_dil, d] = np.mean(skew_days[count_dil][d])
        skew_days_std[count_dil, d] = np.std(skew_days[count_dil][d])
for count_dil, dil in enumerate(dils):
    plt.plot(np.arange(len(days)), skew_days_mean[count_dil, :], linewidth=2, label=dil)
    plt.fill_between(np.arange(len(days)), skew_days_mean[count_dil, :]-skew_days_std[count_dil, :], skew_days_mean[count_dil, :]+skew_days_std[count_dil, :], alpha=0.3)
ax.set_xlabel('Days post surgery', fontsize=16)
ax.set_ylabel('Skewness of fluorescence\nsignal', fontsize=16)
#ax.legend(dils, frameon=False, fontsize=16)
ax.set_xticks(np.arange(len(days)))
ax.set_xticklabels(days)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=16)
#plt.savefig('D:\\Titer analysis\\skew', dpi=256)

fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
maxF_days_mean = np.zeros((len(dils),len(days)))
maxF_days_std = np.zeros((len(dils),len(days)))
for count_dil, dil in enumerate(dils):
    for d in range(len(days)):
        maxF_days_mean[count_dil, d] = np.mean(maxF_days[count_dil][d])
        maxF_days_std[count_dil, d] = np.std(maxF_days[count_dil][d])
for count_dil, dil in enumerate(dils):
    plt.plot(np.arange(len(days)), maxF_days_mean[count_dil, :], linewidth=2, label=dil)
    plt.fill_between(np.arange(len(days)), maxF_days_mean[count_dil, :]-maxF_days_std[count_dil, :], maxF_days_mean[count_dil, :]+maxF_days_std[count_dil, :], alpha=0.3)
ax.set_xlabel('Days post surgery', fontsize=16)
ax.set_ylabel('Maximum of fluorescence\nsignal', fontsize=16)
ax.set_xticks(np.arange(len(days)))
ax.set_xticklabels(days)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=16)
#plt.savefig('D:\\Titer analysis\\maxF', dpi=256)
