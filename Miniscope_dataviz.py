# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 14:13:38 2021

@author: Ana
"""
import os
import numpy as np
import PySimpleGUI as sg
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from scipy.spatial import ConvexHull
import read_roi
import mat73
import cv2
import imageio
import matplotlib.patches as patches
import glob

# path inputs
path = 'E:\\TM RAW FILES\\split ipsi fast\\MC8855\\2021_04_05\\'
path_loco = 'E:\\TM TRACKING FILES\\split ipsi fast\\split ipsi fast S1 050421\\'
os.chdir('C:\\Users\\Ana\\PycharmProjects\\MSCOPEproject\\')
import miniscope_session_class
mscope = miniscope_session_class.miniscope_session(path)
import SlopeThreshold as ST
from statsmodels import robust

class Miniscope_dataviz():

    OutTextElem = None
    OutPrintTxt = 'Press Go! to start GUI and use mouse scroll to move Frame Slider. Can press play video to see frame playback'

    def __init__(self):
        sg.theme('LightGrey1')
        self.gui_win = self.GUI_Init()
        self.figure = False
        self.ax = False

    def GUI_Init(self):        
        self.OutTextElem = sg.Text(self.OutPrintTxt,size=(100,1),relief=sg.RELIEF_SUNKEN)
        layout = [[self.OutTextElem],
                  [sg.Text('Frame slider',size=(12,1),font=('default', 10, ''),pad=((5,0),(18,0))),
                   sg.Slider(range=(0,1800),size=(60,10),
                   default_value=0,enable_events=True,resolution=1,
                   orientation='h',key='Slider')],
                  [sg.Radio('Suite2p', "Suite2pON", key = 'Suite2pAnalysis', default=False, pad=((700,0),(0,0)))],
                  [sg.Radio('Fiji', "FijiON", key = 'FijiAnalysis',default=False, pad=((700,0),(0,0)))],
                  [sg.Radio('EXTRACT', "EXTRACTON", key = 'EXTRACTAnalysis',default=False, pad=((700,0),(0,0)))],
                  [sg.Radio('PROCESSED', "PROCESSED", key='ProcessedFilesAnalysis', default=False, pad=((700, 0), (0, 0)))],
                  [sg.Button('Play video imaging and wait',key='VideoImg')],
                  [sg.Button('Stop video imaging',key='StopI')],
                  [sg.Button('Restart video imaging',key='RestartI')],
                  [sg.Button('Go!',key='Plot')],
                  [sg.Button('Play video behavior and wait',key='VideoBeh')],
                  [sg.Button('Stop video behavior',key='StopB')],
                  [sg.Button('Restart video behavior',key='RestartB')],
                  [sg.Checkbox('Behavior', default=False, key='Behavior', pad=((700,0),(0,0)))],
                  [sg.Checkbox('Imaging', default=True, key='Imaging', pad=((700,0),(0,0)))],
                  [sg.Text('ROI'), sg.InputText(size=(30,1),font=('default', 10, ''),key='ROI')],
                  [sg.Text('Trial'), sg.InputText(size=(30,1),font=('default', 10, ''),key='Trial')]]
        self.window = sg.Window('Miniscope data visualization', layout, 
            return_keyboard_events=True, finalize=True)        
        self.window['Slider'].bind('<Enter>', '+UNDER CURSOR')
        self.focus = None        
        return self.window
    
    def get_registered_frames(self):
        """Function to get the array with the registered frames for a certain trial"""
        self.trial_frames_full =  tiff.imread(self.reg_path+'T'+str(self.trial)+'_reg.tif')        
        return self.trial_frames_full
    
    def plot_video(self):
        self.axes_lst[0][1].clear()
        self.axes_lst[0][1].imshow(self.trial_frames_full[self.slider_value,:,:], cmap=cm.Greys_r, extent=[0,np.shape(self.ref_image)[1]/mscope.pixel_to_um,np.shape(self.ref_image)[0]/mscope.pixel_to_um,0])
        self.axes_lst[0][1].set_title('ROIs '+str(self.rois)+' trial '+str(self.trial),fontsize = 28)
        self.fig.canvas.draw_idle()

    def plot_video_mask(self):
        self.axes_lst[0][2].clear()
        self.axes_lst[0][2].plot(self.coord_cell_crop[self.hull.vertices,0], self.coord_cell_crop[self.hull.vertices,1], color='orange')
        self.axes_lst[0][2].imshow(self.trial_frames_full_mask[self.slider_value,:,:], cmap=cm.Greys_r)
        self.fig.canvas.draw_idle()

    def get_video_mask(self):
        if self.EXTRACT or self.Fiji:
            x_coord_cent = np.int64(np.nanmean(self.coord_cell[self.rois-1][:,0])*mscope.pixel_to_um)
            y_coord_cent = np.int64(np.nanmean(self.coord_cell[self.rois-1][:,1])*mscope.pixel_to_um)
            x_coord = np.int64(self.coord_cell[self.rois - 1][:, 0] * mscope.pixel_to_um)
            y_coord = np.int64(self.coord_cell[self.rois - 1][:, 1] * mscope.pixel_to_um)
        if self.S2P:
            x_coord_cent = np.int64(self.centroid_cell[self.rois-1][1]*mscope.pixel_to_um)
            y_coord_cent = np.int64(self.centroid_cell[self.rois-1][0]*mscope.pixel_to_um)
            x_coord = np.int64(self.coord_cell[self.rois - 1][:, 0] * mscope.pixel_to_um)
            y_coord = np.int64(self.coord_cell[self.rois - 1][:, 1] * mscope.pixel_to_um)
        if self.PROCESSED:
            roi_list = self.dFF_trial.columns[2:]
            roi_result = [i for i in range(len(roi_list)) if 'ROI'+str(self.rois) in roi_list[i]]
            roi_idx = roi_result[0]
            x_coord_cent = np.int64(np.nanmean(self.coord_cell[roi_idx][:,0])*mscope.pixel_to_um)
            y_coord_cent = np.int64(np.nanmean(self.coord_cell[roi_idx][:,1])*mscope.pixel_to_um)
            x_coord = np.int64(self.coord_cell[roi_idx][:,0]*mscope.pixel_to_um)
            y_coord = np.int64(self.coord_cell[roi_idx][:,1]*mscope.pixel_to_um)
        coord_cell_crop_x = x_coord-(x_coord_cent-50)
        coord_cell_crop_y = y_coord-(y_coord_cent-50)
        self.crop_region = np.array([x_coord_cent-50,x_coord_cent+50,y_coord_cent-50,y_coord_cent+50])/mscope.pixel_to_um
        self.width = 100
        self.coord_cell_crop = np.transpose(np.vstack((coord_cell_crop_x,coord_cell_crop_y))) 
        self.hull = ConvexHull(self.coord_cell_crop)
        self.trial_frames_full_mask = self.trial_frames_full[:,y_coord_cent-50:y_coord_cent+50,x_coord_cent-50:x_coord_cent+50]
                    
    def plot_rois_reg_image(self):
        """Plots the desired ROIs on top of reference image"""
        if self.PROCESSED:
            roi_list = self.dFF_trial.columns[2:]
            roi_result = [i for i in range(len(roi_list)) if 'ROI'+str(self.rois) in roi_list[i]]
            roi_idx = roi_result[0]
            x_coord = np.int64(self.coord_cell[roi_idx][:,0]*mscope.pixel_to_um)
            y_coord = np.int64(self.coord_cell[roi_idx][:,1]*mscope.pixel_to_um)
        if self.EXTRACT or self.Fiji or self.S2P:
            x_coord = np.int64(self.coord_cell[self.rois-1][:,0]*mscope.pixel_to_um)
            y_coord = np.int64(self.coord_cell[self.rois-1][:,1]*mscope.pixel_to_um)
        self.axes_lst[0][0].clear()
        rect = patches.Rectangle((self.crop_region[0],self.crop_region[2]),self.width/mscope.pixel_to_um,self.width/mscope.pixel_to_um,linewidth=2,edgecolor='red',facecolor='none')    
        self.axes_lst[0][0].imshow(self.ref_image,cmap='gray', extent=[0,np.shape(self.ref_image)[1]/mscope.pixel_to_um,np.shape(self.ref_image)[0]/mscope.pixel_to_um,0])
        self.axes_lst[0][0].add_patch(rect)
        self.axes_lst[0][0].scatter(x_coord/mscope.pixel_to_um,y_coord/mscope.pixel_to_um,color = 'orange',s=0.1)
        self.fig.canvas.draw_idle()
            
    def get_calcium_trace_events(self):
        """Get the corresponding calcium trace and events for that trial and ROI"""
        r = 'ROI'+str(self.rois)
        self.data_dFF =  np.array(self.dFF_trial.loc[self.dFF_trial['trial']==self.trial,r])
        timeT = 7
        rois_list = self.dFF_trial.columns[2:]
        count_r = 0
        for roi in rois_list:
            if roi == r:
                idx_roi = count_r
            count_r += 1
        amp_arr = np.load(path+'\\processed files\\amplitude_events.npy')
        if len(np.shape(amp_arr))==2:
            amp = amp_arr[idx_roi,self.trial-1]
        if len(np.shape(amp_arr))==1:
            amp = amp_arr[idx_roi]
        [JoinedPosSet_all, JoinedNegSet_all, F_Values_all] = ST.SlopeThreshold(self.data_dFF, amp, timeT,
                                                                               CollapSeq=True, acausal=False,
                                                                               verbose=0, graph=None)
        peaks = []
        for i in range(len(JoinedPosSet_all)):
            peak_idx = np.argmax(self.data_dFF[JoinedPosSet_all[i][0]:JoinedPosSet_all[i][0] + timeT * 2])
            peaks.append(JoinedPosSet_all[i][0] + peak_idx)
        self.data_events = self.frame_time[self.trial-1][np.array(peaks)]
        return self.data_dFF, self.data_events
    
    def plot_trace_events(self):
        self.axes_lst[0][3].clear()
        self.axes_lst[0][3].plot(self.frame_time[self.trial-1], self.data_dFF,color='black')
        self.axes_lst[0][3].scatter(self.frame_time[self.trial-1][self.slider_value], self.data_dFF[self.slider_value],s = 45, color='orange')
        for s in range(len(self.data_events)):
             self.axes_lst[0][3].axvline(self.data_events[s],ymin = 0, ymax = 0.1,color='orange')            
        self.fig.canvas.draw_idle()   
        # cid = self.fig.canvas.mpl_connect('button_press_event', Miniscope_dataviz.onclick)
        
    def plot_animations(self):
        fps_set = 120
        if self.EXTRACT or self.Fiji:
            x_coord_cent = np.int64(np.nanmean(self.coord_cell[self.rois-1][:,0])*mscope.pixel_to_um)
            y_coord_cent = np.int64(np.nanmean(self.coord_cell[self.rois-1][:,1])*mscope.pixel_to_um)
            x_coord = np.int64(self.coord_cell[self.rois - 1][:, 0] * mscope.pixel_to_um)
            y_coord = np.int64(self.coord_cell[self.rois - 1][:, 1] * mscope.pixel_to_um)
        if self.S2P:
            x_coord_cent = np.int64(self.centroid_cell[self.rois-1][1]*mscope.pixel_to_um)
            y_coord_cent = np.int64(self.centroid_cell[self.rois-1][0]*mscope.pixel_to_um)
            x_coord = np.int64(self.coord_cell[self.rois - 1][:, 0] * mscope.pixel_to_um)
            y_coord = np.int64(self.coord_cell[self.rois - 1][:, 1] * mscope.pixel_to_um)
        if self.PROCESSED:
            roi_list = self.dFF_trial.columns[2:]
            roi_result = [i for i in range(len(roi_list)) if 'ROI'+str(self.rois) in roi_list[i]]
            roi_idx = roi_result[0]
            x_coord_cent = np.int64(np.nanmean(self.coord_cell[roi_idx][:,0])*mscope.pixel_to_um)
            y_coord_cent = np.int64(np.nanmean(self.coord_cell[roi_idx][:,1])*mscope.pixel_to_um)
            x_coord = np.int64(self.coord_cell[roi_idx][:,0]*mscope.pixel_to_um)
            y_coord = np.int64(self.coord_cell[roi_idx][:,1]*mscope.pixel_to_um)
        coord_cell_crop_x = x_coord-(x_coord_cent-50)
        coord_cell_crop_y = y_coord-(y_coord_cent-50)
        self.coord_cell_crop = np.transpose(np.vstack((coord_cell_crop_x,coord_cell_crop_y)))
        self.hull = ConvexHull(self.coord_cell_crop)
        trial_frames_full_mask = self.trial_frames_full[:,y_coord_cent-50:y_coord_cent+50,x_coord_cent-50:x_coord_cent+50]
        self.axes_lst[0][1].clear()
        self.axes_lst[0][2].clear()
        self.axes_lst[0][3].clear()
        for s in range(len(self.data_events)):
            self.axes_lst[0][3].axvline(self.data_events[s],ymin = 0, ymax = 0.1,color='orange') 
        self.axes_lst[0][3].plot(self.frame_time[self.trial-1], self.data_dFF,color='black')
        animation_objects = []
        for i in np.arange(self.slider_value, np.shape(self.trial_frames_full)[0]):
            video_ani = self.axes_lst[0][1].imshow(self.trial_frames_full[i,:,:], cmap=cm.Greys_r, extent=[0,np.shape(self.ref_image)[1]/mscope.pixel_to_um,np.shape(self.ref_image)[0]/mscope.pixel_to_um,0])        
            mask_ani = self.axes_lst[0][2].imshow(trial_frames_full_mask[i,:,:], cmap=cm.Greys_r)        
            ell_ani = self.axes_lst[0][2].scatter(self.coord_cell_crop[self.hull.vertices,0], self.coord_cell_crop[self.hull.vertices,1], color='orange')
            dot_ani = self.axes_lst[0][3].scatter(self.frame_time[self.trial-1][i], self.data_dFF[i],s = 45, color='orange')
            animation_objects.append([video_ani,mask_ani,ell_ani,dot_ani])
        self.ani_frame = animation.ArtistAnimation(self.fig, animation_objects, interval=fps_set, blit=True)
        return self.ani_frame
        
    def stop_animation_img(self):
        self.ani_frame.event_source.stop()
        
    def restart_animation_img(self):
        self.ani_frame.event_source.start()

    def draw_figure(self):
        events, values = self.window.read()
        if values['Imaging']:
            p_fsize = 20
            self.fig, self.axes_lst = plt.subplots(figsize=(15,10))
            self.gs = self.fig.add_gridspec(2,3)
            self.axes_lst.remove()
            self.axes_lst = []
            self.axes_lst.append([self.fig.add_subplot(self.gs[0, 0]),self.fig.add_subplot(self.gs[0, 1]),self.fig.add_subplot(self.gs[0, 2]),self.fig.add_subplot(self.gs[1, :])])       
            self.axes_lst[0][0].set_xlabel('FOV in micrometers', fontsize=p_fsize-4)   
            self.axes_lst[0][0].set_ylabel('FOV in micrometers', fontsize=p_fsize-4)     
            self.axes_lst[0][0].tick_params(axis='x', labelsize = p_fsize-4)
            self.axes_lst[0][0].tick_params(axis='y', labelsize = p_fsize-4)
            self.axes_lst[0][1].set_xlabel('FOV in micrometers', fontsize=p_fsize-4)   
            self.axes_lst[0][1].set_ylabel('FOV in micrometers', fontsize=p_fsize-4)   
            self.axes_lst[0][1].set_title('ROI '+str(self.rois)+' trial '+str(self.trial), fontsize=p_fsize)
            self.axes_lst[0][1].tick_params(axis='x', labelsize = p_fsize-4)
            self.axes_lst[0][1].tick_params(axis='y', labelsize = p_fsize-4) 
            self.axes_lst[0][2].set_xlabel('FOV in pixels', fontsize=p_fsize-4)   
            self.axes_lst[0][2].set_ylabel('FOV in pixels', fontsize=p_fsize-4)     
            self.axes_lst[0][2].tick_params(axis='x', labelsize = p_fsize-4)
            self.axes_lst[0][2].tick_params(axis='y', labelsize = p_fsize-4) 
            self.axes_lst[0][3].set_xlabel('Time (s)', fontsize = p_fsize-4)
            self.axes_lst[0][3].set_ylabel('ΔF/F', fontsize = p_fsize-4)
            self.axes_lst[0][3].spines['right'].set_visible(False)
            self.axes_lst[0][3].spines['top'].set_visible(False)
            self.axes_lst[0][3].tick_params(axis='x', labelsize = p_fsize-4)
            self.axes_lst[0][3].tick_params(axis='y', labelsize = p_fsize-4)        
            plt.show(block=False)
    
        if values['Behavior']:
            p_fsize = 20
            self.fig2, self.axes_lst2 = plt.subplots(figsize=(25,10))
            self.gs2 = self.fig2.add_gridspec(3,2)
            self.axes_lst2.remove()
            self.axes_lst2 = []
            self.axes_lst2.append([self.fig2.add_subplot(self.gs2[0, 0]),self.fig2.add_subplot(self.gs2[2, 0]),self.fig2.add_subplot(self.gs2[0, 1]),self.fig2.add_subplot(self.gs2[1, 1]),self.fig2.add_subplot(self.gs2[2, 1])])       
            self.axes_lst2[0][0].set_xlabel('Pixels', fontsize=p_fsize-4)   
            self.axes_lst2[0][0].set_ylabel('Pixels', fontsize=p_fsize-4) 
            self.axes_lst2[0][0].set_title('Behavior frame at slider value', fontsize=p_fsize)
            self.axes_lst2[0][0].tick_params(axis='x', labelsize = p_fsize-4)
            self.axes_lst2[0][0].tick_params(axis='y', labelsize = p_fsize-4)
            self.axes_lst2[0][1].set_xlabel('Pixels', fontsize=p_fsize-4)   
            self.axes_lst2[0][1].set_ylabel('Pixels', fontsize=p_fsize-4) 
            self.axes_lst2[0][1].set_title('Behavior frame at slider value + 1s', fontsize=p_fsize)
            self.axes_lst2[0][1].tick_params(axis='x', labelsize = p_fsize-4)
            self.axes_lst2[0][1].tick_params(axis='y', labelsize = p_fsize-4)
            self.axes_lst2[0][2].set_xlabel('Time(s)', fontsize=p_fsize-4)   
            self.axes_lst2[0][2].set_ylabel('Yaw (degrees)', fontsize=p_fsize-4)   
            self.axes_lst2[0][2].tick_params(axis='x', labelsize = p_fsize-4)
            self.axes_lst2[0][2].tick_params(axis='y', labelsize = p_fsize-4) 
            self.axes_lst2[0][3].set_xlabel('Time(s)', fontsize=p_fsize-4)   
            self.axes_lst2[0][3].tick_params(axis='x', labelsize = p_fsize-4)
            self.axes_lst2[0][3].tick_params(axis='y', labelsize = p_fsize-4) 
            self.axes_lst2[0][4].set_xlabel('Time (s)', fontsize = p_fsize-4)
            self.axes_lst2[0][4].spines['right'].set_visible(False)
            self.axes_lst2[0][4].spines['top'].set_visible(False)
            self.axes_lst2[0][4].tick_params(axis='x', labelsize = p_fsize-4)
            self.axes_lst2[0][4].tick_params(axis='y', labelsize = p_fsize-4)        
            plt.show(block=False)
            
    def get_video_behavior(self):
        path_split = path.split(path[-1])
        self.animal_name = path_split[-3]
        path_loco_beg = path.split(path[-1])[0]+'\\TM TRACKING FILES\\'+path_split[2]
        date = path_split[-2].split('_')[-1]+path_split[-2].split('_')[-2]+path_split[-2].split('_')[-3][2:]
        dirs = os.listdir(path_loco_beg)
        folder_loco = []
        for d in dirs:
            if d.find(date)>1:
                folder_loco = d
        path_loco = path_loco_beg+'\\'+folder_loco+'\\'
        mp4_files = glob.glob(path_loco+'*.mp4')
        video_path = []
        for f in mp4_files:
            filename = f.split(path[-1])[-1]
            if len(filename)<60:
                filename_split = filename.split('_')
                if filename_split[0] == self.animal_name:
                    if int(filename_split[-1][:-4]) == self.trial:
                        video_path = f
                        self.session = int(filename_split[-2])
        self.video_path = video_path
        vid = imageio.get_reader(video_path)  
        mscope_time1 = self.frame_time[self.trial-1][self.slider_value]
        mscope_time2 = self.frame_time[self.trial-1][self.slider_value]+1 #1s later
        import locomotion_class
        loco = locomotion_class.loco_class(path_loco)
        [trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(self.animal_name,self.session,self.frames_dFF)  
        self.bcam_frame1 = np.abs(bcam_time[self.trial-1] - mscope_time1).argmin()
        self.bcam_frame2 = np.abs(bcam_time[self.trial-1] - mscope_time2).argmin()
        nums = [self.bcam_frame1, self.bcam_frame2]
        array_vid = np.zeros((278,636,nums[1]-nums[0]))
        count_f = 0
        for num in np.arange(nums[0],nums[1]):
            image = vid.get_data(num)
            image_arr = np.mean(image,axis=2)
            array_vid[:,:,count_f] = image_arr
            count_f += 1
        self.frames_bcam = array_vid
    
    def plot_behavior(self):
        p_fsize = 20
        self.axes_lst2[0][0].clear()
        self.axes_lst2[0][0].imshow(self.frames_bcam[:,:,0], cmap='Blues')
        self.axes_lst2[0][0].set_title('Behavior frame at slider value', fontsize=p_fsize)
        self.axes_lst2[0][1].clear()
        self.axes_lst2[0][1].imshow(self.frames_bcam[:,:,np.shape(self.frames_bcam)[2]-1], cmap='Reds')
        self.axes_lst2[0][1].set_title('Behavior frame at slider value + 1s', fontsize=p_fsize)
        self.fig2.canvas.draw_idle()
    
    def plot_acc_values(self):
        p_fsize = 20
        trials = mscope.get_trial_id()
        self.head_angles_raw = mscope.compute_head_angles(trials)
        mscope_time2 = self.frame_time[self.trial-1][self.slider_value]+1 #1s later
        mscope_frame2 = np.abs(self.frame_time[self.trial-1] - mscope_time2).argmin()
        self.axes_lst2[0][2].clear()
        self.axes_lst2[0][2].plot(self.frame_time[self.trial-1][self.slider_value:mscope_frame2],np.rad2deg(np.array(self.head_angles_raw.loc[self.head_angles_raw['trial']==self.trial,'yaw'])[self.slider_value:mscope_frame2]),color='black')
        self.axes_lst2[0][2].set_title('Yaw', fontsize=p_fsize)
        self.axes_lst2[0][2].spines['right'].set_visible(False)
        self.axes_lst2[0][2].spines['top'].set_visible(False)
        self.axes_lst2[0][3].clear()
        self.axes_lst2[0][3].plot(self.frame_time[self.trial-1][self.slider_value:mscope_frame2],np.rad2deg(np.array(self.head_angles_raw.loc[self.head_angles_raw['trial']==self.trial,'roll'])[self.slider_value:mscope_frame2]),color='black')
        self.axes_lst2[0][3].set_title('Roll', fontsize=p_fsize)
        self.axes_lst2[0][3].spines['right'].set_visible(False)
        self.axes_lst2[0][3].spines['top'].set_visible(False)
        self.axes_lst2[0][4].clear()
        self.axes_lst2[0][4].plot(self.frame_time[self.trial-1][self.slider_value:mscope_frame2],np.rad2deg(np.array(self.head_angles_raw.loc[self.head_angles_raw['trial']==self.trial,'pitch'])[self.slider_value:mscope_frame2]),color='black')
        self.axes_lst2[0][4].set_title('Pitch', fontsize=p_fsize)
        self.axes_lst2[0][4].spines['right'].set_visible(False)
        self.axes_lst2[0][4].spines['top'].set_visible(False)
        self.fig2.canvas.draw_idle()

    def plot_animations_behavior(self):
        fps_set = 200
        p_fsize = 20
        import locomotion_class
        loco = locomotion_class.loco_class(path_loco)
        [trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(self.animal_name,self.session,self.frames_dFF)  
        mscope_time2 = self.frame_time[self.trial-1][self.slider_value]+1 #1s later
        mscope_frame2 = np.abs(self.frame_time[self.trial-1] - mscope_time2).argmin()
        bcam_frames_plot = np.linspace(0,np.shape(self.frames_bcam)[2],num=(mscope_frame2-self.slider_value)+1,dtype='int')
        self.axes_lst2[0][1].clear()
        self.axes_lst2[0][2].clear()
        self.axes_lst2[0][2].plot(self.frame_time[self.trial-1][self.slider_value:mscope_frame2],np.rad2deg(np.array(self.head_angles_raw.loc[self.head_angles_raw['trial']==self.trial,'yaw'])[self.slider_value:mscope_frame2]),color='black')
        self.axes_lst2[0][2].set_title('Yaw', fontsize=p_fsize)
        self.axes_lst2[0][2].spines['right'].set_visible(False)
        self.axes_lst2[0][2].spines['top'].set_visible(False)
        self.axes_lst2[0][3].clear()
        self.axes_lst2[0][3].plot(self.frame_time[self.trial-1][self.slider_value:mscope_frame2],np.rad2deg(np.array(self.head_angles_raw.loc[self.head_angles_raw['trial']==self.trial,'roll'])[self.slider_value:mscope_frame2]),color='black')
        self.axes_lst2[0][3].set_title('Roll', fontsize=p_fsize)
        self.axes_lst2[0][3].spines['right'].set_visible(False)
        self.axes_lst2[0][3].spines['top'].set_visible(False)
        self.axes_lst2[0][4].clear()
        self.axes_lst2[0][4].plot(self.frame_time[self.trial-1][self.slider_value:mscope_frame2],np.rad2deg(np.array(self.head_angles_raw.loc[self.head_angles_raw['trial']==self.trial,'pitch'])[self.slider_value:mscope_frame2]),color='black')
        self.axes_lst2[0][4].set_title('Pitch', fontsize=p_fsize)
        self.axes_lst2[0][4].spines['right'].set_visible(False)
        self.axes_lst2[0][4].spines['top'].set_visible(False)
        animation_objects = []
        count_i = 0
        for i in np.arange(self.slider_value, mscope_frame2):
            video_ani = self.axes_lst2[0][1].imshow(self.frames_bcam[:,:,bcam_frames_plot[count_i]], cmap=cm.Greys_r)        
            dot_ani_yaw = self.axes_lst2[0][2].scatter(self.frame_time[self.trial-1][i],np.rad2deg(np.array(self.head_angles_raw.loc[self.head_angles_raw['trial']==self.trial,'yaw']))[i],s = 45, color='orange')
            dot_ani_roll = self.axes_lst2[0][3].scatter(self.frame_time[self.trial-1][i],np.rad2deg(np.array(self.head_angles_raw.loc[self.head_angles_raw['trial']==self.trial,'roll']))[i],s = 45, color='orange')
            dot_ani_pitch = self.axes_lst2[0][4].scatter(self.frame_time[self.trial-1][i],np.rad2deg(np.array(self.head_angles_raw.loc[self.head_angles_raw['trial']==self.trial,'pitch']))[i],s = 45, color='orange')
            animation_objects.append([video_ani,dot_ani_yaw,dot_ani_roll,dot_ani_pitch])
            count_i += 1
        self.ani_frameb = animation.ArtistAnimation(self.fig2, animation_objects, interval=fps_set, blit=True)
        return self.ani_frameb

    def stop_animation_behavior(self):
        self.ani_frameb.event_source.stop()
        
    def restart_animation_behavior(self):
        self.ani_frameb.event_source.start()
    
    def Main(self):                
        while True:
            events, values = self.window.read()

            #ACTING ON GUI BUTTONS
            if events == 'Plot' and int(values['Slider']) == 0: #GUI starts with slider at 1 and you press Go
                self.slider_value = int(values['Slider'])    
                self.rois = int(values['ROI'])
                self.trial = int(values['Trial'])    
                self.draw_figure()
               #Inputs
                self.S2P = values['Suite2pAnalysis']
                self.Fiji = values['FijiAnalysis']
                self.EXTRACT = values['EXTRACTAnalysis']
                self.PROCESSED = values['ProcessedFilesAnalysis']
                version_mscope = 'v4'
                count_overlap = 0
                trials = mscope.get_trial_id()
                self.frames_dFF = mscope.get_black_frames() #black frames removed before ROI segmentation
                self.frame_time = mscope.get_miniscope_frame_time(trials,self.frames_dFF,version_mscope) #get frame time for each trial
                self.ref_image = mscope.get_ref_image()
                self.reg_path = mscope.path +'\\Registered video\\'
                if self.S2P:
                    [dFF, dFF_trial, trials, keep_rois, reg_th, reg_bad_frames] = mscope.load_processed_dFF()
                    self.centroid_cell = mscope.get_centroid(keep_rois)
                    self.coord_cell = mscope.get_pixel_coordinates(keep_rois, count_overlap)
                    self.dFF_trial = dFF_trial
                if self.Fiji:
                    path_fiji = self.reg_path + 'RoiSet.zip'
                    filename_traces = 'RoiSet_Results_trial'+str(self.trial)+'_raw.csv'
                    rois = read_roi.read_roi_zip(path_fiji)
                    rois_names = list(rois.keys())
                    x1_list = []
                    x2_list = []
                    y1_list = []
                    y2_list = []
                    for key, value in rois.items():
                         x1_list.append(value['x1'])
                         x2_list.append(value['x2'])
                         y1_list.append(value['y1'])
                         y2_list.append(value['y2'])
                    rois_dict = {'x1': x1_list, 'x2': x2_list, 'y1': y1_list, 'y2': y2_list}
                    rois_df = pd.DataFrame(rois_dict, columns = ['x1','x2','y1','y2'], index=rois_names)
                    self.coord_cell = []
                    self.centroid_cell = []
                    for r in range(np.shape(rois_df)[0]):
                        self.coord_cell.append(np.transpose(np.vstack((np.linspace(rois_df.iloc[r,0]/mscope.pixel_to_um,rois_df.iloc[r,1]/mscope.pixel_to_um,num=10),np.linspace(rois_df.iloc[r,2]/mscope.pixel_to_um,rois_df.iloc[r,3]/mscope.pixel_to_um,num=10)))))
                        self.centroid_cell.append([rois_df.iloc[r,2]/mscope.pixel_to_um,rois_df.iloc[r,1]/mscope.pixel_to_um])
                    df_trace_bgsub = pd.read_csv(self.reg_path+filename_traces)
                    self.dFF_trial = np.transpose(np.array(df_trace_bgsub.iloc[:,1::3]))
                if self.EXTRACT:
                    thrs_spatial_weights = 0
                    trials = mscope.get_trial_id()
                    [self.coord_cell, self.dFF_trial] = mscope.read_extract_output(thrs_spatial_weights, self.frame_time, trials)
                    self.centroid_cell = mscope.get_roi_centroids(self.coord_cell)
                if self.PROCESSED:
                    [self.dFF_trial, df_events_extract, trials, self.coord_cell, reg_th, amp_arr,
                     reg_bad_frames] = mscope.load_processed_files()
                    self.centroid_cell = mscope.get_roi_centroids(self.coord_cell)
                if values['Behavior']:
                    self.get_video_behavior()
                    self.plot_behavior()
                    self.plot_acc_values()

            if events == 'Slider+UNDER CURSOR':
                self.focus = 'Slider'
                
            if events == 'MouseWheel:Up' and self.focus == 'Slider':
                self.window['Slider'].update(values['Slider'] + 1)
                self.slider_value = int(values['Slider'])
                self.rois = int(values['ROI'])
                self.trial = int(values['Trial'])
                self.get_registered_frames()
                self.get_video_mask()
                self.plot_rois_reg_image()
                self.plot_video()
                self.plot_video_mask()
                self.get_calcium_trace_events()
                self.plot_trace_events() 
                if values['Behavior']:
                    self.get_video_behavior()
                    self.plot_behavior()
                    self.plot_acc_values()
    
            if events == 'MouseWheel:Down' and self.focus == 'Slider':
                self.window['Slider'].update(values['Slider'] - 1)
                self.slider_value = int(values['Slider'])
                self.rois = int(values['ROI'])
                self.trial = int(values['Trial'])
                self.get_registered_frames()
                self.get_video_mask()
                self.plot_rois_reg_image()
                self.plot_video()
                self.plot_video_mask()
                self.get_calcium_trace_events()
                self.plot_trace_events() 
                if values['Behavior']:
                    self.get_video_behavior()
                    self.plot_behavior()
                    self.plot_acc_values()
            
            if events is None or events == 'Exit':  
                self.window.close()
                break
                            
            if events == 'Plot' or events == 'Slider':
                self.slider_value = int(values['Slider'])
                self.rois = int(values['ROI'])
                self.trial = int(values['Trial'])
                self.get_registered_frames()
                self.get_video_mask()
                self.plot_rois_reg_image()
                self.plot_video()
                self.plot_video_mask()
                self.get_calcium_trace_events()
                self.plot_trace_events() 
                if values['Behavior']:
                    self.get_video_behavior()
                    self.plot_behavior()
                    self.plot_acc_values()
                
            if events == 'VideoImg':
                self.rois = int(values['ROI'])
                self.trial = int(values['Trial'])
                self.get_registered_frames()
                self.get_video_mask()
                self.plot_rois_reg_image()
                self.get_calcium_trace_events()
                self.ani_frame = self.plot_animations()
                
            if events == 'StopI':
                self.stop_animation_img()

            if events == 'RestartI':
                self.restart_animation_img()
            
            if events == 'VideoBeh' and values['Behavior']:
                self.window['Slider'].update(values['Slider'] - 1)
                self.slider_value = int(values['Slider'])
                self.trial = int(values['Trial'])
                self.get_video_behavior()
                self.plot_behavior()
                self.plot_acc_values()
                self.ani_frameb = self.plot_animations_behavior()

            if events == 'StopB':
                self.stop_animation_behavior()

            if events == 'RestartB':
                self.restart_animation_behavior()

                
        cv2.destroyAllWindows()
   
if __name__=='__main__':
    Miniscope_dataviz().Main()
        




            
            
