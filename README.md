# miniscope_analysis

This repository is used to analyze widefield imaging data from Purkinje cell dendrites using Miniscopes.

All collected data is already preprocessed and the preprocessed files are in Dropbox: LocoCf-Data/Data/Miniscope/.
Most of the analysis rely on the mean calcium event rate that you find also in this folder under Rasters & Heatmaps.

In the folder <strong>figures anag thesis</strong> you can find the code for each plot in the thesis identified by figure number.
To plot the animals on top of the whole brain, use brainrender and then run the fig2_1_histology

In the folder <strong>figures learning</strong> there is the code for the calcium modulation plots from (20250705_miniscope_imaging_calcium_modulation.pptx). 

In the folder <strong>population analysis</strong> there is the code to process spike-triggered averages (miniscope_sta.py) and raster plots (miniscope_sw_st.py).
miniscope_sw_st.py also computes the mean calcium rate per trial for each ROI and phase bin. It also does it in time, if you specifiy the right parameters.

In the folder <strong>preprocessing scripts</strong> there is the code to preprocess miniscope data. 
First the videos from Minisocpe software need to be converted to TIFF (there is the folder AVI2TIFF MATLAB) that has the MATLAB code for that.
Then, they are processed with Suite2p.
Afterwards, you can concatenate, if needed, the registered TIFFs from Suite2p using preprocess_tiff_registered.

On the main dir, you find the locomotion_class.py and miniscope_session_class.py that have all the necessary functions to run locomotion and miniscope analysis.
ROI_analysis.py is the script that processes the Suite2p data, cleaning up after motion correction, detects calcium events and then outputs the processed files dataframes.
The most important dataframe is df_events_extract_rawtrace that is going to be used in miniscope_sw_st.py to compute the mean calcium rate across phase bins and trials that is used for most relevant analysis.



