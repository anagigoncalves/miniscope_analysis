%% tiff to h5 - trial videos
path_ses = 'F:\TM RAW FILES\split ipsi fast 480\Test\2022_06_01';
clearvars -except path_ses; clc; close all;
cd('C:\Users\Ana\Documents\PhD\Dev\EXTRACT-public-master\')
path_data = strcat(path_ses,'\Registered video\');
mkdir(strcat(path_data,'EXTRACT\'))
filelist = dir(strcat(path_data,'*.tif'));
for f=1:length(filelist)
    filename = filelist(f).name;
    disp(strcat('Creating h5 for', {' '}, filename))
    tiff_info = imfinfo([path_data, filename]); % return tiff structure, one element per image
    datasetname = '/data';
    h5create([path_data, '\EXTRACT\', filename(1:end-4), '.h5'],datasetname,[608 608 size(tiff_info, 1)],'Datatype','single','ChunkSize',[608,608,1]);
    %concatenate each successive tiff to tiff_stack
    for ii = 1 : size(tiff_info, 1)
        temp_tiff = single(imread([path_data, filename], ii));
        h5write([path_data, '\EXTRACT\', filename(1:end-4), '.h5'],datasetname,temp_tiff,[1,1,ii],[608,608,1]);
    end
end

%% tiff to h5 - downsampled video
clearvars -except path_ses; clc; close all;
cd('C:\Users\Ana\Documents\PhD\Dev\EXTRACT-public-master\')
path_data = strcat(path_ses,'\Registered downsampled session\');
mkdir(strcat(path_data,'EXTRACT\'))
filelist = dir(strcat(path_data,'*.tif'));
filename = filelist(1).name;
disp(strcat('Creating h5 for', {' '}, filename))
tiff_info = imfinfo([path_data, filename]); % return tiff structure, one element per image
datasetname = '/data';
h5create([path_data, '\EXTRACT\', filename(1:end-4), '.h5'],datasetname,[608 608 size(tiff_info, 1)],'Datatype','single','ChunkSize',[608,608,1]);
%concatenate each successive tiff to tiff_stack
for ii = 1 : size(tiff_info, 1)
    temp_tiff = single(imread([path_data, filename], ii));
    h5write([path_data, '\EXTRACT\', filename(1:end-4), '.h5'],datasetname,temp_tiff,[1,1,ii],[608,608,1]);
end

%% inputs to run
clearvars -except path_ses; clc; close all;
addpath(genpath('C:\Users\Ana\Documents\PhD\Dev\EXTRACT-public-master\'))
filename = dir(strcat(path_ses,'\Registered downsampled session\EXTRACT\','*.h5'));
M = single(hdf5read(strcat(path_ses,'\Registered downsampled session\EXTRACT\',filename(1).name),'/data'));

%% EXTRACT settings
config=[];
config = get_defaults(config); %calls the defaults

% Essentials, without these EXTRACT will give an error:
config.preprocess=1;
config.fix_zero_FOV_strips=0; %fixes zero strips from motion correction
config.medfilt_outlier_pixels=0; %median filter to fix dead pixels
config.skip_dff=0; %when 0 does dFF to prevent issues with dividing by small values
config.baseline_quantile=0.4; %quantile to compute baseline for dFF
config.skip_highpass = 0; %0 high pass video to correct wandering baseline
config.spatial_highpass_cutoff=10; %can be changed
config.temporal_denoising=0; %denoises the movie in time using wavelets
config.remove_background=1; %removes wandering baseline 
config.avg_cell_radius=1.5; %output gives avg_cell_radius around 5 for input of 1.5
config.num_partitions_x=1;
config.num_partitions_y=1; 
config.dendrite_aware=1;
config.use_gpu=1; 
config.cellfind_max_steps=1000;
config.cellfind_min_snr=0;
config.cellfind_filter_type='none'; %spatial smoothing filter for ROI
config.max_iter=4; %iterations to improve mask drawing
config.parallel_cpu=1;
config.thresholds.spatial_corrupt_thresh=2;
config.thresholds.T_min_snr = 2; %choose more or less cells
config.trace_output_option = 'nonneg';
config.adaptive_kappa = 0;

% disp(strcat('Running EXTRACT for', {' '}, filename))
output=extractor(M,config);

%% Cell check
cell_check(output, M)

%% Plot masks and traces
size_frame = 608;
ref_image = nanmean(M,3);
coord_cells = cell(1,size(output.spatial_weights,3));
for c = 1:size(output.spatial_weights,3)
    [coord_cells{c}(:,1),coord_cells{c}(:,2)] = find(flipud(output.spatial_weights(:,:,c)));
end
figure()
axis square
hold on
imagesc(flipud(ref_image))
xlim([0 size_frame])
ylim([0 size_frame])
colormap(gray)
for c = 1:length(coord_cells)
    scatter(coord_cells{c}(:,2),coord_cells{c}(:,1),10,'o','filled')
end

%% save data
trace_nonneg = output.temporal_weights;
spatial_weights = output.spatial_weights;
save(strcat(path_ses,'\Registered downsampled session\EXTRACT\','\extract_output'),'trace_nonneg','spatial_weights','config','-v7.3')

%% apply it to each trial
filelist = dir(strcat(path_ses,'\Registered video\','*.tif'));
trials_notordered = zeros(1,length(filelist));
for f=1:length(filelist)
    filename = filelist(f).name;
    trials_notordered(f) = str2double(filename(2:strfind(filelist(f).name,'_')-1));
end
trials = sort(trials_notordered);
path_data_trials = strcat(path_ses,'\Registered video\EXTRACT\');
for t=1:length(trials)
    disp(strcat('Running EXTRACT with downsampled masks for  T', num2str(trials(t))))
    M_trial = single(hdf5read(strcat(path_data_trials,'T',num2str(trials(t)),'_reg'),'/data'));
    S_in=output.spatial_weights;
    config.max_iter=0;
    config.S_init=full(reshape(S_in, size(S_in, 1) * size(S_in, 2), size(S_in, 3)));
    output_trial=extractor(M_trial,config);
    trace_nonneg = output_trial.temporal_weights;
    spatial_weights = output_trial.spatial_weights;
    save(strcat(path_data_trials,'\extract_output_T',num2str(trials(t)),'_reg'),'trace_nonneg','spatial_weights','config','-v7.3')
end
