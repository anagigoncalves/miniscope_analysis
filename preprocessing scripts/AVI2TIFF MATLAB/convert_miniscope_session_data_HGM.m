function convert_miniscope_session_data_HGM(session_input_path)
% miniscope_dir = [session_input_path, filesep, 'Miniscopes'];
miniscope_dir = session_input_path;
files = dir([miniscope_dir, filesep, '*']);
dirs = files([files.isdir]);
dirs = files(3:end);   % eliminates the '.' and '..' folders
for d = 1:length(dirs)
    disp(['[convert_miniscope_session_data]  Trial ', num2str(d), '\', num2str(length(dirs)), ': ', dirs(d).name]);
    files = rdir([miniscope_dir, filesep, dirs(d).name, filesep, 'Miniscope', filesep, '*.avi']);    
    % sort file names alphabetically to ensure that the videos are concatenated in the right order
    filenames = natsortfiles({files.name});
    for f = 1:length(filenames)
        files_sorted(f).name = filenames{f};
    end
%         concatenate_videos_HGM(files_sorted, [session_input_path, filesep, dirs(d).name, '.tif'], [], 'TIFF');
        concatenate_videos_HGM(files_sorted, [session_input_path, filesep, dirs(d).name, '.tif'],[]);
end
end