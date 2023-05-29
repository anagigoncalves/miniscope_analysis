function [ file_out ] = concatenate_videos_HGM( file_list, file_out, frames )
%CONCATENATE_VIDEOS Summary of this function goes here
%   Detailed explanation goes here

global COMMAND_DEBUG;
if (isempty('COMMAND_DEBUG'))
    COMMAND_DEBUG = 0;
end


reader = video_reader(file_list(1).name);
close_reader(reader);

writer = video_writer(file_out, reader.frame_rate);

for file = 1:length(file_list)
    
    reader = video_reader(file_list(file).name);
    
    if(isempty(frames))
        use_frames = 1:reader.n_frames;
    else
        use_frames = frames;
    end
        
    for f = use_frames
        
        if(COMMAND_DEBUG)
            disp(['files: ' num2str(file) '/' num2str(length(file_list)) '     frames: ' num2str(f) '/' num2str(length(use_frames))]);
        end
        
        write_frame(writer, read_frame(reader, f));
    
    end

    close_reader(reader)
    
end

close_writer(writer);

end

