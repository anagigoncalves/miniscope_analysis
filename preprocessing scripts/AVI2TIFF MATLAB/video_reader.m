function [ reader ] = video_reader( video )
%VIDEO_READER Summary of this function goes here
%   Detailed explanation goes here

    if isnumeric(video)             % video is a matrix
        reader.type = 'mat';
        reader.handle = [];
        reader.width = size(video, 2); 
        reader.height = size(video, 1); 
        reader.n_frames = size(video, 3);
        reader.frame_rate = [];
        reader.encoder = 'matrix';
        reader.data = video;
        return;
    end

    % video is a file
    [~, ~, extension] = fileparts(video);
    reader.type = lower(extension(2:end));

    if (strcmpi(reader.type, 'tif') || strcmpi(reader.type, 'tiff'))
    
        info = imfinfo(video);
        reader.handle = video;
        reader.width = info(1).Width; 
        reader.height = info(1).Height; 
        reader.n_frames = numel(info);
        reader.frame_rate = [];
        reader.encoder = 'TIFF';
        
    elseif strcmp(reader.type, 'txt')
    
        reader.handle = fopen(video);
        file_data = dir(video);
    
        if(file_data.bytes > 0)
        
            [ height_bytes ] = fread(reader.handle, 4, 'uint8=>uint8');
            [ width_bytes ] = fread(reader.handle, 4, 'uint8=>uint8');

            reader.height = swapbytes(typecast(uint8(height_bytes),'uint32'));
            reader.width = swapbytes(typecast(uint8(width_bytes),'uint32'));
            reader.n_frames = floor(file_data.bytes / (reader.width * reader.height + 8));    

            frewind(reader.handle)
            reader.frame_rate = [];
            reader.encoder = 'Raw';
            
        else
            reader = [];
        end
    else
        
        reader.handle = VideoReader(video);
        reader.width = reader.handle.Width;
        reader.height = reader.handle.Height;
        reader.n_frames = reader.handle.Duration * reader.handle.FrameRate;
        reader.frame_rate = reader.handle.FrameRate;
        reader.encoder = [reader.handle.VideoFormat,' AVI'];
        
    end

end

