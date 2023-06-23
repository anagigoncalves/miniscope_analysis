function [ I ] = read_frame( reader, frame )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
        
    if (strcmp(reader.type, 'mat'))
        
        I = squeeze(reader.data(:,:,frame,:));

    elseif (strcmpi(reader.type, 'tif') || strcmpi(reader.type, 'tiff'))
    
        I = imread(reader.handle, frame);
    
    elseif strcmp(reader.type, 'txt')

        fseek(reader.handle, (8 + reader.width * reader.height) * (frame-1) + 8, 'bof');
        I = fread(reader.handle, [reader.width, reader.height], 'uint8=>uint8');
        I = I';
        
    else
        
        I = read(reader.handle, frame);    
    
    end

end

