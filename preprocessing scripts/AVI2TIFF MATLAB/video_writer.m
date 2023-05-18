function [ writer ] = video_writer( video, frame_rate, encoder )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
% 
% % Delete video if it exists
% if exist(video, 'file') == 2
%         delete(video);
% end

if(isnumeric(video)) % write to matrix
    writer.type = 'mat';
else
    [~, ~, extension] = fileparts(lower(video));
    writer.type = extension(2:end);
end

% if(isempty(frame_rate))
    writer.frame_rate = 30;
%     disp('[video_writer] Frame rate not provided. Setting to 30 fps.');
% else
%     writer.frame_rate = frame_rate;
% end

% if isempty(encoder)
%     if (strcmpi(writer.type, 'tif') || strcmpi(writer.type, 'tiff') )
%         writer.encoder = 'TIFF';
%     else
%         writer.encoder = 'Grayscale AVI';
%     end
% else
%     writer.encoder = encoder;
% end

if ~exist('encoder','var')
    if (strcmpi(writer.type, 'tif') || strcmpi(writer.type, 'tiff') )
        writer.encoder = 'TIFF';
    else
        writer.encoder = 'Grayscale AVI';
    end
else
    writer.encoder = encoder;
end

if strcmp(writer.type, 'mat')
    writer.handle = writer_handle;
elseif (strcmpi(writer.type, 'tif') || strcmpi(writer.type, 'tiff'))
    writer.handle = video;
else 
    writer.handle = VideoWriter(video, writer.encoder );
    writer.frame_rate = frame_rate;
    open(writer.handle);
end

end

