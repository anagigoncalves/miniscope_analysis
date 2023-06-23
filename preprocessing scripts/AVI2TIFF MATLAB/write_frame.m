function write_frame( writer, frame, varargin )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    if strcmp(writer.type, 'mat')
        writer.handle.curr_frame = writer.handle.curr_frame + 1;
        writer.data(:,:, writer.handle.curr_frame,:) = frame;
    elseif (strcmpi(writer.type, 'tif') || strcmpi(writer.type, 'tiff'))
        imwrite(frame, writer.handle, 'writemode', 'append');
    else
        writeVideo(writer.handle, frame);
    end

end

