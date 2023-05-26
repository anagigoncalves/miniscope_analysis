function close_reader( reader )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

    if strcmp(reader.type, 'txt')

        fclose(reader.handle);
%         
%     elseif ~strcmp(reader.type, 'tif')
%         
%         close(reader.handler);    
    
    end


end

