function close_writer( writer )
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here

if ~(strcmpi(writer.type, 'tif') || strcmpi(writer.type, 'tiff'))
    close(writer.handle);
end;


end

