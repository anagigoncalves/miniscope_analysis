clear all; clc;

session_input_path = {'F:\TM RAW FILES\split_contra_fast_480\MC13420\2022_06_03\'};
for s = 1:length(session_input_path)
    convert_miniscope_session_data_HGM(session_input_path{s})
end

