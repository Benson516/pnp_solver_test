import numpy as np

data_dir_str = '/home/benson516/test_PnP_solver/dataset/Huey_face_landmarks_pose/'
data_file_str = 'test_Alexander.txt'
#
data_path_str = data_dir_str + data_file_str

is_limiting_line_count = True
data_str_list_list = list()
data_name_split_list_list = list()
with open(data_path_str, 'r') as _f:
    # Read and print the entire file line by line
    _line = _f.readline()
    _count = 0
    while (_line != '') and ((not is_limiting_line_count) or (_count < 3) ):  # The EOF char is an empty string
        _count += 1
        # print(_line, end='')
        _line_split_list = _line.split()
        # print(_line_split_list)
        data_str_list_list.append(_line_split_list)
        #
        data_name_split_list = _line_split_list[0].split('_')
        print("data_name_split_list = %s" % str(data_name_split_list))
        data_name_split_list_list.append( data_name_split_list )
        # Update
        _line = _f.readline()
#

print(data_str_list_list[0][0:5])
print(data_name_split_list_list[0][9:12])
