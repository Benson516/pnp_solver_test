import numpy as np

data_dir_str = '/home/benson516/test_PnP_solver/dataset/Huey_face_landmarks_pose/'
data_file_str = 'test_Alexander.txt'
#
data_path_str = data_dir_str + data_file_str

is_limiting_line_count = False # True
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
        # print("data_name_split_list = %s" % str(data_name_split_list))
        data_name_split_list_list.append( data_name_split_list )
        # Update
        _line = _f.readline()
#
print(data_str_list_list[0][0:5]) # [data_idx][column in line of file]
print(data_name_split_list_list[0][9:12]) # [data_idx][column in file name split]


# Convert the original data to structured data_list
#-------------------------------------------------------#
data_list = list()
for _idx in range(len(data_str_list_list)):
    data_id_dict = dict()
    # Grund truth
    data_id_dict['file_name'] = data_str_list_list[_idx][0]
    data_id_dict['distance'] = float(data_str_list_list[_idx][1])
    data_id_dict['pitch'] = float(data_str_list_list[_idx][2])
    data_id_dict['roll'] = float(data_str_list_list[_idx][3])
    data_id_dict['yaw'] = float(data_str_list_list[_idx][4])
    # Test inputs
    data_id_dict['box_xy'] = np.array(data_name_split_list_list[_idx][9:11]).astype(np.float) # np array, shape=(2,)
    data_id_dict['box_h'] = float(data_name_split_list_list[_idx][11]) # np array, shape=(2,)
    data_id_dict['LM_local_norm'] = (np.array([data_str_list_list[_idx][5::2], data_str_list_list[_idx][6::2]]).T).astype(np.float) # np array, shape=(2,)
    #
    data_id_dict['LM_pixel'] = data_id_dict['LM_local_norm'] * data_id_dict['box_h'] + data_id_dict['box_xy'].reshape((1,2))
    # Sppend to the total data list
    data_list.append(data_id_dict)
#
# print(data_list[0])
#-------------------------------------------------------#

# Start testing
#-------------------------------------------------------#
