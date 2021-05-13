import numpy as np
#
import PNP_SOLVER_LIB as PNPS

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



# ============= Start testing ================
#-------------------------------------------------------#
# Parameters and data
# Camera intrinsic matrix (Ground truth)
fx_camera = 111.0 # lower resolution in face
# fx_camera = 201.0 # higher resolution in face
# fx_camera = 251.0 # even higher resolution in face
fy_camera = fx_camera # 111.0
xo_camera = 100.0
yo_camera = 100.0
# np_K_camera_GT = np.array([[fx_camera, 0.0, xo_camera], [0.0, fy_camera, yo_camera], [0.0, 0.0, 1.0]]) # Grund truth
np_K_camera_est = np.array([[fx_camera, 0.0, xo_camera], [0.0, fy_camera, yo_camera], [0.0, 0.0, 1.0]]) # Estimated
# print("np_K_camera_GT = \n%s" % str(np_K_camera_GT))
print("np_K_camera_est = \n%s" % str(np_K_camera_est))

# 3D landmark point - local coordinate
#----------------------------------------#
# list: [x,y,z]
point_3d_dict = dict()
# Note: Each axis should exist at least 3 different values to make A_all full rank
point_3d_dict["eye_l_97"] = [ 0.035, 0.0, 0.0]
point_3d_dict["eye_r_96"] = [-0.035, 0.0, 0.0]
point_3d_dict["mouse_l_82"] = [ 0.025, 0.085, 0.0]
point_3d_dict["mouse_r_76"] = [ -0.025, 0.085, 0.0]
point_3d_dict["nose_t_54"] = [ 0.0, 0.046, 0.03]
# point_3d_dict["face_c"] = [ 0.0, 0.035, 0.0]
# point_3d_dict["chin"] = [ 0.0, 0.08, -0.005]
# point_3d_dict["far"] = [ 0.0, 0.0, -0.5]
# Convert to numpy vector, shape: (3,1)
np_point_3d_dict = dict()
print("-"*35)
print("3D points in local coordinate:")
for _k in point_3d_dict:
    np_point_3d_dict[_k] = np.array(point_3d_dict[_k]).reshape((3,1))
    print("%s:\n%s" % (_k, str(np_point_3d_dict[_k])))
print("-"*35)
# print(np_point_3d_dict)
#----------------------------------------#

# Create the solver
#----------------------------------------#
pnp_solver = PNPS.PNP_SOLVER_A2_M3(np_K_camera_est, point_3d_dict)


def convert_pixel_to_homo(pixel_xy):
    '''
    pixel_xy: np array, shape=(2,)
    '''
    return np.array([pixel_xy[0], pixel_xy[1], 1.0]).reshape((3,1))

# Loop through data
for _idx in range(len(data_list)):
    print("\n--------- (idx = %d)---------\n" % _idx)

    LM_pixel_data_matrix = data_list[_idx]['LM_pixel'] # [LM_id] --> [x,y]
    np_point_image_dict = dict()
    # [x,y,1].T, shape: (3,1)
    np_point_image_dict["eye_l_97"] = convert_pixel_to_homo(LM_pixel_data_matrix[97])
    np_point_image_dict["eye_r_96"] = convert_pixel_to_homo(LM_pixel_data_matrix[96])
    np_point_image_dict["mouse_l_82"] = convert_pixel_to_homo(LM_pixel_data_matrix[82])
    np_point_image_dict["mouse_r_76"] = convert_pixel_to_homo(LM_pixel_data_matrix[76])
    np_point_image_dict["nose_t_54"] = convert_pixel_to_homo(LM_pixel_data_matrix[54])
    np_R_est, np_t_est, t3_est, roll_est, yaw_est, pitch_est = pnp_solver.solve_pnp(np_point_image_dict)
