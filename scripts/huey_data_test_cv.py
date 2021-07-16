import numpy as np
import copy
import time
import csv
#
import cv2
#
import PNP_SOLVER_LIB as PNPS
import TEST_TOOLBOX as TTBX


#---------------------------#
# Landmark (LM) dataset
data_dir_str = '/home/benson516/test_PnP_solver/dataset/Huey_face_landmarks_pose/'
data_file_str = 'test_Alexander.txt'
# data_file_str = 'test_Alexey.txt'
# data_file_str = "test_Holly.txt"
# data_file_str = "test_Pantea.txt"
#
# data_file_str = "train_head03.txt"
#---------------------------#
# Image of Alexander
# Original image
image_dir_str = '/home/benson516/test_PnP_solver/dataset/Huey_face_landmarks_pose/images/alexander_SZ/'
# The image used for analysis
image_result_unflipped_dir_str = '/home/benson516/test_PnP_solver/dataset/Huey_face_landmarks_pose/images/alexander_SZ_result_unflipped/'
# The same as the original image
image_result_dir_str = '/home/benson516/test_PnP_solver/dataset/Huey_face_landmarks_pose/images/alexander_SZ_result/'
#---------------------------#
# Result CSV file
result_csv_dir_str = '/home/benson516/test_PnP_solver/dataset/Huey_face_landmarks_pose/result_CSVs/'
result_csv_file_prefix_str = "result_csv_"
result_statistic_txt_file_prefix_str = "statistic_"

# Behavior of this program
#---------------------------#
# is_run_through_all_data = True
is_run_through_all_data = False
# Data
is_limiting_line_count = True
# is_limiting_line_count = False
# DATA_START_ID = 0
# DATA_START_ID = 658
DATA_START_ID = 379 # (0, 0, 0), Note: #380 and #381 has dramatical shift in pose estimation by current method (m1)
# DATA_START_ID = 926 # (0, -20, 0)
# DATA_START_ID = 1070 # (0, 40, 0)
# DATA_START_ID = 922
# DATA_START_ID = 969
# DATA_START_ID = 235
# DATA_START_ID = 146 # 1203 # 1205 # 1124 # 616 # 487 # 379 # 934 # 893 # 540 # 512 # 775 # (0, 0, 0), d=220~20
#
# specific_drpy = dict()
# specific_drpy["distance"] = '140'
# specific_drpy["roll"] = "-45"
# specific_drpy["pitch"] = "-30"
# specific_drpy["yaw"] = "40"
# specific_drpy = {"distance":"40", "roll":"-25", "pitch":"0", "yaw":"0"}
# specific_drpy = {"distance":"100", "roll":"0", "pitch":"0", "yaw":"40"}
# specific_drpy = {"distance":"100", "roll":"-45", "pitch":"-15", "yaw":"0"} # The "flipping" case
# specific_drpy = {"distance":"120", "roll":"45", "pitch":"-15", "yaw":"40"}
# specific_drpy = {"distance":"20", "roll":"0", "pitch":"-30", "yaw":"-40"}
# specific_drpy = {"distance":"20", "roll":"-45", "pitch":"30", "yaw":"40"}
# specific_drpy = {"distance":"160", "roll":"25", "pitch":"-30", "yaw":"-40"} # Error type LM
# specific_drpy = {"distance":"100", "roll":"45", "pitch":"-30", "yaw":"-40"} # Error type LM
# specific_drpy = {"distance":"160", "roll":"-25", "pitch":"0", "yaw":"-40"} # Error type fitting
# specific_drpy = {"distance":"100", "roll":"0", "pitch":"0", "yaw":"20"}
specific_drpy = None
#
DATA_COUNT = 3
# DATA_COUNT = 1000
# DATA_COUNT = 10000 # 1000
#
verbose = True
# verbose = False
# Image display
is_showing_image = True
# is_showing_image = False
#
# Fail cases investigation
is_storing_fail_case_image = True
# is_storing_fail_case_image = False
# Note: pass_count >= pass_count_threshold --> passed!!
pass_count_threshold = 4 # 3 # Note: max=4. If there are less than (not included) pass_count_threshold pass items, store the image
#
# Statistic CSV file
is_statistic_csv_horizontal = True # class --> (right)
# is_statistic_csv_horizontal = False # class (|, down)
#---------------------------#

# Not to flush the screen
if is_run_through_all_data:
    DATA_START_ID = 0
    is_limiting_line_count = False
    verbose = False
    is_showing_image = False
    specific_drpy = None
elif specific_drpy is not None:
    DATA_START_ID = 0
    is_limiting_line_count = False
    # verbose = False # Inherent the above setting
    # is_showing_image = False # Inherent the above setting

# No image, son't try to open the image
# The set is too big, son't bother to do so
if data_file_str == "train_head03.txt":
    is_showing_image = False
    is_storing_fail_case_image = False
    #
    verbose = False

#

# Parameters of the data
#---------------------------#
is_mirrored_image = True
pattern_scale = 1.0 # 0.85 # Multiply onto the golden pattern
#---------------------------#


# Loading Data
#----------------------------------------------------------#
data_path_str = data_dir_str + data_file_str
#
data_idx_list = list()
data_str_list_list = list()
data_name_split_list_list = list()
with open(data_path_str, 'r') as _f:
    # Read and print the entire file line by line
    _line = _f.readline()
    _idx = 0
    while (_line != '') and ((not is_limiting_line_count) or (_idx < (DATA_START_ID+DATA_COUNT) ) ):  # The EOF char is an empty string
        if _idx >= DATA_START_ID:
            data_idx_list.append(_idx)
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
        _idx += 1
#
print("-"*70)
print("data count = %d" % len(data_idx_list))
print("-"*70)
print(data_idx_list[0])
print(data_str_list_list[0][0:5]) # [data_idx][column in line of file]
print(data_name_split_list_list[0][9:12]) # [data_idx][column in file name split]
#
# time.sleep(10.0)
#----------------------------------------------------------#


# Ground truth classification
#-------------------------------#
# is_classified_by_label = True
is_classified_by_label = False

# Formate
drpy_class_format = "drpy_expand"
# drpy_class_format = "drpy_expand_zoom_in"
# drpy_class_format = "HMI_inspection"

if drpy_class_format == "drpy_expand":
    # class label and bins
    # Depth
    class_depth_nominal_value = np.arange(20.0, 240.1, 20) # Note: the length of label should be one element longer than the bin
    class_depth_label = [str(int(_e)) for _e in class_depth_nominal_value] # Using nominal value as class label
    class_depth_bins = list( class_depth_nominal_value[:-1] + 10.0 ) # Note: Keep the last label. Calculate the upper bound, since the np.digitize() return the index of ubber bound bin
    print("class_depth_label = %s" % class_depth_label)
    print("class_depth_bins = %s" % class_depth_bins)
    # Roll
    class_roll_nominal_value = np.array([-45, -25, 0, 25, 45]) # Note: the length of label should be one element longer than the bin
    class_roll_label = [str(int(_e)) for _e in class_roll_nominal_value] # Using nominal value as class label
    class_roll_bins = [-35, -15, 15, 35] # Only the middle bound values
    print("class_roll_label = %s" % class_roll_label)
    print("class_roll_bins = %s" % class_roll_bins)
    # Pitch
    class_pitch_nominal_value = np.array([-30, -15, 0, 15, 30]) # Note: the length of label should be one element longer than the bin
    class_pitch_label = [str(int(_e)) for _e in class_pitch_nominal_value] # Using nominal value as class label
    class_pitch_bins = [-23, -8, 8, 23] # Only the middle bound values
    print("class_pitch_label = %s" % class_pitch_label)
    print("class_pitch_bins = %s" % class_pitch_bins)
    # Yaw
    class_yaw_nominal_value = np.array([-40, -20, 0, 20, 40]) # Note: the length of label should be one element longer than the bin
    class_yaw_label = [str(int(_e)) for _e in class_yaw_nominal_value] # Using nominal value as class label
    class_yaw_bins = [-30, -10, 10, 30] # Only the middle bound values
    print("class_yaw_label = %s" % class_yaw_label)
    print("class_yaw_bins = %s" % class_yaw_bins)
    #
elif drpy_class_format == "drpy_expand_zoom_in":
    # class label and bins
    # Depth
    class_depth_nominal_value = np.arange(20.0, 240.1, 20) # Note: the length of label should be one element longer than the bin
    class_depth_label = [str(int(_e)) for _e in class_depth_nominal_value] # Using nominal value as class label
    class_depth_bins = list( class_depth_nominal_value[:-1] + 10.0 ) # Note: Keep the last label. Calculate the upper bound, since the np.digitize() return the index of ubber bound bin
    print("class_depth_label = %s" % class_depth_label)
    print("class_depth_bins = %s" % class_depth_bins)
    # Roll
    class_roll_nominal_value = np.array([-20, -10, 0, 10, 20]) # Note: the length of label should be one element longer than the bin
    class_roll_label = [str(int(_e)) for _e in class_roll_nominal_value] # Using nominal value as class label
    class_roll_bins = [-20, -10, 10, 20] # Only the middle bound values
    print("class_roll_label = %s" % class_roll_label)
    print("class_roll_bins = %s" % class_roll_bins)
    # Pitch
    class_pitch_nominal_value = np.array([-30, -15, 0, 15, 30]) # Note: the length of label should be one element longer than the bin
    class_pitch_label = [str(int(_e)) for _e in class_pitch_nominal_value] # Using nominal value as class label
    class_pitch_bins = [-23, -8, 8, 23] # Only the middle bound values
    print("class_pitch_label = %s" % class_pitch_label)
    print("class_pitch_bins = %s" % class_pitch_bins)
    # Yaw
    class_yaw_nominal_value = np.array([-40, -20, 0, 20, 40]) # Note: the length of label should be one element longer than the bin
    class_yaw_label = [str(int(_e)) for _e in class_yaw_nominal_value] # Using nominal value as class label
    class_yaw_bins = [-30, -10, 10, 30] # Only the middle bound values
    print("class_yaw_label = %s" % class_yaw_label)
    print("class_yaw_bins = %s" % class_yaw_bins)
    #
elif drpy_class_format == "HMI_inspection":
    # class label and bins
    # Depth
    class_depth_label = ["0", "100", "200"] # Using lower bound as the label
    class_depth_bins = [100, 200] # Only the middle bound values
    print("class_depth_label = %s" % class_depth_label)
    print("class_depth_bins = %s" % class_depth_bins)
    # Roll
    class_roll_nominal_value = np.array([-15, 0, 15]) # Note: the length of label should be one element longer than the bin
    class_roll_label = [str(int(_e)) for _e in class_roll_nominal_value] # Using nominal value as class label
    class_roll_bins = [-15, 15] # Only the middle bound values
    print("class_roll_label = %s" % class_roll_label)
    print("class_roll_bins = %s" % class_roll_bins)
    # Pitch
    class_pitch_nominal_value = np.array([-20, -10, 0, 10, 20]) # Note: the length of label should be one element longer than the bin
    class_pitch_label = [str(int(_e)) for _e in class_pitch_nominal_value] # Using nominal value as class label
    class_pitch_bins = [-20, -10, 10, 20] # Only the middle bound values
    print("class_pitch_label = %s" % class_pitch_label)
    print("class_pitch_bins = %s" % class_pitch_bins)
    # Yaw
    class_yaw_nominal_value = np.array([-20, -10, 0, 10, 20]) # Note: the length of label should be one element longer than the bin
    class_yaw_label = [str(int(_e)) for _e in class_yaw_nominal_value] # Using nominal value as class label
    class_yaw_bins = [-20.01, -10, 10, 20.01] # Only the middle bound values
    print("class_yaw_label = %s" % class_yaw_label)
    print("class_yaw_bins = %s" % class_yaw_bins)
    #
else:
    is_classified_by_label = True
#-------------------------------#

# Convert the original data to structured data_list
#-------------------------------------------------------#
data_list = list()
for _idx in range(len(data_str_list_list)):
    data_id_dict = dict()
    # File info
    data_id_dict['idx'] = data_idx_list[_idx]
    data_id_dict['file_name'] = data_str_list_list[_idx][0]
    # "Label" of classes, type: string
    data_id_dict['class'] = None # Move to below. Put this here just for keeping the order of key.
    # Grund truth
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

    # Classify ground truth data! (drpy class)
    #----------------------------------------------#
    _class_dict = dict()
    if is_classified_by_label:
        _class_dict['distance'] = data_str_list_list[_idx][1]
        _class_dict['pitch'] = data_str_list_list[_idx][2]
        _class_dict['roll'] = data_str_list_list[_idx][3]
        _class_dict['yaw'] = data_str_list_list[_idx][4]
    else:
        _class_dict['distance'] = class_depth_label[ np.digitize( data_id_dict['distance'], class_depth_bins) ]
        _class_dict['pitch'] = class_pitch_label[ np.digitize( data_id_dict['pitch'], class_pitch_bins) ]
        _class_dict['roll'] = class_roll_label[ np.digitize( data_id_dict['roll'], class_roll_bins) ]
        _class_dict['yaw'] = class_yaw_label[ np.digitize( data_id_dict['yaw'], class_yaw_bins) ]
    data_id_dict['class'] = _class_dict
    #----------------------------------------------#

    # Get only the specified data
    #----------------------------------#
    if (specific_drpy is not None) and (specific_drpy != _class_dict):
        continue
    #----------------------------------#

    # Sppend to the total data list
    data_list.append(data_id_dict)
#
# print(data_list[0])
#-------------------------------------------------------#



# ============= Start testing ================
#-------------------------------------------------------#
# Parameters and data
# Camera intrinsic matrix (Ground truth)
# f_camera = 188.55 # 175.0
f_camera = 225.68717584155982 # / 1.15
# f_camera = 225.68717584155982  / 1.2
#
fx_camera = f_camera
# fx_camera = (-f_camera) if is_mirrored_image else f_camera # Note: mirrored image LM features
fy_camera = f_camera
xo_camera = 320/2.0
yo_camera = 240/2.0
# np_K_camera_GT = np.array([[fx_camera, 0.0, xo_camera], [0.0, fy_camera, yo_camera], [0.0, 0.0, 1.0]]) # Grund truth
np_K_camera_est = np.array([[fx_camera, 0.0, xo_camera], [0.0, fy_camera, yo_camera], [0.0, 0.0, 1.0]]) # Estimated
# print("np_K_camera_GT = \n%s" % str(np_K_camera_GT))
print("np_K_camera_est = \n%s" % str(np_K_camera_est))

# 3D landmark point - local coordinate
#----------------------------------------#
point_3d_dict_list = list()
pattern_scale_list = list()

#-------------------------------------------------#
# Alexander
# list: [x,y,z]
point_3d_dict = dict()
# Note: Each axis should exist at least 3 different values to make A_all full rank
# Note: the Landmark definition in the pitcture in reversed
point_3d_dict["eye_l_96"] = [ 0.032, 0.0, 0.0] # [ 0.035, 0.0, 0.0]
point_3d_dict["eye_r_97"] = [-0.032, 0.0, 0.0] # [ 0.035, 0.0, 0.0]
point_3d_dict["eye_c_51"] = [0.0, 0.0, -0.015]
point_3d_dict["mouse_l_76"] = [ 0.027, 0.070, 0.0] # [ 0.025, 0.085, 0.0]
point_3d_dict["mouse_r_82"] = [ -0.027, 0.070, 0.0] # [ -0.025, 0.085, 0.0]
point_3d_dict["nose_t_54"] = [ -0.005, 0.0455, -0.03] # [ 0.0, 0.0455, 0.03] # [ 0.0, 0.046, 0.03]
point_3d_dict["chin_t_16"] = [0.0, 0.12, 0.0]
# point_3d_dict["brow_cl_35"] = [ 0.035, -0.0228, 0.0]
# point_3d_dict["brow_il_37"] = [ 0.0135, -0.017, 0.0]
# point_3d_dict["brow_ir_42"] = [ -0.0135, -0.017, 0.0]
# point_3d_dict["brow_cr_44"] = [ -0.035, -0.0228, 0.0]
#
# point_3d_dict["face_c"] = TTBX.solving_center_point(
#                         point_3d_dict["eye_r_97"],
#                         point_3d_dict["eye_l_96"],
#                         point_3d_dict["mouse_l_76"],
#                         point_3d_dict["mouse_r_82"]
#                         )
# Append to the list
point_3d_dict_list.append(point_3d_dict)
pattern_scale_list.append(pattern_scale)
#-------------------------------------------------#
# # Holly
# # list: [x,y,z]
# point_3d_dict = dict()
# # Note: Each axis should exist at least 3 different values to make A_all full rank
# # Note: the Landmark definition in the pitcture in reversed
# point_3d_dict["eye_l_96"] = [ 0.028, 0.0, 0.0] # [ 0.035, 0.0, 0.0]
# point_3d_dict["eye_r_97"] = [-0.028, 0.0, 0.0] # [ 0.035, 0.0, 0.0]
# point_3d_dict["eye_c_51"] = [0.0, 0.0, 0.0]
# point_3d_dict["mouse_l_76"] = [ 0.025, 0.060, 0.0] # [ 0.025, 0.085, 0.0]
# point_3d_dict["mouse_r_82"] = [ -0.025, 0.060, 0.0] # [ -0.025, 0.085, 0.0]
# point_3d_dict["nose_t_54"] = [ 0.00, 0.039, -0.03] # [ 0.0, 0.0455, 0.03] # [ 0.0, 0.046, 0.03]
# point_3d_dict["chin_t_16"] = [0.0, 0.098, 0.0]
# # point_3d_dict["brow_cl_35"] = [ 0.035, -0.0228, 0.0]
# # point_3d_dict["brow_il_37"] = [ 0.0135, -0.017, 0.0]
# # point_3d_dict["brow_ir_42"] = [ -0.0135, -0.017, 0.0]
# # point_3d_dict["brow_cr_44"] = [ -0.035, -0.0228, 0.0]
# #
# # point_3d_dict["face_c"] = TTBX.solving_center_point(
# #                         point_3d_dict["eye_r_97"],
# #                         point_3d_dict["eye_l_96"],
# #                         point_3d_dict["mouse_l_76"],
# #                         point_3d_dict["mouse_r_82"]
# #                         )
# # Append to the list
# point_3d_dict_list.append(point_3d_dict)
# pattern_scale_list.append(pattern_scale)
#-------------------------------------------------#



# # Convert to numpy vector, shape: (3,1)
# # Applying the scale as well
# np_point_3d_dict = dict()
# print("-"*35)
# print("3D points in local coordinate:")
# print("pattern_scale = %f" % pattern_scale)
# for _k in point_3d_dict:
#     np_point_3d_dict[_k] = np.array(point_3d_dict[_k]).reshape((3,1))
#     np_point_3d_dict[_k] = np_point_3d_dict[_k] * pattern_scale # Multiply the scale
#     print("%s:\n%s" % (_k, str(np_point_3d_dict[_k])))
# print("-"*35)
# # print(np_point_3d_dict)
#----------------------------------------#

# Create the solver
#----------------------------------------#
pnp_solver = PNPS.PNP_SOLVER(np_K_camera_est, point_3d_dict_list, pattern_scale_list=[pattern_scale], verbose=verbose)




# Loop through data
#-------------------------------------------------------#

# Collect the result
#--------------------------#
result_list = list()
failed_sample_filename_list = list()
failed_sample_count = 0
failed_sample_fit_error_count = 0
#--------------------------#

s_stamp = time.time()

# Loop thrugh data
for _idx in range(len(data_list)):
    print("\n-------------- data_idx = %d (process idx = %d)--------------\n" % (data_list[_idx]['idx'], _idx))
    print('file file_name: [%s]' % data_list[_idx]['file_name'])


    LM_pixel_data_matrix = data_list[_idx]['LM_pixel'] # [LM_id] --> [x,y]
    np_point_image_dict = dict()
    # [x,y,1].T, shape: (3,1)
    np_point_image_dict["eye_l_96"] = TTBX.convert_pixel_to_homo(LM_pixel_data_matrix[96], mirrored=is_mirrored_image)
    np_point_image_dict["eye_r_97"] = TTBX.convert_pixel_to_homo(LM_pixel_data_matrix[97], mirrored=is_mirrored_image)
    np_point_image_dict["eye_c_51"] = TTBX.convert_pixel_to_homo(LM_pixel_data_matrix[51], mirrored=is_mirrored_image)
    np_point_image_dict["mouse_l_76"] = TTBX.convert_pixel_to_homo(LM_pixel_data_matrix[76], mirrored=is_mirrored_image)
    np_point_image_dict["mouse_r_82"] = TTBX.convert_pixel_to_homo(LM_pixel_data_matrix[82], mirrored=is_mirrored_image)
    np_point_image_dict["nose_t_54"] = TTBX.convert_pixel_to_homo(LM_pixel_data_matrix[54], mirrored=is_mirrored_image)
    np_point_image_dict["chin_t_16"] = TTBX.convert_pixel_to_homo(LM_pixel_data_matrix[16], mirrored=is_mirrored_image)
    # np_point_image_dict["brow_cl_35"] = TTBX.convert_pixel_to_homo(LM_pixel_data_matrix[35], mirrored=is_mirrored_image)
    # np_point_image_dict["brow_il_37"] = TTBX.convert_pixel_to_homo(LM_pixel_data_matrix[37], mirrored=is_mirrored_image)
    # np_point_image_dict["brow_ir_42"] = TTBX.convert_pixel_to_homo(LM_pixel_data_matrix[42], mirrored=is_mirrored_image)
    # np_point_image_dict["brow_cr_44"] = TTBX.convert_pixel_to_homo(LM_pixel_data_matrix[44], mirrored=is_mirrored_image)
    #
    # np_point_image_dict["face_c"] = TTBX.convert_pixel_to_homo(      TTBX.solving_center_point(
    #                                                             LM_pixel_data_matrix[97],
    #                                                             LM_pixel_data_matrix[96],
    #                                                             LM_pixel_data_matrix[76],
    #                                                             LM_pixel_data_matrix[82]),
    #                                                             mirrored=is_mirrored_image
    #                                                         )

    # # Print
    # print("-"*35)
    # print("2D points on image:")
    # for _k in np_point_image_dict:
    #     # print("%s:%sp=%s.T | p_no_q_err=%s.T | q_e=%s.T" % (_k, " "*(12-len(_k)), str(np_point_image_dict[_k].T), str(np_point_image_no_q_err_dict[_k].T), str(np_point_quantization_error_dict[_k].T) ))
    #     print("%s:\n%s.T" % (_k, str(np_point_image_dict[_k].T)))
    #     # print("%s:\n%s" % (_k, str(np_point_quantization_error_dict[_k])))
    # print("-"*35)

    # Solve
    np_R_est, np_t_est, t3_est, roll_est, yaw_est, pitch_est, res_norm = pnp_solver.solve_pnp(np_point_image_dict)

    # # OpenCV method
    # #----------------------------------------------------#
    # _key_list = list(np_point_image_dict.keys())
    # model_points = np.array([ point_3d_dict[_k] for _k in _key_list] )
    # image_points = np.array([ np_point_image_dict[_k][0:2,0] for _k in _key_list])
    # camera_matrix = np_K_camera_est
    # dist_coeffs = None #  np.zeros((4,1)) # Assuming no lens distortion
    # # Solve
    # # flags = cv2.SOLVEPNP_ITERATIVE
    # flags = cv2.SOLVEPNP_EPNP
    # inliers = None
    # # success, rotation_vector, t_est_CV = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=flags )
    # success, rotation_vector, t_est_CV, inliers = cv2.solvePnPRansac(model_points, image_points, camera_matrix, dist_coeffs, flags=flags, reprojectionError=0.8 )
    # R_est_CV, _ = cv2.Rodrigues(rotation_vector)
    # roll_est_CV, yaw_est_CV, pitch_est_CV = pnp_solver.get_Euler_from_rotation_matrix(R_est_CV, verbose=False, is_degree=True)
    # print()
    # print("success = %s" % str(success))
    # print("R_est_CV = \n%s" % str(R_est_CV))
    # print("(roll_est_CV, yaw_est_CV, pitch_est_CV) \t\t= %s" % str( (roll_est_CV, yaw_est_CV, pitch_est_CV) )  ) # Already in degree
    # print("t_est_CV = \n%s" % str(t_est_CV))
    # print("inliers = %s" % str(inliers))
    # print()
    # # Overwrite the results
    # np_R_est, np_t_est, t3_est = R_est_CV, t_est_CV, t_est_CV[2,0]
    # roll_est, yaw_est, pitch_est = roll_est_CV, yaw_est_CV, pitch_est_CV
    # res_norm = 0.0 # Not being returned
    # #----------------------------------------------------#

    # Note: Euler angles are in degree
    np_R_ca_est = pnp_solver.np_R_c_a_est
    np_t_ca_est = pnp_solver.np_t_c_a_est
    # np_R_ca_est = copy.deepcopy(pnp_solver.np_R_c_a_est)
    # np_t_ca_est = copy.deepcopy(pnp_solver.np_t_c_a_est)

    # Compare result
    #-----------------------------#

    # Grund truth (R,t)
    roll_GT = data_list[_idx]['roll']
    pitch_GT = data_list[_idx]['pitch']
    yaw_GT = data_list[_idx]['yaw']
    np_R_GT = pnp_solver.get_rotation_matrix_from_Euler( roll_GT, yaw_GT, pitch_GT, is_degree=True )
    # _det = np.linalg.det(np_R_GT)
    # print("np_R_GT = \n%s" % str(np_R_GT))
    # print("_det = %f" % _det)
    distance_GT = data_list[_idx]['distance'] *0.01 # cm --> m
    np_t_GT_est = (np_t_est/t3_est) * distance_GT


    # Reprojections
    np_point_image_dict_reproject = pnp_solver.perspective_projection_golden_landmarks(np_R_est, np_t_est, is_quantized=False, is_pretrans_points=False)
    # np_point_image_dict_reproject = pnp_solver.perspective_projection_golden_landmarks(np_R_ca_est, np_t_ca_est, is_quantized=False, is_pretrans_points=True)
    np_point_image_dict_reproject_GT_ori_golden_patern = pnp_solver.perspective_projection_golden_landmarks(np_R_GT, np_t_GT_est, is_quantized=False, is_pretrans_points=False)


    # Reprojection errors
    #-----------------------------------------------------------#
    # Calculate the pixel error of the LMs and the ground-truth projection of golden pattern
    LM_GT_ed_dict = TTBX.cal_LM_error_distances(np_point_image_dict, np_point_image_dict_reproject_GT_ori_golden_patern)
    LM_GT_error_average = LM_GT_ed_dict["average_error"]
    LM_GT_error_max = LM_GT_ed_dict["max_error"]
    LM_GT_error_max_key = LM_GT_ed_dict["max_error_key"]
    print("(LM_GT_error_average, LM_GT_error_max, LM_GT_error_max_key) = (%f, %f, %s)" % (LM_GT_error_average, LM_GT_error_max, LM_GT_error_max_key))

    # Calculate the pixel error of the LMs and the reprojections of golden pattern
    predict_LM_ed_dict = TTBX.cal_LM_error_distances(np_point_image_dict_reproject, np_point_image_dict)
    predict_LM_error_average = predict_LM_ed_dict["average_error"]
    predict_LM_error_max = predict_LM_ed_dict["max_error"]
    predict_LM_error_max_key = predict_LM_ed_dict["max_error_key"]
    print("(predict_LM_error_average, predict_LM_error_max, predict_LM_error_max_key) = (%f, %f, %s)" % (predict_LM_error_average, predict_LM_error_max, predict_LM_error_max_key))

    # Calculate the pixel error of the LMs and the reprojections of golden pattern
    predict_GT_ed_dict = TTBX.cal_LM_error_distances(np_point_image_dict_reproject, np_point_image_dict_reproject_GT_ori_golden_patern)
    predict_GT_error_average = predict_GT_ed_dict["average_error"]
    predict_GT_error_max = predict_GT_ed_dict["max_error"]
    predict_GT_error_max_key = predict_GT_ed_dict["max_error_key"]
    print("(predict_GT_error_average, predict_GT_error_max, predict_GT_error_max_key) = (%f, %f, %s)" % (predict_GT_error_average, predict_GT_error_max, predict_GT_error_max_key))
    #-----------------------------------------------------------#


    #
    print("Result from the solver:\n")
    print("2D points on image (re-projection):")
    # print("2D points on image (is_mirrored_image=%s):" % str(is_mirrored_image))
    print("-"*35)
    for _k in np_point_image_dict:
        np.set_printoptions(suppress=True, precision=2)
        print("%s:%sp_data=%s.T | p_reproject=%s.T | err=%s.T" %
            (   _k,
                " "*(12-len(_k)),
                str(np_point_image_dict[_k].T),
                str(np_point_image_dict_reproject[_k].T),
                str((np_point_image_dict_reproject[_k]-np_point_image_dict[_k]).T)
            )
        )
        np.set_printoptions(suppress=False, precision=8)
        # print("%s:\n%s.T" % (_k, str(np_point_image_dict[_k].T)))
        # print("%s:\n%s" % (_k, str(np_point_quantization_error_dict[_k])))
    #
    print("-"*35)
    #
    print("res_norm = %f" % res_norm)
    print("np_R_est = \n%s" % str(np_R_est))
    _det = np.linalg.det(np_R_est)
    print("_det = %f" % _det)
    # print("(roll_est, yaw_est, pitch_est) \t\t= %s" % str( np.rad2deg( (roll_est, yaw_est, pitch_est) ) ) )
    print("t3_est = %f" % t3_est)
    print("np_t_est = \n%s" % str(np_t_est))
    print()
    # Result
    print("-"*30 + " Result " + "-"*30)
    print("distance = %f cm" % (t3_est*100.0))
    print("(roll_est, yaw_est, pitch_est) \t\t= %s" % str( [roll_est, yaw_est, pitch_est] ) )
    print("np_R_est = \n%s" % str(np_R_est))
    print("np_t_est = \n%s" % str(np_t_est))
    # Grund truth
    print("-"*28 + " Grund Truth " + "-"*28)
    print("distance = %f cm" % data_list[_idx]['distance'])
    print("(roll_GT, yaw_GT, pitch_GT) \t\t= %s" % str(  [ data_list[_idx]['roll'], data_list[_idx]['yaw'], data_list[_idx]['pitch'] ]) )
    if verbose:
        print("np_Gamma_GT = \n%s" % str( np_R_GT / np_t_GT_est[2,0] ))
    print("np_R_GT = \n%s" % str(np_R_GT))
    print("np_t_GT_est = \n%s" % str(np_t_GT_est))
    print("-"*30 + " The End " + "-"*30)
    print()
    #----------------------------#

    # Claulate the errors
    #----------------------------#
    drpy_pass_list, pass_count = TTBX.check_if_the_sample_passed(
                            (t3_est*100.0, roll_est, pitch_est, yaw_est),
                            (data_list[_idx]['distance'], data_list[_idx]['roll'], data_list[_idx]['pitch'], data_list[_idx]['yaw']),
                            (10.0, 10.0, 10.0, 10.0) )
    print("pass_count = %d  |  drpy_pass_list = %s" % (pass_count, str(drpy_pass_list)))
    fail_count = len(drpy_pass_list) - pass_count

    fitting_error = (predict_LM_error_average * distance_GT)

    # Determin if we want to further investigate this sample
    is_storing_case_image = False
    if pass_count < pass_count_threshold: # Note: pass_count >= pass_count_threshold --> passed!!
        failed_sample_count += 1
        if fitting_error > 1.0: # 1.5
            failed_sample_fit_error_count += 1
        # if fitting_error <= 1.5:
            failed_sample_filename_list.append(data_list[_idx]['file_name'])
            is_storing_case_image = is_storing_fail_case_image

    # if not drpy_pass_list[0]:
    #     failed_sample_filename_list.append(data_list[_idx]['file_name'])
    #     is_storing_case_image = is_storing_fail_case_image
    #----------------------------#


    # Store the error for statistic
    #----------------------------#
    _result_idx_dict = dict()
    _result_idx_dict['idx'] = data_list[_idx]['idx']
    _result_idx_dict["file_name"] = data_list[_idx]['file_name']
    _result_idx_dict["drpy"] = (distance_GT, roll_GT, pitch_GT, yaw_GT)
    _result_idx_dict["class"] = data_list[_idx]['class']
    # Result summary
    #-------------------------------#
    _result_idx_dict["fail_count"] = fail_count
    _result_idx_dict["pass_count"] = pass_count
    _result_idx_dict["is_depth_passed"] = drpy_pass_list[0]
    _result_idx_dict["is_roll_passed"] = drpy_pass_list[1]
    _result_idx_dict["is_pitch_passed"] = drpy_pass_list[2]
    _result_idx_dict["is_yaw_passed"] = drpy_pass_list[3]
    #-------------------------------#
    # R, t, depth, roll, pitch, yaw, residual
    #---#
    # GT
    # Result
    # Error, err = (est - GT)
    # abs(Error), for scalar values
    #-------------------------------#
    # R
    _result_idx_dict["np_R_GT"] = np_R_GT
    _result_idx_dict["np_R_est"] = np_R_est
    _result_idx_dict["np_R_err"] = np_R_est - np_R_GT
    # t
    _result_idx_dict["np_t_GT_est"] = np_t_GT_est
    _result_idx_dict["np_t_est"] = np_t_est
    _result_idx_dict["np_t_err"] = np_t_est - np_t_GT_est
    # depth
    _result_idx_dict["distance_GT"] = distance_GT
    _result_idx_dict["t3_est"] = t3_est
    _result_idx_dict["depth_err"] = t3_est - distance_GT
    _result_idx_dict["abs_depth_err"] = abs(t3_est - distance_GT)
    # roll
    _result_idx_dict["roll_GT"] = roll_GT
    _result_idx_dict["roll_est"] = roll_est
    _result_idx_dict["roll_err"] = roll_est - roll_GT
    _result_idx_dict["abs_roll_err"] = abs(roll_est - roll_GT)
    # pitch
    _result_idx_dict["pitch_GT"] = pitch_GT
    _result_idx_dict["pitch_est"] = pitch_est
    _result_idx_dict["pitch_err"] = pitch_est - pitch_GT
    _result_idx_dict["abs_pitch_err"] = abs(pitch_est - pitch_GT)
    # yaw
    _result_idx_dict["yaw_GT"] = yaw_GT
    _result_idx_dict["yaw_est"] = yaw_est
    _result_idx_dict["yaw_err"] = yaw_est - yaw_GT
    _result_idx_dict["abs_yaw_err"] = abs(yaw_est - yaw_GT)
    # residual
    _result_idx_dict["res_norm"] = res_norm
    _result_idx_dict["res_norm_1000x"] = res_norm * 1000.0
    _result_idx_dict["res_norm_10000x_n_est"] = res_norm * 1000.0 * t3_est # Note: normalized by estimatd value
    _result_idx_dict["res_norm_10000x_n_GT"] = res_norm * 1000.0 * distance_GT # Note: normalized by estimatd value
    # LM-GT error
    _result_idx_dict["LM_GT_error_average_normalize"] = LM_GT_error_average * distance_GT
    _result_idx_dict["LM_GT_error_max_normalize"] = LM_GT_error_max * distance_GT
    _result_idx_dict["LM_GT_error_max_key"] = LM_GT_error_max_key
    # predict-LM error
    _result_idx_dict["predict_LM_error_average_normalize"] = predict_LM_error_average * distance_GT
    _result_idx_dict["predict_LM_error_max_normalize"] = predict_LM_error_max * distance_GT
    _result_idx_dict["predict_LM_error_max_key"] = predict_LM_error_max_key
    # predict-GT error
    _result_idx_dict["predict_GT_error_average_normalize"] = predict_GT_error_average * distance_GT
    _result_idx_dict["predict_GT_error_max_normalize"] = predict_GT_error_max * distance_GT
    _result_idx_dict["predict_GT_error_max_key"] = predict_GT_error_max_key
    #
    result_list.append(_result_idx_dict)
    #----------------------------#


    # Image Display
    #====================================================#
    if not (is_showing_image or is_storing_case_image):
        continue


    # Get the file name of the image
    #--------------------------------------------#
    _file_name = data_list[_idx]['file_name']
    image_file_name_str = '_'.join(_file_name.split('_')[0:9]) + '.png'
    print('image file name: [%s]' % image_file_name_str)
    #--------------------------------------------#

    TTBX.plot_LMs_and_axies(
                image_file_name_str,
                image_dir_str,
                image_result_unflipped_dir_str, image_result_dir_str,
                pnp_solver,
                _result_idx_dict,
                np_point_image_dict,
                np_point_image_dict_reproject_GT_ori_golden_patern,
                np_point_image_dict_reproject,
                is_mirrored_image=is_mirrored_image,
                LM_img_width=320,
                is_showing_image=is_showing_image)

    # # Get the file name of the image
    # #--------------------------------------------#
    # _file_name = data_list[_idx]['file_name']
    # _image_file_name_str = '_'.join(_file_name.split('_')[0:9]) + '.png'
    # print('image file name: [%s]' % _image_file_name_str)
    # _image_ori_path_str = image_dir_str + _image_file_name_str
    # _image_result_unflipped_path_str = image_result_unflipped_dir_str + _image_file_name_str
    # _image_result_path_str = image_result_dir_str + _image_file_name_str
    # #--------------------------------------------#
    #
    # # Load the original image
    # #--------------------------------------------#
    # _img = cv2.imread(_image_ori_path_str)
    # if _img is None:
    #     print("!! Error occured while loading the image !!\n")
    #     # time.sleep(3.0)
    #     # continue
    #     # Fake one
    #     #---------------------#
    #     _scale = 3
    #     _width = 320 * _scale
    #     _height = 240 * _scale
    #     _intensity = 200
    #     _img = np.ones( (_height, _width, 3), dtype=np.uint8) * _intensity
    #     #---------------------#
    # # Resize the image
    # #----------------------#
    # _desired_img_display_width = 960 # 2048
    # if (_img.shape[1] != _desired_img_display_width):
    #     _aspext_ratio_img_in = float(_img.shape[1]) / float(_img.shape[0]) # width / height
    #     _desired_dim = ( int(_desired_img_display_width), int(_desired_img_display_width / _aspext_ratio_img_in) )
    #     _img = cv2.resize(_img, _desired_dim, interpolation=cv2.INTER_AREA) # Resize 1.5 times
    # #----------------------#
    # _img_shape = _img.shape
    # print("_img.shape = %s" % str(_img_shape))
    # LM_2_image_scale = _img_shape[1] / 320.0 # Note: 320x240 is the image scale used by LM
    # #--------------------------------------------#
    #
    #
    # # Flip the image if needed
    # #----------------------------------#
    # if is_mirrored_image:
    #     _img_preprocessed = cv2.flip(_img, 1)
    # else:
    #     _img_preprocessed = _img
    # #----------------------------------#
    #
    # # Ploting LMs onto the image
    # _img_LM = copy.deepcopy(_img_preprocessed)
    # # Colors
    # _color_RED   = (0, 0, 255)
    # _color_GREEN = (0, 255, 0)
    # _color_BLUE  = (255, 0, 0)
    # # _color_RED   = np.array((0, 0, 255))
    # # _color_GREEN = np.array((0, 255, 0))
    # # _color_BLUE  = np.array((255, 0, 0))
    #
    #
    # # Ploting axies
    # #-----------------------------------#
    # # Grund truth axes
    # vector_scale = 0.2
    # uv_o, dir_x, dir_y, dir_z = pnp_solver.perspective_projection_obj_axis(np_R_GT, np_t_GT_est, scale=vector_scale) # Note: use the estimated t since we don't have the grund truth.
    # print("(uv_o, dir_x, dir_y, dir_z) = %s" % str((uv_o, dir_x, dir_y, dir_z)))
    # _pixel_uv_o = (LM_2_image_scale*uv_o).astype('int')
    # _pixel_uv_x1 = (LM_2_image_scale*(uv_o+dir_x)).astype('int')
    # _pixel_uv_y1 = (LM_2_image_scale*(uv_o+dir_y)).astype('int')
    # _pixel_uv_z1 = (LM_2_image_scale*(uv_o+dir_z)).astype('int')
    # # Draw lines
    # _line_width = 2
    # cv2.line(_img_LM, _pixel_uv_o, _pixel_uv_x1, (0,0,127), _line_width)
    # cv2.line(_img_LM, _pixel_uv_o, _pixel_uv_y1, (0,127,0), _line_width)
    # cv2.line(_img_LM, _pixel_uv_o, _pixel_uv_z1, (127,0,0), _line_width)
    #
    # # Estimated axes
    # vector_scale = 0.2
    # uv_o, dir_x, dir_y, dir_z = pnp_solver.perspective_projection_obj_axis(np_R_est, np_t_est, scale=vector_scale)
    # print("(uv_o, dir_x, dir_y, dir_z) = %s" % str((uv_o, dir_x, dir_y, dir_z)))
    # _pixel_uv_o = (LM_2_image_scale*uv_o).astype('int')
    # _pixel_uv_x1 = (LM_2_image_scale*(uv_o+dir_x)).astype('int')
    # _pixel_uv_y1 = (LM_2_image_scale*(uv_o+dir_y)).astype('int')
    # _pixel_uv_z1 = (LM_2_image_scale*(uv_o+dir_z)).astype('int')
    # # Draw lines
    # _line_width = 1
    # cv2.line(_img_LM, _pixel_uv_o, _pixel_uv_x1, (0, 0, 180), _line_width)
    # cv2.line(_img_LM, _pixel_uv_o, _pixel_uv_y1, (0, 180, 0), _line_width)
    # cv2.line(_img_LM, _pixel_uv_o, _pixel_uv_z1, (180, 0, 0), _line_width)
    # #-----------------------------------#
    #
    # # Landmarks
    # #----------------------------------#
    # # [[u,v,1]].T
    # for _k in np_point_image_dict:
    #     # Landmarks
    #     _center_pixel = (np_point_image_dict[_k][0:2,0] * LM_2_image_scale).astype('int')
    #     _radius = 3
    #     _color = _color_BLUE # BGR
    #     # _color = _color_RED # BGR
    #     cv2.circle(_img_LM, _center_pixel, _radius, _color, -1)
    #     # Reprojections of golden pattern onto image using grund truth pose
    #     _center_pixel = (np_point_image_dict_reproject_GT_ori_golden_patern[_k][0:2,0] * LM_2_image_scale).astype('int')
    #     _radius = 2
    #     _color = (127, 127, 0) # BGR
    #     # _color = _color_RED # BGR
    #     cv2.circle(_img_LM, _center_pixel, _radius, _color, -1)
    #     # Reprojections of the golden pattern onto the image using estimated pose
    #     _center_pixel = (np_point_image_dict_reproject[_k][0:2,0] * LM_2_image_scale).astype('int')
    #     _radius = 1
    #     _color = _color_RED # BGR
    #     # _color = _color_BLUE # BGR
    #     cv2.circle(_img_LM, _center_pixel, _radius, _color, -1)
    # #----------------------------------#
    #
    # # Text
    # #----------------------------------#
    # font = cv2.FONT_HERSHEY_COMPLEX_SMALL # = 5
    # fontScale = 1.5
    # thickness = 2
    # # _text = "Hello world! 012345"
    # _est_text = "drpy_est = (%.2f, %.2f, %.2f, %.2f)" % (_result_idx_dict["t3_est"]*100.0,
    #                                         _result_idx_dict["roll_est"],
    #                                         _result_idx_dict["pitch_est"],
    #                                         _result_idx_dict["yaw_est"])
    # _GT_text = "drpy_GT = (%.2f, %.2f, %.2f, %.2f)" % (_result_idx_dict["distance_GT"]*100.0,
    #                                         _result_idx_dict["roll_GT"],
    #                                         _result_idx_dict["pitch_GT"],
    #                                         _result_idx_dict["yaw_GT"])
    # _err_text = "drpy_est = (%.2f, %.2f, %.2f, %.2f)" % (_result_idx_dict["depth_err"]*100.0,
    #                                         _result_idx_dict["roll_err"],
    #                                         _result_idx_dict["pitch_err"],
    #                                         _result_idx_dict["yaw_err"])
    # cv2.putText(_img_LM, _est_text, (0, 50), font, fontScale, (200, 128, 0), thickness, cv2.LINE_AA)
    # cv2.putText(_img_LM, _GT_text, (0, 80), font, fontScale, (0, 100, 180), thickness, cv2.LINE_AA)
    # cv2.putText(_img_LM, _err_text, (0, 110), font, fontScale, (0, 0, 150), thickness, cv2.LINE_AA)
    # #----------------------------------#
    #
    # # Dtermine the final image
    # #-------------------------#
    # # _img_result = _img
    # # _img_result = _img_preprocessed
    # _img_result = _img_LM
    # #-------------------------#
    #
    # # Flip the result image if needed
    # #----------------------------------#
    # if is_mirrored_image:
    #     _img_result_flipped = cv2.flip(_img_result, 1)
    # else:
    #     _img_result_flipped = _img_result
    # #----------------------------------#
    #
    # # Save the resulted image
    # #----------------------------------#
    # cv2.imwrite(_image_result_unflipped_path_str, _img_result )
    # cv2.imwrite(_image_result_path_str, _img_result_flipped )
    # #----------------------------------#
    #
    # # Displaying the image
    # #----------------------------------#
    # if is_showing_image:
    #     cv2.imshow(_image_file_name_str, _img_result)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # #----------------------------------#

    #--------------------------------------------#


#-------------------------------------------------------#

delta_time = time.time() - s_stamp
print()
print("Time elapsed for %d data = %f" % (len(result_list), delta_time))
print("Average processing time for single data = %f" % (delta_time / len(result_list)) )
print()

print("len(failed_sample_filename_list) = %d" % len(failed_sample_filename_list))
print("failed_sample_count = %d" % failed_sample_count)
print("failed_sample_fit_error_count = %d" % failed_sample_fit_error_count)
print()

failed_sample_filename_list_file_path = image_result_unflipped_dir_str + "fail_case_list.txt"
with open(failed_sample_filename_list_file_path, "w") as _f:
    _f.writelines('\n'.join(failed_sample_filename_list) )



# Data analysis and saving
#-----------------------------------#
TTBX.data_analysis_and_saving(result_list, result_csv_dir_str, result_csv_file_prefix_str, result_statistic_txt_file_prefix_str, data_file_str, is_statistic_csv_horizontal=is_statistic_csv_horizontal)
#-----------------------------------#
