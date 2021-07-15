import numpy as np
import copy
import time
import csv
#
import glob # For searching in the directory
import os
import math
import json
#
import cv2
#
import PNP_SOLVER_LIB as PNPS
import TEST_TOOLBOX as TTBX


#---------------------------#
# Landmark (LM) dataset
data_rool_str = '/home/benson516/test_PnP_solver/dataset/Huey_face_landmarks_pose/M1_test/'
# data_version = 'M1_test_EIF2_alexander_20210707'
# data_version = 'M1_test_QEIF_alexander_20210709'
data_version = 'M1_test_QEIF_alexander_20210714'
#
data_dir_str = data_rool_str + data_version + '/results/'
data_file_str = data_version + ".txt" # Fake the file name for storing the post analysis results
#---------------------------#
# Image of Alexander
# Original image
image_dir_str = data_dir_str
# The image used for analysis
image_result_unflipped_dir_str = '/home/benson516/test_PnP_solver/dataset/Huey_face_landmarks_pose/M1_test/' + data_version + '/result_analysis/result_images/image_unflipped/'
# The same as the original image
image_result_dir_str = '/home/benson516/test_PnP_solver/dataset/Huey_face_landmarks_pose/M1_test/' + data_version + '/result_analysis/result_images/image/'
#---------------------------#
# Result CSV file
result_csv_dir_str = '/home/benson516/test_PnP_solver/dataset/Huey_face_landmarks_pose/M1_test/' + data_version + '/result_analysis/result_CSVs/'
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
data_idx_list = list()
data_file_name_list = list()
#
bin_file_gen = glob.iglob( data_dir_str + "*.bin" )
_idx = 0
for bin_file_path in bin_file_gen:
    if is_limiting_line_count and (_idx >= (DATA_START_ID+DATA_COUNT)):
        break
    #
    if _idx >= DATA_START_ID:
        data_idx_list.append(_idx)
        bin_file_name = os.path.basename(bin_file_path)
        # print("bin_file_name = %s" % bin_file_name)
        data_file_name_list.append(bin_file_name)
    _idx += 1
#
print("-"*70)
print("data count = %d" % len(data_idx_list))
print("-"*70)
print(data_idx_list[0])
print(data_file_name_list[0]) # [data_idx][column in line of file]
#
# time.sleep(10.0)
#----------------------------------------------------------#



# Ground truth classification
#-------------------------------#
# is_classified_by_label = True
is_classified_by_label = False

# Formate
# drpy_class_format = "drpy_expand"
# drpy_class_format = "drpy_expand_zoom_in"
drpy_class_format = "HMI_inspection"

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
for _idx, data_file_name in enumerate(data_file_name_list):
    # Process the file name
    data_name_split_list = data_file_name.split('_')
    # print("data_name_split_list = %s" % str(data_name_split_list))

    # Raw value in file name
    _raw_roll_value = float(data_name_split_list[2])
    _raw_pitch_value = float(data_name_split_list[6])
    _raw_yaw_value = float(data_name_split_list[8])
    # Sign
    _sign_roll = -1.0 # Note: The sign of the value if fully determined by the value itself.
    _sign_pitch = (-1.0 if data_name_split_list[5] == 'd' else 1.0)
    _sign_yaw = (-1.0 if data_name_split_list[1] == 'left' else 1.0)
    #

    # Read the result
    #--------------------------#
    _result_list = list() # [dist, yaw, roll, pitch]
    with open(data_dir_str + data_file_name, 'r') as _f:
        _line = _f.readline()
        while (_line != ''):
            # print(_line, end='')
            _line_split = _line.split()
            # print(_line_split)
            try:
                _result_list.append( float(_line_split[1]) )
            except:
                pass
            _line = _f.readline()
    #--------------------------#
    print(_result_list)

    # No face detection result, pass
    if len(_result_list) <= 1:
        continue

    data_id_dict = dict()
    # File info
    data_id_dict['idx'] = data_idx_list[_idx]
    data_id_dict['file_name'] = data_file_name
    # "Label" of classes, type: string
    data_id_dict['class'] = None # Move to below. Put this here just for keeping the order of key.
    # Grund truth
    data_id_dict['distance'] = float(data_name_split_list[3])
    data_id_dict['roll'] = ( math.fmod( (_raw_roll_value + 180.0), 360.0 ) - 180.0 ) * _sign_roll
    data_id_dict['pitch'] = _raw_pitch_value * _sign_pitch
    data_id_dict['yaw'] = _raw_yaw_value * _sign_yaw
    #
    data_id_dict['depth_est'] = _result_list[0] * 0.01 # m
    data_id_dict['roll_est'] = _result_list[2] # deg.
    data_id_dict['pitch_est'] = _result_list[3] # deg.
    data_id_dict['yaw_est'] = _result_list[1] # deg.

    # Classify ground truth data! (drpy class)
    #----------------------------------------------#
    _class_dict = dict()
    if is_classified_by_label:
        _class_dict['distance'] = "%d" % data_id_dict['distance']
        _class_dict['pitch'] = "%d" % data_id_dict['pitch']
        _class_dict['roll'] = "%d" % data_id_dict['roll']
        _class_dict['yaw'] = "%d" % data_id_dict['yaw']
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
print(json.dumps(data_list[0], indent=4))
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


    # Get the result
    # np_R_est, np_t_est, t3_est, roll_est, yaw_est, pitch_est, res_norm = pnp_solver.solve_pnp(np_point_image_dict)

    # Get the result from file
    t3_est = data_list[_idx]['depth_est']
    roll_est = data_list[_idx]['roll_est']
    pitch_est = data_list[_idx]['pitch_est']
    yaw_est = data_list[_idx]['yaw_est']
    np_R_est = pnp_solver.get_rotation_matrix_from_Euler( roll_est, yaw_est, pitch_est, is_degree=True )
    np_t_est = np.array([0.0, 0.0, t3_est]).reshape((3,1)) # Assume that the head is at the center of the picture
    res_norm = 0.0 # Ignore this



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
    np_point_image_dict_reproject_GT_ori_golden_patern = pnp_solver.perspective_projection_golden_landmarks(np_R_GT, np_t_GT_est, is_quantized=False, is_pretrans_points=False)

    # Note: Fake this
    np_point_image_dict = np_point_image_dict_reproject_GT_ori_golden_patern


    # Calculate the pixel error of the LMs and the ground-truth projection of golden pattern
    np_LM_GT_error_dict = dict()
    np_LM_GT_error_norm_dict = dict()
    LM_GT_error_total = 0.0
    LM_GT_error_max = 0.0
    LM_GT_error_max_key = ""
    for _k in np_point_image_dict:
        np_LM_GT_error_dict[_k] = np_point_image_dict[_k] - np_point_image_dict_reproject_GT_ori_golden_patern[_k]
        np_LM_GT_error_norm_dict[_k] = np.linalg.norm( np_LM_GT_error_dict[_k] )
        LM_GT_error_total += np_LM_GT_error_norm_dict[_k]
        if np_LM_GT_error_norm_dict[_k] > LM_GT_error_max:
            LM_GT_error_max = np_LM_GT_error_norm_dict[_k]
            LM_GT_error_max_key = _k
    LM_GT_error_average = LM_GT_error_total / len(np_point_image_dict)
    print("(LM_GT_error_average, LM_GT_error_max, LM_GT_error_max_key) = (%f, %f, %s)" % (LM_GT_error_average, LM_GT_error_max, LM_GT_error_max_key))

    # Calculate the pixel error of the LMs and the reprojections of golden pattern
    predict_LM_error_dict = dict()
    np_predict_LM_error_norm_dict = dict()
    predict_LM_error_total = 0.0
    predict_LM_error_max = 0.0
    predict_LM_error_max_key = ""
    for _k in np_point_image_dict:
        predict_LM_error_dict[_k] = np_point_image_dict_reproject[_k] - np_point_image_dict[_k]
        np_predict_LM_error_norm_dict[_k] = np.linalg.norm( predict_LM_error_dict[_k] )
        predict_LM_error_total += np_predict_LM_error_norm_dict[_k]
        if np_predict_LM_error_norm_dict[_k] > predict_LM_error_max:
            predict_LM_error_max = np_predict_LM_error_norm_dict[_k]
            predict_LM_error_max_key = _k
    predict_LM_error_average = predict_LM_error_total / len(np_point_image_dict)
    print("(predict_LM_error_average, predict_LM_error_max, predict_LM_error_max_key) = (%f, %f, %s)" % (predict_LM_error_average, predict_LM_error_max, predict_LM_error_max_key))

    # Calculate the pixel error of the LMs and the reprojections of golden pattern
    predict_GT_error_dict = dict()
    np_predict_GT_error_norm_dict = dict()
    predict_GT_error_total = 0.0
    predict_GT_error_max = 0.0
    predict_GT_error_max_key = ""
    for _k in np_point_image_dict:
        predict_GT_error_dict[_k] = np_point_image_dict_reproject[_k] - np_point_image_dict_reproject_GT_ori_golden_patern[_k]
        np_predict_GT_error_norm_dict[_k] = np.linalg.norm( predict_GT_error_dict[_k] )
        predict_GT_error_total += np_predict_GT_error_norm_dict[_k]
        if np_predict_GT_error_norm_dict[_k] > predict_GT_error_max:
            predict_GT_error_max = np_predict_GT_error_norm_dict[_k]
            predict_GT_error_max_key = _k
    predict_GT_error_average = predict_GT_error_total / len(np_point_image_dict)
    print("(predict_GT_error_average, predict_GT_error_max, predict_GT_error_max_key) = (%f, %f, %s)" % (predict_GT_error_average, predict_GT_error_max, predict_GT_error_max_key))
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

    # # By pass count and fitting error
    # if pass_count < pass_count_threshold: # Note: pass_count >= pass_count_threshold --> passed!!
    #     failed_sample_count += 1
    #     if fitting_error > 1.5:
    #         failed_sample_fit_error_count += 1
    #     # if fitting_error <= 1.5:
    #         failed_sample_filename_list.append(data_list[_idx]['file_name'])
    #         is_storing_case_image = is_storing_fail_case_image

    # By pitch error
    if not drpy_pass_list[2]: # Pitch
        failed_sample_filename_list.append(data_list[_idx]['file_name'])
        is_storing_case_image = is_storing_fail_case_image
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
    # _image_file_name_str = '_'.join(_file_name.split('_')[0:9]) + '.png'
    _image_file_name_str = _file_name[:-4] + '.jpg'
    print('image file name: [%s]' % _image_file_name_str)
    _image_ori_path_str = image_dir_str + _image_file_name_str
    _image_result_unflipped_path_str = image_result_unflipped_dir_str + _image_file_name_str
    _image_result_path_str = image_result_dir_str + _image_file_name_str
    #--------------------------------------------#

    # Load the original image
    #--------------------------------------------#
    _img = cv2.imread(_image_ori_path_str)
    if _img is None:
        print("!! Error occured while loading the image !!\n")
        time.sleep(3.0)
        continue
    _dim = (int(_img.shape[1]*1.5), int(_img.shape[0]*1.5))
    _img = cv2.resize(_img, _dim, interpolation=cv2.INTER_AREA) # Resize 1.5 times
    _img_shape = _img.shape
    print("_img.shape = %s" % str(_img_shape))
    LM_2_image_scale = _img_shape[1] / 320.0

    # Flip the image if needed
    #----------------------------------#
    if is_mirrored_image:
        _img_preprocessed = cv2.flip(_img, 1)
    else:
        _img_preprocessed = _img
    #----------------------------------#

    # Ploting LMs onto the image
    _img_LM = copy.deepcopy(_img_preprocessed)
    # Colors
    _color_RED   = (0, 0, 255)
    _color_GREEN = (0, 255, 0)
    _color_BLUE  = (255, 0, 0)
    # _color_RED   = np.array((0, 0, 255))
    # _color_GREEN = np.array((0, 255, 0))
    # _color_BLUE  = np.array((255, 0, 0))


    # Ploting axies
    #-----------------------------------#
    # Grund truth axes
    vector_scale = 0.2
    uv_o, dir_x, dir_y, dir_z = pnp_solver.perspective_projection_obj_axis(np_R_GT, np_t_GT_est, scale=vector_scale) # Note: use the estimated t since we don't have the grund truth.
    print("(uv_o, dir_x, dir_y, dir_z) = %s" % str((uv_o, dir_x, dir_y, dir_z)))
    _pixel_uv_o = (LM_2_image_scale*uv_o).astype('int')
    _pixel_uv_x1 = (LM_2_image_scale*(uv_o+dir_x)).astype('int')
    _pixel_uv_y1 = (LM_2_image_scale*(uv_o+dir_y)).astype('int')
    _pixel_uv_z1 = (LM_2_image_scale*(uv_o+dir_z)).astype('int')
    # Draw lines
    _line_width = 2
    cv2.line(_img_LM, _pixel_uv_o, _pixel_uv_x1, (0,0,127), _line_width)
    cv2.line(_img_LM, _pixel_uv_o, _pixel_uv_y1, (0,127,0), _line_width)
    cv2.line(_img_LM, _pixel_uv_o, _pixel_uv_z1, (127,0,0), _line_width)

    # Estimated axes
    vector_scale = 0.2
    uv_o, dir_x, dir_y, dir_z = pnp_solver.perspective_projection_obj_axis(np_R_est, np_t_est, scale=vector_scale)
    print("(uv_o, dir_x, dir_y, dir_z) = %s" % str((uv_o, dir_x, dir_y, dir_z)))
    _pixel_uv_o = (LM_2_image_scale*uv_o).astype('int')
    _pixel_uv_x1 = (LM_2_image_scale*(uv_o+dir_x)).astype('int')
    _pixel_uv_y1 = (LM_2_image_scale*(uv_o+dir_y)).astype('int')
    _pixel_uv_z1 = (LM_2_image_scale*(uv_o+dir_z)).astype('int')
    # Draw lines
    _line_width = 1
    cv2.line(_img_LM, _pixel_uv_o, _pixel_uv_x1, (0, 0, 180), _line_width)
    cv2.line(_img_LM, _pixel_uv_o, _pixel_uv_y1, (0, 180, 0), _line_width)
    cv2.line(_img_LM, _pixel_uv_o, _pixel_uv_z1, (180, 0, 0), _line_width)
    #-----------------------------------#

    # Landmarks
    #----------------------------------#
    # [[u,v,1]].T
    for _k in np_point_image_dict:
        # # Landmarks
        # _center_pixel = (np_point_image_dict[_k][0:2,0] * LM_2_image_scale).astype('int')
        # _radius = 3
        # _color = _color_BLUE # BGR
        # # _color = _color_RED # BGR
        # cv2.circle(_img_LM, _center_pixel, _radius, _color, -1)
        # Reprojections of golden pattern onto image using grund truth pose
        _center_pixel = (np_point_image_dict_reproject_GT_ori_golden_patern[_k][0:2,0] * LM_2_image_scale).astype('int')
        _radius = 2
        _color = (127, 127, 0) # BGR
        # _color = _color_RED # BGR
        cv2.circle(_img_LM, _center_pixel, _radius, _color, -1)
        # Reprojections of the golden pattern onto the image using estimated pose
        _center_pixel = (np_point_image_dict_reproject[_k][0:2,0] * LM_2_image_scale).astype('int')
        _radius = 1
        _color = _color_RED # BGR
        # _color = _color_BLUE # BGR
        cv2.circle(_img_LM, _center_pixel, _radius, _color, -1)
    #----------------------------------#

    # Text
    #----------------------------------#
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL # = 5
    fontScale = 1.5
    thickness = 2
    # _text = "Hello world! 012345"
    _est_text = "drpy_est = (%.2f, %.2f, %.2f, %.2f)" % (_result_idx_dict["t3_est"]*100.0,
                                            _result_idx_dict["roll_est"],
                                            _result_idx_dict["pitch_est"],
                                            _result_idx_dict["yaw_est"])
    _GT_text = "drpy_GT = (%.2f, %.2f, %.2f, %.2f)" % (_result_idx_dict["distance_GT"]*100.0,
                                            _result_idx_dict["roll_GT"],
                                            _result_idx_dict["pitch_GT"],
                                            _result_idx_dict["yaw_GT"])
    _err_text = "drpy_est = (%.2f, %.2f, %.2f, %.2f)" % (_result_idx_dict["depth_err"]*100.0,
                                            _result_idx_dict["roll_err"],
                                            _result_idx_dict["pitch_err"],
                                            _result_idx_dict["yaw_err"])
    cv2.putText(_img_LM, _est_text, (0, 50), font, fontScale, (200, 128, 0), thickness, cv2.LINE_AA)
    cv2.putText(_img_LM, _GT_text, (0, 80), font, fontScale, (0, 100, 180), thickness, cv2.LINE_AA)
    cv2.putText(_img_LM, _err_text, (0, 110), font, fontScale, (0, 0, 150), thickness, cv2.LINE_AA)
    #----------------------------------#

    # Dtermine the final image
    #-------------------------#
    # _img_result = _img
    # _img_result = _img_preprocessed
    _img_result = _img_LM
    #-------------------------#

    # Flip the result image if needed
    #----------------------------------#
    if is_mirrored_image:
        _img_result_flipped = cv2.flip(_img_result, 1)
    else:
        _img_result_flipped = _img_result
    #----------------------------------#

    # Save the resulted image
    #----------------------------------#
    cv2.imwrite(_image_result_unflipped_path_str, _img_result )
    cv2.imwrite(_image_result_path_str, _img_result_flipped )
    #----------------------------------#

    # Displaying the image
    #----------------------------------#
    if is_showing_image:
        cv2.imshow(_image_file_name_str, _img_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    #----------------------------------#

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

# # Store the error for statistic
# #----------------------------#
# _result_idx_dict = dict()
# _result_idx_dict['idx'] = data_list[_idx]['idx']
# _result_idx_dict["file_name"] = data_list[_idx]['file_name']
# _result_idx_dict["drpy"] = (distance_GT, roll_GT, pitch_GT, yaw_GT)
# _result_idx_dict["class"] = data_list[_idx]['class']
# # Result summary
# #-------------------------------#
# _result_idx_dict["passed_count"] = pass_count
# _result_idx_dict["is_depth_passed"] = drpy_pass_list[0]
# _result_idx_dict["is_roll_passed"] = drpy_pass_list[1]
# _result_idx_dict["is_pitch_passed"] = drpy_pass_list[2]
# _result_idx_dict["is_yaw_passed"] = drpy_pass_list[3]
# #-------------------------------#
# # R, t, depth, roll, pitch, yaw, residual
# #---#
# # GT
# # Result
# # Error, err = (est - GT)
# # abs(Error), for scalar values
# #-------------------------------#
# # R
# _result_idx_dict["np_R_GT"] = np_R_GT
# _result_idx_dict["np_R_est"] = np_R_est
# _result_idx_dict["np_R_err"] = np_R_est - np_R_GT
# # t
# _result_idx_dict["np_t_GT_est"] = np_t_GT_est
# _result_idx_dict["np_t_est"] = np_t_est
# _result_idx_dict["np_t_err"] = np_t_est - np_t_GT_est
# # depth
# _result_idx_dict["distance_GT"] = distance_GT
# _result_idx_dict["t3_est"] = t3_est
# _result_idx_dict["depth_err"] = t3_est - distance_GT
# _result_idx_dict["abs_depth_err"] = abs(t3_est - distance_GT)
# # roll
# _result_idx_dict["roll_GT"] = roll_GT
# _result_idx_dict["roll_est"] = roll_est
# _result_idx_dict["roll_err"] = roll_est - roll_GT
# _result_idx_dict["abs_roll_err"] = abs(roll_est - roll_GT)
# # pitch
# _result_idx_dict["pitch_GT"] = pitch_GT
# _result_idx_dict["pitch_est"] = pitch_est
# _result_idx_dict["pitch_err"] = pitch_est - pitch_GT
# _result_idx_dict["abs_pitch_err"] = abs(pitch_est - pitch_GT)
# # yaw
# _result_idx_dict["yaw_GT"] = yaw_GT
# _result_idx_dict["yaw_est"] = yaw_est
# _result_idx_dict["yaw_err"] = yaw_est - yaw_GT
# _result_idx_dict["abs_yaw_err"] = abs(yaw_est - yaw_GT)
# # residual
# _result_idx_dict["res_norm"] = res_norm
# _result_idx_dict["res_norm_1000x"] = res_norm * 1000.0
# _result_idx_dict["res_norm_10000x_n_est"] = res_norm * 1000.0 * t3_est # Note: normalized by estimatd value
# _result_idx_dict["res_norm_10000x_n_GT"] = res_norm * 1000.0 * distance_GT # Note: normalized by estimatd value
# # LM-GT error
# _result_idx_dict["LM_GT_error_average_normalize"] = LM_GT_error_average * distance_GT
# _result_idx_dict["LM_GT_error_max_normalize"] = LM_GT_error_max * distance_GT
# _result_idx_dict["LM_GT_error_max_key"] = LM_GT_error_max_key
# # predict-LM error
# _result_idx_dict["predict_LM_error_average_normalize"] = predict_LM_error_average * distance_GT
# _result_idx_dict["predict_LM_error_max_normalize"] = predict_LM_error_max * distance_GT
# _result_idx_dict["predict_LM_error_max_key"] = predict_LM_error_max_key
# # predict-GT error
# _result_idx_dict["predict_GT_error_average_normalize"] = predict_GT_error_average * distance_GT
# _result_idx_dict["predict_GT_error_max_normalize"] = predict_GT_error_max * distance_GT
# _result_idx_dict["predict_GT_error_max_key"] = predict_GT_error_max_key
# #
# result_list.append(_result_idx_dict)
# #----------------------------#




#-------------------------------------------------------#
def get_statistic_of_result(result_list, class_name='all', class_label='all', data_est_key="t3_est", data_GT_key="distance_GT", unit="m", unit_scale=1.0, verbose=True):
    '''
    If data_GT_key is None, we calculate the statistic property of that value;
    whereas if the data_GT_key is given, we calculate the statistic of the error.
    '''
    n_data = len(result_list)
    if n_data == 0:
        return None
    data_est_vec = np.vstack( [ _d[ data_est_key ] for _d in result_list] )
    if data_GT_key is not None:
        data_GT_vec = np.vstack( [ _d[ data_GT_key ] for _d in result_list] )
        _np_data_ratio_vec = data_est_vec / data_GT_vec
        _np_data_error_vec = data_est_vec - data_GT_vec
    else: # We just want to get the statistic of the value itself instead of the statistic of the error
        _np_data_ratio_vec = data_est_vec
        _np_data_error_vec = data_est_vec

    ratio_mean = np.average(_np_data_ratio_vec)
    error_mean = np.average(_np_data_error_vec)
    error_variance = (np.linalg.norm( (_np_data_error_vec - error_mean), ord=2)**2)  / (_np_data_error_vec.shape[0])
    error_stddev = error_variance**0.5
    MAE_2_GT = np.linalg.norm(_np_data_error_vec, ord=1)/(_np_data_error_vec.shape[0])
    MAE_2_mean = np.linalg.norm((_np_data_error_vec - error_mean), ord=1)/(_np_data_error_vec.shape[0])
    max_dev = np.linalg.norm((_np_data_error_vec - error_mean), ord=float('inf'))
    #
    if verbose:
        print("class: [%s], class_label: [%s], n_data=[%d]" % (class_name, class_label, n_data) )
        print("ratio_mean (estimated/actual) = %f" % ratio_mean)
        print("error_mean = %f %s" % (error_mean*unit_scale, unit))
        print("error_stddev = %f %s" % (error_stddev*unit_scale, unit))
        print("max_dev = %f %s" % (max_dev*unit_scale, unit))
        print("MAE_2_GT = %f %s" % (MAE_2_GT*unit_scale, unit))
        print("MAE_2_mean = %f %s" % (MAE_2_mean*unit_scale, unit))
        print("\n")
    # return (ratio_mean, error_mean, error_stddev, MAE_2_GT, MAE_2_mean)

    # Capsulate the statistic results as a dict, which contains all the matric names.
    statis_dict = dict()
    statis_dict['n_data'] = n_data
    statis_dict['m_ratio'] = ratio_mean
    statis_dict['mean(%s)' % unit] = error_mean * unit_scale
    statis_dict['stddev(%s)' % unit] = error_stddev * unit_scale
    statis_dict['max_dev(%s)' % unit] = max_dev * unit_scale
    statis_dict['MAE_2_GT(%s)' % unit] = MAE_2_GT * unit_scale
    statis_dict['MAE_2_mean(%s)' % unit] = MAE_2_mean * unit_scale
    return statis_dict





def _class_order_func(e):
    '''
    For sorting the key of class
    Note: "all" class is placed specifically at the top.
    '''
    # return ( (-1) if (e == "all") else int(e))
    return ( float("-inf") if (e == "all") else float(e))


#-------------------------------------------------------#





# Get simple statistic data
print("\n")
# get_statistic_of_result(result_list)
print("Distance to depth:")
get_statistic_of_result(result_list, class_name='all', class_label='all', data_est_key="t3_est", data_GT_key="distance_GT", unit="cm", unit_scale=100.0, verbose=True)
print("Distance to yaw:")
get_statistic_of_result(result_list, class_name='all', class_label='all', data_est_key="roll_est", data_GT_key="roll_GT", unit="deg.", unit_scale=1.0, verbose=True)
print("Distance to pitch:")
get_statistic_of_result(result_list, class_name='all', class_label='all', data_est_key="pitch_est", data_GT_key="pitch_GT", unit="deg.", unit_scale=1.0, verbose=True)
print("Distance to yaw:")
get_statistic_of_result(result_list, class_name='all', class_label='all', data_est_key="yaw_est", data_GT_key="yaw_GT", unit="deg.", unit_scale=1.0, verbose=True)


# Write to result CSV file
csv_path = result_csv_dir_str + result_csv_file_prefix_str + data_file_str[:-4] + '.csv'
TTBX.write_result_to_csv(result_list, csv_path)




# Get statistic result in each class, filter (by approval func) if required
def approval_func_small_angle(result_list_idx):
    angle_th = 30
    if abs(result_list_idx["roll_GT"]) > angle_th:
        return False
    if abs(result_list_idx["pitch_GT"]) > angle_th:
        return False
    if abs(result_list_idx["yaw_GT"]) > angle_th:
        return False
    return True
def approval_func_large_angle(result_list_idx):
    return (not approval_func_small_angle(result_list_idx))
distance_class_dict = TTBX.get_classified_result(result_list, class_name='distance', approval_func=None)
# distance_class_dict = TTBX.get_classified_result(result_list, class_name='distance', approval_func=approval_func_small_angle)
# distance_class_dict = TTBX.get_classified_result(result_list, class_name='distance', approval_func=approval_func_large_angle)


# Get several statistic of each data in the data subset of each class
#-----------------------------------------------------#
dist_2_depth_statistic_dict = dict()
dist_2_roll_statistic_dict = dict()
dist_2_pitch_statistic_dict = dict()
dist_2_yaw_statistic_dict = dict()
for _label in distance_class_dict:
    dist_2_depth_statistic_dict[_label] = get_statistic_of_result(distance_class_dict[_label], class_name="distance", class_label=_label, data_est_key="t3_est", data_GT_key="distance_GT", unit="cm", unit_scale=100.0, verbose=False)
    dist_2_roll_statistic_dict[_label] = get_statistic_of_result(distance_class_dict[_label], class_name="distance", class_label=_label, data_est_key="roll_est", data_GT_key="roll_GT", unit="deg.", unit_scale=1.0, verbose=False)
    dist_2_pitch_statistic_dict[_label] = get_statistic_of_result(distance_class_dict[_label], class_name="distance", class_label=_label, data_est_key="pitch_est", data_GT_key="pitch_GT", unit="deg.", unit_scale=1.0, verbose=False)
    dist_2_yaw_statistic_dict[_label] = get_statistic_of_result(distance_class_dict[_label], class_name="distance", class_label=_label, data_est_key="yaw_est", data_GT_key="yaw_GT", unit="deg.", unit_scale=1.0, verbose=False)
#-----------------------------------------------------#


class_name = "distance" # Just the name as the info. to the reader
#------------------------------#
statistic_data_name = "depth" # Just the name as the info. to the reader
class_statistic_dict = dist_2_depth_statistic_dict
statistic_txt_path = result_csv_dir_str + result_statistic_txt_file_prefix_str + data_file_str[:-4] + ( "_%s_to_%s" % (class_name, statistic_data_name) ) + '.txt'
statistic_csv_path = result_csv_dir_str + result_statistic_txt_file_prefix_str + data_file_str[:-4] + ( "_%s_to_%s" % (class_name, statistic_data_name) ) + '.csv'
TTBX.write_statistic_to_txt(class_statistic_dict, statistic_txt_path, class_name=class_name, statistic_data_name=statistic_data_name)
TTBX.write_statistic_to_csv(class_statistic_dict, statistic_csv_path, class_name=class_name, statistic_data_name=statistic_data_name, is_horizontal=is_statistic_csv_horizontal)
#
statistic_data_name = "roll" # Just the name as the info. to the reader
class_statistic_dict = dist_2_roll_statistic_dict
statistic_txt_path = result_csv_dir_str + result_statistic_txt_file_prefix_str + data_file_str[:-4] + ( "_%s_to_%s" % (class_name, statistic_data_name) ) + '.txt'
statistic_csv_path = result_csv_dir_str + result_statistic_txt_file_prefix_str + data_file_str[:-4] + ( "_%s_to_%s" % (class_name, statistic_data_name) ) + '.csv'
TTBX.write_statistic_to_txt(class_statistic_dict, statistic_txt_path, class_name=class_name, statistic_data_name=statistic_data_name)
TTBX.write_statistic_to_csv(class_statistic_dict, statistic_csv_path, class_name=class_name, statistic_data_name=statistic_data_name, is_horizontal=is_statistic_csv_horizontal)
#
statistic_data_name = "pitch" # Just the name as the info. to the reader
class_statistic_dict = dist_2_pitch_statistic_dict
statistic_txt_path = result_csv_dir_str + result_statistic_txt_file_prefix_str + data_file_str[:-4] + ( "_%s_to_%s" % (class_name, statistic_data_name) ) + '.txt'
statistic_csv_path = result_csv_dir_str + result_statistic_txt_file_prefix_str + data_file_str[:-4] + ( "_%s_to_%s" % (class_name, statistic_data_name) ) + '.csv'
TTBX.write_statistic_to_txt(class_statistic_dict, statistic_txt_path, class_name=class_name, statistic_data_name=statistic_data_name)
TTBX.write_statistic_to_csv(class_statistic_dict, statistic_csv_path, class_name=class_name, statistic_data_name=statistic_data_name, is_horizontal=is_statistic_csv_horizontal)
#
statistic_data_name = "yaw" # Just the name as the info. to the reader
class_statistic_dict = dist_2_yaw_statistic_dict
statistic_txt_path = result_csv_dir_str + result_statistic_txt_file_prefix_str + data_file_str[:-4] + ( "_%s_to_%s" % (class_name, statistic_data_name) ) + '.txt'
statistic_csv_path = result_csv_dir_str + result_statistic_txt_file_prefix_str + data_file_str[:-4] + ( "_%s_to_%s" % (class_name, statistic_data_name) ) + '.csv'
TTBX.write_statistic_to_txt(class_statistic_dict, statistic_txt_path, class_name=class_name, statistic_data_name=statistic_data_name)
TTBX.write_statistic_to_csv(class_statistic_dict, statistic_csv_path, class_name=class_name, statistic_data_name=statistic_data_name, is_horizontal=is_statistic_csv_horizontal)










def get_all_class_seperated_result(result_list):
    '''
    Hirachy: drpy
    [distance][roll][pitch][yaw]
    '''
    class_dict = dict()
    _d_label_set = set()
    _r_label_set = set()
    _p_label_set = set()
    _y_label_set = set()
    for _idx in range(len(result_list)):
        # Loop thrugh all data
        _d_label = result_list[_idx]['class']["distance"]
        _r_label = result_list[_idx]['class']["roll"]
        _p_label = result_list[_idx]['class']["pitch"]
        _y_label = result_list[_idx]['class']["yaw"]
        # d
        _d_dict = class_dict
        if not _d_label in _d_dict:
            _d_dict[_d_label] = dict()
        # r
        _r_dict = _d_dict[_d_label]
        if not _r_label in _r_dict:
            _r_dict[_r_label] = dict()
        # p
        _p_dict = _r_dict[_r_label]
        if not _p_label in _p_dict:
            _p_dict[_p_label] = dict()
        # y
        _y_dict = _p_dict[_p_label]
        if not _y_label in _y_dict:
            _y_dict[_y_label] = list()

        # Update label set
        _d_label_set.add(_d_label)
        _r_label_set.add(_r_label)
        _p_label_set.add(_p_label)
        _y_label_set.add(_y_label)
        # Append to the class list
        class_dict[_d_label][_r_label][_p_label][_y_label].append(result_list[_idx])

        # # Decide wether to record or not
        # if (approval_func is None) or approval_func( result_list[_idx] ):
        #     class_dict[_label].append(result_list[_idx])

    # Prepare the class label list
    d_label_list = list(_d_label_set)
    r_label_list = list(_r_label_set)
    p_label_list = list(_p_label_set)
    y_label_list = list(_y_label_set)
    d_label_list.sort(key=_class_order_func)
    r_label_list.sort(key=_class_order_func)
    p_label_list.sort(key=_class_order_func)
    y_label_list.sort(key=_class_order_func)
    #
    return (class_dict, d_label_list, r_label_list, p_label_list, y_label_list)
#

def get_drpy_statistic( drpy_class_dict, class_name="distance", data_est_key="t3_est", data_GT_key="distance_GT", unit="m", unit_scale=1.0):
    '''
    '''
    # Get (distance) statistic of each data in the data subset of each class
    #-----------------------------------------------------#
    drpy_2_data_statistic_dict = dict()
    for _d in drpy_class_dict:
        for _r in drpy_class_dict[_d]:
            for _p in drpy_class_dict[_d][_r]:
                for _y in drpy_class_dict[_d][_r][_p]:
                    _result_list = drpy_class_dict[_d][_r][_p][_y]
                    # print(_result_list)
                    _s_data = get_statistic_of_result(_result_list, class_name=class_name, class_label='', data_est_key=data_est_key, data_GT_key=data_GT_key, unit=unit, unit_scale=unit_scale, verbose=False)
                    # print(_s_data)
                    # d
                    _d_dict = drpy_2_data_statistic_dict
                    if not _d in _d_dict:
                        _d_dict[_d] = dict()
                    # r
                    _r_dict = _d_dict[_d]
                    if not _r in _r_dict:
                        _r_dict[_r] = dict()
                    # p
                    _p_dict = _r_dict[_r]
                    if not _p in _p_dict:
                        _p_dict[_p] = dict()
                    # # y
                    # _y_dict = _p_dict[_p]
                    # if not _y in _y_dict:
                    #     _y_dict[_y] = list()
                    drpy_2_data_statistic_dict[_d][_r][_p][_y] = _s_data
    #-----------------------------------------------------#
    return drpy_2_data_statistic_dict



#-----------------------------#
drpy_class_dict, d_label_list, r_label_list, p_label_list, y_label_list = get_all_class_seperated_result(result_list)
# print(drpy_class_dict)
#-----------------------------#

# Get (distance) statistic of each data in the data subset of each class
#-----------------------------#
# drpy to drpy
drpy_2_depth_statistic_dict = get_drpy_statistic(drpy_class_dict, class_name="distance", data_est_key="t3_est", data_GT_key="distance_GT", unit="cm", unit_scale=100.0)
drpy_2_roll_statistic_dict = get_drpy_statistic(drpy_class_dict, class_name="roll", data_est_key="roll_est", data_GT_key="roll_GT", unit="deg.", unit_scale=1.0)
drpy_2_pitch_statistic_dict = get_drpy_statistic(drpy_class_dict, class_name="pitch", data_est_key="pitch_est", data_GT_key="pitch_GT", unit="deg.", unit_scale=1.0)
drpy_2_yaw_statistic_dict = get_drpy_statistic(drpy_class_dict, class_name="yaw", data_est_key="yaw_est", data_GT_key="yaw_GT", unit="deg.", unit_scale=1.0)
# drpy to other values
drpy_2_LM_GT_error_average_normalize_statistic_dict = get_drpy_statistic(drpy_class_dict, class_name="LM_GT_error_average_normalize", data_est_key="LM_GT_error_average_normalize", data_GT_key=None, unit="px_m", unit_scale=1.0)
#-----------------------------#



def write_drpy_2_depth_statistic_CSV(drpy_2_statistic_dict, csv_path, d_label_list=None, r_label_list=None, p_label_list=None, y_label_list=None, matric_label="mean(cm)"):
    '''
    '''
    # # Collect the key list
    # if d_label_list is None:
    #     d_label_list = list( drpy_2_statistic_dict.keys() )
    # if r_label_list is None:
    #     r_label_list = list( drpy_2_statistic_dict[ d_label_list[0] ].keys() )
    # if p_label_list is None:
    #     p_label_list = list( drpy_2_statistic_dict[ d_label_list[0] ][ r_label_list[0] ].keys() )
    # if y_label_list is None:
    #     y_label_list = list( drpy_2_statistic_dict[ d_label_list[0] ][ r_label_list[0] ][ p_label_list[0] ].keys() )
    #
    # _matric_label = "mean" + "(cm)"
    # _matric_label = "MAE_2_GT" + "(cm)"
    _matric_label = matric_label
    #
    row_dict_list = list()
    _d_p_list = list()
    # Roll, verticall
    for _r in r_label_list: # For vertical table (outer row)
        for _y in y_label_list: # In each table (inner row)
            _row_dict = dict()
            _r_y_label = "r=%s, y=%s" % (_r, _y)

            _bar_count = 0
            for _d in d_label_list: # For horizontal table (outer column)
                _d_p_label = "d=%s" % (_d)
                if not _d_p_label in _d_p_list:
                    _d_p_list.append( _d_p_label )
                _row_dict[_d_p_label] = _r_y_label # 1st column of a table
                #
                for _p in p_label_list: # In each table (inner column)
                    _bar_count += 1
                    _d_p_label = "d=%s, p=%s" % (_d, _p)
                    #
                    if not _d_p_label in _d_p_list:
                        _d_p_list.append( _d_p_label )
                    #
                    # print("(d,r,p,y) = %s" % str((_d, _r, _p, _y)))
                    # _a = drpy_2_statistic_dict[_d][_r][_p][_y]
                    # print(_a)
                    try:
                        _row_dict[_d_p_label] = drpy_2_statistic_dict[_d][_r][_p][_y][_matric_label] # other columns
                    except:
                        _row_dict[_d_p_label] = "-"
                # Finish a table horizontally
                _d_p_label = "|%d" % _bar_count
                if not _d_p_label in _d_p_list:
                    _d_p_list.append( _d_p_label )
                _row_dict[_d_p_label] = ""
            # Finish a table virtically
            row_dict_list.append(_row_dict)
        #
        row_dict_list.append(dict()) # Empty line
    #
    fieldnames = _d_p_list

    with open(statistic_csv_path, mode='w') as _csv_f:
        _csv_w = csv.DictWriter(_csv_f, fieldnames=fieldnames, extrasaction='ignore')
        #
        _csv_w.writeheader()
        _csv_w.writerows(row_dict_list)
        # for _e_dict in row_dict_list:
        #     _csv_w.writerow(_e_dict)
        print("\n*** Wrote the drpy statistic results to the csv file:\n\t[%s]\n" % ( csv_path))




#---------------------------------------------------#
# Record the data distribution (drpy histogram)
# Generate the drpy data
matric_name = "n_data"
statistic_data_name = "all" # Just the name as the info. to the reader
drpy_2_data_statistic_dict = drpy_2_depth_statistic_dict # Since our data is complete, it's no matter we use the depth's data or other's
matric_label = "n_data"
statistic_csv_path = result_csv_dir_str + result_statistic_txt_file_prefix_str + data_file_str[:-4] + ( "_%s_to_%s" % ("drpy", statistic_data_name) ) + '_' + matric_label + '.csv'
write_drpy_2_depth_statistic_CSV(drpy_2_data_statistic_dict, statistic_csv_path, d_label_list, r_label_list, p_label_list, y_label_list, matric_label=matric_label)



# Statistic about bias
#---------------------------------------------------#
# Generate the drpy data
matric_name = "mean"
# matric_name = "MAE_2_GT" <-- bias + deviation

statistic_data_name = "depth" # Just the name as the info. to the reader
drpy_2_data_statistic_dict = drpy_2_depth_statistic_dict
unit = "cm"
matric_label = "%s(%s)" % (matric_name, unit)
statistic_csv_path = result_csv_dir_str + result_statistic_txt_file_prefix_str + data_file_str[:-4] + ( "_%s_to_%s" % ("drpy", statistic_data_name) ) + '_' + matric_label + '.csv'
write_drpy_2_depth_statistic_CSV(drpy_2_data_statistic_dict, statistic_csv_path, d_label_list, r_label_list, p_label_list, y_label_list, matric_label=matric_label)

statistic_data_name = "roll" # Just the name as the info. to the reader
drpy_2_data_statistic_dict = drpy_2_roll_statistic_dict
unit = "deg."
matric_label = "%s(%s)" % (matric_name, unit)
statistic_csv_path = result_csv_dir_str + result_statistic_txt_file_prefix_str + data_file_str[:-4] + ( "_%s_to_%s" % ("drpy", statistic_data_name) ) + '_' + matric_label + '.csv'
write_drpy_2_depth_statistic_CSV(drpy_2_data_statistic_dict, statistic_csv_path, d_label_list, r_label_list, p_label_list, y_label_list, matric_label=matric_label)

statistic_data_name = "pitch" # Just the name as the info. to the reader
drpy_2_data_statistic_dict = drpy_2_pitch_statistic_dict
unit = "deg."
matric_label = "%s(%s)" % (matric_name, unit)
statistic_csv_path = result_csv_dir_str + result_statistic_txt_file_prefix_str + data_file_str[:-4] + ( "_%s_to_%s" % ("drpy", statistic_data_name) ) + '_' + matric_label + '.csv'
write_drpy_2_depth_statistic_CSV(drpy_2_data_statistic_dict, statistic_csv_path, d_label_list, r_label_list, p_label_list, y_label_list, matric_label=matric_label)

statistic_data_name = "yaw" # Just the name as the info. to the reader
drpy_2_data_statistic_dict = drpy_2_yaw_statistic_dict
unit = "deg."
matric_label = "%s(%s)" % (matric_name, unit)
statistic_csv_path = result_csv_dir_str + result_statistic_txt_file_prefix_str + data_file_str[:-4] + ( "_%s_to_%s" % ("drpy", statistic_data_name) ) + '_' + matric_label + '.csv'
write_drpy_2_depth_statistic_CSV(drpy_2_data_statistic_dict, statistic_csv_path, d_label_list, r_label_list, p_label_list, y_label_list, matric_label=matric_label)

# LM_GT_error_average_normalize
statistic_data_name = "LM_GT_error_average_normalize" # Just the name as the info. to the reader
drpy_2_data_statistic_dict = drpy_2_LM_GT_error_average_normalize_statistic_dict
unit = "px_m"
matric_label = "%s(%s)" % (matric_name, unit)
statistic_csv_path = result_csv_dir_str + result_statistic_txt_file_prefix_str + data_file_str[:-4] + ( "_%s_to_%s" % ("drpy", statistic_data_name) ) + '_' + matric_label + '.csv'
write_drpy_2_depth_statistic_CSV(drpy_2_data_statistic_dict, statistic_csv_path, d_label_list, r_label_list, p_label_list, y_label_list, matric_label=matric_label)


# Statistic about deviation
#---------------------------------------------------#
# Generate the drpy data
matric_name = "stddev"
# matric_name = "MAE_2_mean"

statistic_data_name = "depth" # Just the name as the info. to the reader
drpy_2_data_statistic_dict = drpy_2_depth_statistic_dict
unit = "cm"
matric_label = "%s(%s)" % (matric_name, unit)
statistic_csv_path = result_csv_dir_str + result_statistic_txt_file_prefix_str + data_file_str[:-4] + ( "_%s_to_%s" % ("drpy", statistic_data_name) ) + '_' + matric_label + '.csv'
write_drpy_2_depth_statistic_CSV(drpy_2_data_statistic_dict, statistic_csv_path, d_label_list, r_label_list, p_label_list, y_label_list, matric_label=matric_label)

statistic_data_name = "roll" # Just the name as the info. to the reader
drpy_2_data_statistic_dict = drpy_2_roll_statistic_dict
unit = "deg."
matric_label = "%s(%s)" % (matric_name, unit)
statistic_csv_path = result_csv_dir_str + result_statistic_txt_file_prefix_str + data_file_str[:-4] + ( "_%s_to_%s" % ("drpy", statistic_data_name) ) + '_' + matric_label + '.csv'
write_drpy_2_depth_statistic_CSV(drpy_2_data_statistic_dict, statistic_csv_path, d_label_list, r_label_list, p_label_list, y_label_list, matric_label=matric_label)

statistic_data_name = "pitch" # Just the name as the info. to the reader
drpy_2_data_statistic_dict = drpy_2_pitch_statistic_dict
unit = "deg."
matric_label = "%s(%s)" % (matric_name, unit)
statistic_csv_path = result_csv_dir_str + result_statistic_txt_file_prefix_str + data_file_str[:-4] + ( "_%s_to_%s" % ("drpy", statistic_data_name) ) + '_' + matric_label + '.csv'
write_drpy_2_depth_statistic_CSV(drpy_2_data_statistic_dict, statistic_csv_path, d_label_list, r_label_list, p_label_list, y_label_list, matric_label=matric_label)

statistic_data_name = "yaw" # Just the name as the info. to the reader
drpy_2_data_statistic_dict = drpy_2_yaw_statistic_dict
unit = "deg."
matric_label = "%s(%s)" % (matric_name, unit)
statistic_csv_path = result_csv_dir_str + result_statistic_txt_file_prefix_str + data_file_str[:-4] + ( "_%s_to_%s" % ("drpy", statistic_data_name) ) + '_' + matric_label + '.csv'
write_drpy_2_depth_statistic_CSV(drpy_2_data_statistic_dict, statistic_csv_path, d_label_list, r_label_list, p_label_list, y_label_list, matric_label=matric_label)

# LM_GT_error_average_normalize
statistic_data_name = "LM_GT_error_average_normalize" # Just the name as the info. to the reader
drpy_2_data_statistic_dict = drpy_2_LM_GT_error_average_normalize_statistic_dict
unit = "px_m"
matric_label = "%s(%s)" % (matric_name, unit)
statistic_csv_path = result_csv_dir_str + result_statistic_txt_file_prefix_str + data_file_str[:-4] + ( "_%s_to_%s" % ("drpy", statistic_data_name) ) + '_' + matric_label + '.csv'
write_drpy_2_depth_statistic_CSV(drpy_2_data_statistic_dict, statistic_csv_path, d_label_list, r_label_list, p_label_list, y_label_list, matric_label=matric_label)
