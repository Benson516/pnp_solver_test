import numpy as np
import copy
import time
import csv
import os
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

# Make directories for storing the results if they do not exist.
#------------------------------------------------------------------#
os.makedirs(image_result_unflipped_dir_str, mode=0o777, exist_ok=True)
os.makedirs(image_result_dir_str, mode=0o777, exist_ok=True)
os.makedirs(result_csv_dir_str, mode=0o777, exist_ok=True)
#------------------------------------------------------------------#

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

# Get parameters for classification
class_drpy_param_dict = TTBX.get_classification_parameters(drpy_class_format=drpy_class_format)
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
    _class_dict['distance'] = TTBX.classify_drpy(class_drpy_param_dict, data_id_dict['distance'], class_name="depth", is_classified_by_label=is_classified_by_label)
    _class_dict['roll']     = TTBX.classify_drpy(class_drpy_param_dict, data_id_dict['roll'], class_name="roll", is_classified_by_label=is_classified_by_label)
    _class_dict['pitch']    = TTBX.classify_drpy(class_drpy_param_dict, data_id_dict['pitch'], class_name="pitch", is_classified_by_label=is_classified_by_label)
    _class_dict['yaw']      = TTBX.classify_drpy(class_drpy_param_dict, data_id_dict['yaw'], class_name="yaw", is_classified_by_label=is_classified_by_label)
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







# Golden patterns
# 3D landmark point - local coordinate
#----------------------------------------#
point_3d_dict_list = list()
pattern_scale_list = list()
# Alexander
point_3d_dict_list.append( TTBX.get_golden_pattern(pattern_name="Alexander") )
pattern_scale_list.append(pattern_scale)
# # Holly
# point_3d_dict_list.append( TTBX.get_golden_pattern(pattern_name="Holly") )
# pattern_scale_list.append(pattern_scale)
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
    image_file_name_str = _file_name[:-4] + '.jpg'
    # print('image file name: [%s]' % image_file_name_str)
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
                is_ploting_LMs=False,
                LM_img_width=320,
                is_showing_image=is_showing_image)
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
