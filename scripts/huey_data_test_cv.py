import numpy as np
import copy
import time
import csv
import os
import joblib
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
# specific_drpy = {"distance":"60", "roll":"0", "pitch":"0", "yaw":"-40"}
# specific_drpy = {"distance":"60", "roll":"0", "pitch":"0", "yaw":"0"}
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

# Get parameters for classification
class_drpy_param_dict = TTBX.get_classification_parameters(drpy_class_format=drpy_class_format)
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
#
GT_R_t_dict = dict() # Note: t is estimated from the prediction since we only got the GT depth.
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
    np_point_image_dict["brow_cl_35"] = TTBX.convert_pixel_to_homo(LM_pixel_data_matrix[35], mirrored=is_mirrored_image)
    np_point_image_dict["brow_il_37"] = TTBX.convert_pixel_to_homo(LM_pixel_data_matrix[37], mirrored=is_mirrored_image)
    np_point_image_dict["brow_ir_42"] = TTBX.convert_pixel_to_homo(LM_pixel_data_matrix[42], mirrored=is_mirrored_image)
    np_point_image_dict["brow_cr_44"] = TTBX.convert_pixel_to_homo(LM_pixel_data_matrix[44], mirrored=is_mirrored_image)
    np_point_image_dict["eye_lo_60"] = TTBX.convert_pixel_to_homo(LM_pixel_data_matrix[60], mirrored=is_mirrored_image)
    np_point_image_dict["eye_li_64"] = TTBX.convert_pixel_to_homo(LM_pixel_data_matrix[64], mirrored=is_mirrored_image)
    np_point_image_dict["eye_ro_72"] = TTBX.convert_pixel_to_homo(LM_pixel_data_matrix[72], mirrored=is_mirrored_image)
    np_point_image_dict["eye_ri_68"] = TTBX.convert_pixel_to_homo(LM_pixel_data_matrix[68], mirrored=is_mirrored_image)
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
    # point_3d_dict = point_3d_dict_list[0] # Use the first golden pattern
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

    # Convert the LMs to frontal look
    #----------------------------------------------------#
    # d2 = 1.0 # m
    d2 = 0.3 # m
    np_nf = np.array([0., 0., 1.]).reshape((3,1))
    H_a2o = np_R_est + (np_t_est / d2 - np_R_est @ np_nf) @ (np_nf.T)
    # H_a2o = np_R_est
    G_a2o = np_K_camera_est @ H_a2o @ np.linalg.inv(np_K_camera_est)
    G_o2a = np.linalg.inv(G_a2o)
    #
    np_point_image_frontal_dict = dict()
    for _key in np_point_image_dict:
        _homo_point_o = np_point_image_dict[_key]
        _homo_point_a = G_o2a @ _homo_point_o
        _homo_point_a /= _homo_point_a[2,0] # Normalize
        np_point_image_frontal_dict[_key] = _homo_point_a
    #----------------------------------------------------#


    # Compare result
    #-----------------------------#

    # Grund truth (R,t)
    roll_GT = data_list[_idx]['roll']
    pitch_GT = data_list[_idx]['pitch']
    yaw_GT = data_list[_idx]['yaw']
    np_R_GT = pnp_solver.get_rotation_matrix_from_Euler( roll_GT, yaw_GT, pitch_GT, is_degree=True )
    # print("np_R_GT = \n%s" % str(np_R_GT))
    distance_GT = data_list[_idx]['distance'] *0.01 # cm --> m
    np_t_GT_est = (np_t_est/t3_est) * distance_GT

    # Collect the ground truth (especially the np_t_GT_est) for other programs
    #-----------------------------------------#
    _file_name = data_list[_idx]['file_name']
    original_file_name = '_'.join(_file_name.split('_')[0:9])
    print("original_file_name = %s" % original_file_name)
    GT_R_t_dict[original_file_name] = dict()
    GT_R_t_dict[original_file_name]["np_R_GT"] = np_R_GT
    GT_R_t_dict[original_file_name]["np_t_GT_est"] = np_t_GT_est
    #-----------------------------------------#


    # Apply criteria
    #----------------------------------------------#
    drpy_pass_list, pass_count = TTBX.check_if_the_sample_passed(
                            (t3_est*100.0, roll_est, pitch_est, yaw_est),
                            (data_list[_idx]['distance'], data_list[_idx]['roll'], data_list[_idx]['pitch'], data_list[_idx]['yaw']),
                            (10.0, 10.0, 10.0, 10.0) )
    print("pass_count = %d  |  drpy_pass_list = %s" % (pass_count, str(drpy_pass_list)))
    fail_count = len(drpy_pass_list) - pass_count
    #----------------------------------------------#


    # Compare the results and wrote to the result dict
    #----------------------------------------------------------#
    rpy_est = (roll_est, pitch_est, yaw_est)
    rpy_GT = (roll_GT, pitch_GT, yaw_GT)
    _res_list = TTBX.compare_result_and_generate_result_dict(
                                pnp_solver,
                                data_list[_idx],
                                np_R_est, np_t_est, rpy_est,
                                np_R_GT, np_t_GT_est, rpy_GT,
                                res_norm,
                                fail_count, pass_count, drpy_pass_list,
                                np_point_image_dict=np_point_image_dict,
                                verbose=verbose
                            )
    _result_idx_dict, np_point_image_dict_reproject_GT_ori_golden_patern, np_point_image_dict_reproject = _res_list
    #----------------------------------------------------------#

    result_list.append(_result_idx_dict)
    #----------------------------#




    # Determine if the case should be stored for further inspection
    #----------------------------------------------#
    fitting_error = _result_idx_dict["predict_LM_error_average_normalize"]

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
    #----------------------------------------------#




    # Image Display
    #====================================================#
    if not (is_showing_image or is_storing_case_image):
        continue


    # Get the file name of the image
    #--------------------------------------------#
    _file_name = data_list[_idx]['file_name']
    image_file_name_str = '_'.join(_file_name.split('_')[0:9]) + '.png'
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
                np_homo_point_LM_frontal_view_dict=np_point_image_frontal_dict,
                is_mirrored_image=is_mirrored_image,
                is_ploting_LMs=True,
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


# # Pack the GT_R_t_dict for other program
# #-------------------#
# GT_R_t_dict_dir = "/home/benson516/test_PnP_solver/pnp_solver_test/scripts/ground_truth_R_t/"
# GT_R_t_dict_path = GT_R_t_dict_dir + "GT_R_t_dict.pkl"
# joblib.dump(GT_R_t_dict, GT_R_t_dict_path)
# #-------------------#



# Data analysis and saving
#-----------------------------------#
TTBX.data_analysis_and_saving(result_list, result_csv_dir_str, result_csv_file_prefix_str, result_statistic_txt_file_prefix_str, data_file_str, is_statistic_csv_horizontal=is_statistic_csv_horizontal)

TTBX.data_analysis_pandas(result_list, result_csv_dir_str)
#-----------------------------------#
