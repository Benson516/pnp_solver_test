import numpy as np
import copy
import time
import csv
import os
#
import cv2
#
import PNP_SOLVER_LIB as PNPS
import TEST_TOOLBOX as TTBX

# ctrl+c
from signal import signal, SIGINT


#---------------------------#
# Landmark (LM) dataset
data_dir_str = '/home/benson516/test_PnP_solver/dataset/Huey_face_landmarks_pose/'
data_file_str = 'random.txt'
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
is_stress_test = True
# is_stress_test = False

# Data generation
# is_random_pose = True
is_random_pose = False
#
is_quantized = True
# is_quantized = False

# stop_at_fail_cases = True
stop_at_fail_cases = False

#
# DATA_COUNT = 3
# DATA_COUNT = 1000
DATA_COUNT = 10000 # 1000
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
if is_stress_test:
    is_random_pose = True
    is_showing_image = False
    if not stop_at_fail_cases:
        verbose = False
if not is_random_pose:
    DATA_COUNT = 1

#

# Parameters of the data
#---------------------------#
# is_mirrored_image = True
is_mirrored_image = False
pattern_scale = 1.0 # 0.85 # Multiply onto the golden pattern
#---------------------------#



# ============= Random data ================
#-------------------------------------------------------#
# Parameters and data
# Camera intrinsic matrix (Ground truth)
#----------------------------------------#
f_camera = 225.68717584155982
#
fx_camera = f_camera
# fx_camera = (-f_camera) if is_mirrored_image else f_camera # Note: mirrored image LM features
fy_camera = f_camera
xo_camera = 320/2.0
yo_camera = 240/2.0
np_K_camera_GT = np.array([[fx_camera, 0.0, xo_camera], [0.0, fy_camera, yo_camera], [0.0, 0.0, 1.0]]) # Ground truth
print("np_K_camera_GT = \n%s" % str(np_K_camera_GT))
#----------------------------------------#









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


# Create the GT PnP class for generating the LMs by projecting the golden patterns
#----------------------------------------#
# Golden pattern (simply copy the GT one)
point_3d_dict_GT_list = copy.deepcopy(point_3d_dict_list)
pattern_scale_GT_list = copy.deepcopy(pattern_scale_list)
pnp_solver_GT = PNPS.PNP_SOLVER(np_K_camera_GT, point_3d_dict_GT_list, pattern_scale_list=pattern_scale_GT_list, verbose=verbose)
#----------------------------------------#







#-------------------------------------------------------#
# Parameters and data
# Camera intrinsic matrix (Estimated)
#----------------------------------------#
f_camera = 225.68717584155982
#
fx_camera = f_camera
# fx_camera = (-f_camera) if is_mirrored_image else f_camera # Note: mirrored image LM features
fy_camera = f_camera
xo_camera = 320/2.0
yo_camera = 240/2.0
np_K_camera_est = np.array([[fx_camera, 0.0, xo_camera], [0.0, fy_camera, yo_camera], [0.0, 0.0, 1.0]]) # Estimated
print("np_K_camera_est = \n%s" % str(np_K_camera_est))
#----------------------------------------#

# Create the solver
#----------------------------------------#
pnp_solver = PNPS.PNP_SOLVER(np_K_camera_est, point_3d_dict_list, pattern_scale_list=pattern_scale_list, verbose=verbose)
#-------------------------------------------------------#




# ============= Loading Data ================
#----------------------------------------------------------#

# Ground truth classification
#-------------------------------#
# Format
drpy_class_format = "drpy_expand"
# drpy_class_format = "HMI_inspection"

# Get parameters for classification
class_drpy_param_dict = TTBX.get_classification_parameters(drpy_class_format=drpy_class_format)
#-------------------------------#



#-------------------------------#
# SIGINT
received_SIGINT = False
def SIGINT_handler(signal_received, frame):
    global received_SIGINT
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected. Exiting gracefully')
    received_SIGINT = True

# Tell Python to run the handler() function when SIGINT is recieved
signal(SIGINT, SIGINT_handler)
#-------------------------------#




# ============= Start testing ================
# Loop through data
#-------------------------------------------------------#
# Random generator
# random_seed = 42
random_seed = None
random_gen = np.random.default_rng(seed=random_seed)

# Collect the result
#--------------------------#
data_list = list()
result_list = list()
failed_sample_filename_list = list()
failed_sample_count = 0
failed_sample_fit_error_count = 0
#--------------------------#

s_stamp = time.time()

# Loop, stress test
is_continuing_to_next_sample = True
sample_count = 0
while (sample_count < DATA_COUNT) and (not received_SIGINT):
    #
    if not is_continuing_to_next_sample:
        break
    sample_count += 1
    # The idx for reference into the data list
    _idx = sample_count-1
    #

    # Convert the original data to structured data_list
    #-------------------------------------------------------#
    data_id_dict = dict()
    # File info
    data_id_dict['idx'] = _idx # data_idx_list[_idx]
    data_id_dict['file_name'] = None # Move to below. Put this here just for keeping the order of key.
    # "Label" of classes, type: string
    data_id_dict['class'] = None # Move to below. Put this here just for keeping the order of key.

    # Grund truth pose
    if is_random_pose:
        _angle_range = 45.0 # deg
        _roll = random_gen.uniform( (-_angle_range), _angle_range, None)
        _pitch = random_gen.uniform( (-_angle_range), _angle_range, None)
        _yaw = random_gen.uniform( (-_angle_range), _angle_range, None)
        #
        _depth = random_gen.uniform(20, 225, None)/100.0 # m
        _FOV_max = 45.0 # 1.0 # 45.0 # deg.
        _FOV_x = random_gen.uniform((-_FOV_max), _FOV_max, None)
        _FOV_y = random_gen.uniform((-_FOV_max), _FOV_max, None)
        _np_t_GT = np.zeros((3,1))
        _np_t_GT[0,0] = _depth * np.tan( np.deg2rad(_FOV_x) )
        _np_t_GT[1,0] = _depth * np.tan( np.deg2rad(_FOV_y) )
        _np_t_GT[2,0] = _depth
    else:
        _roll = 15.0 # deg.
        _pitch = -15.0 # deg.
        _yaw = 45.0 # deg.
        # _roll = 40.0 # deg.
        # _pitch = -60.0 # deg.
        # _yaw = 67.0 # deg.
        #
        _np_t_GT = np.zeros((3,1))
        _np_t_GT[0,0] = 0.35 # m
        _np_t_GT[1,0] = 0.35 # m
        _np_t_GT[2,0] = 0.5 # m
        # _np_t_GT[0,0] = 3.5 # m
        # _np_t_GT[1,0] = -3.5 # m
        # _np_t_GT[2,0] = 0.1 # m
        # _np_t_GT[0,0] = 0.0 # m
        # _np_t_GT[1,0] = 0.0 # m
        # _np_t_GT[2,0] = 1.2 # m
    #
    np_R_GT = pnp_solver_GT.get_rotation_matrix_from_Euler(_roll, _yaw, _pitch, is_degree=True)
    np_t_GT = _np_t_GT
    #
    data_id_dict['distance'] = _np_t_GT[2,0] * 100.0 # cm
    data_id_dict['roll'] = _roll
    data_id_dict['pitch'] = _pitch
    data_id_dict['yaw'] = _yaw
    #
    data_id_dict['np_R_GT'] = np_R_GT
    data_id_dict['np_t_GT'] = np_t_GT
    # Test inputs
    pnp_solver_GT.set_golden_pattern_id(0) # Use the number 0 pattern to generate the LM
    data_id_dict['LM_pixel_dict'] = pnp_solver_GT.perspective_projection_golden_landmarks(np_R_GT, np_t_GT, is_quantized=is_quantized, is_pretrans_points=False, is_returning_homogeneous_vec=True) # Homogeneous coordinate

    # Classify ground truth data! (drpy class)
    #----------------------------------------------#
    _class_dict = dict()
    _class_dict['distance'] = TTBX.classify_drpy(class_drpy_param_dict, data_id_dict['distance'], class_name="depth", is_classified_by_label=False)
    _class_dict['roll']     = TTBX.classify_drpy(class_drpy_param_dict, data_id_dict['roll'], class_name="roll", is_classified_by_label=False)
    _class_dict['pitch']    = TTBX.classify_drpy(class_drpy_param_dict, data_id_dict['pitch'], class_name="pitch", is_classified_by_label=False)
    _class_dict['yaw']      = TTBX.classify_drpy(class_drpy_param_dict, data_id_dict['yaw'], class_name="yaw", is_classified_by_label=False)
    data_id_dict['class'] = _class_dict
    #
    data_id_dict['file_name'] = "random_drpy_%s_%s_%s_%s" % (_class_dict['distance'], _class_dict['roll'], _class_dict['pitch'], _class_dict['yaw'])
    #----------------------------------------------#


    # Sppend to the total data list
    data_list.append(data_id_dict)
    #
    # print(data_list[0])
    #-------------------------------------------------------#


    print("\n-------------- data_idx = %d (process idx = %d)--------------\n" % (data_list[_idx]['idx'], _idx))
    # print('file file_name: [%s]' % data_list[_idx]['file_name'])


    # Just reference the original projection data
    LM_pixel_dict = data_list[_idx]['LM_pixel_dict']
    np_point_image_dict = LM_pixel_dict


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

    # Break the stress test
    #----------------------------#
    if stop_at_fail_cases:
        if is_stress_test:
            if pass_count < pass_count_threshold:
                print("Fail, break the stress test!!")
                is_continuing_to_next_sample = False
    #----------------------------#




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



# Data analysis and saving
#-----------------------------------#
TTBX.data_analysis_and_saving(result_list, result_csv_dir_str, result_csv_file_prefix_str, result_statistic_txt_file_prefix_str, data_file_str, is_statistic_csv_horizontal=is_statistic_csv_horizontal)
#-----------------------------------#
