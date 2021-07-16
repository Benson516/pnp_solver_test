import numpy as np
import copy
import time
import csv
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
    class_yaw_bins = [-20, -10, 10, 20] # Only the middle bound values
    print("class_yaw_label = %s" % class_yaw_label)
    print("class_yaw_bins = %s" % class_yaw_bins)
    #
else:
    pass
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
    _class_dict['distance'] = class_depth_label[ np.digitize( data_id_dict['distance'], class_depth_bins) ]
    _class_dict['roll'] = class_roll_label[ np.digitize( data_id_dict['roll'], class_roll_bins) ]
    _class_dict['pitch'] = class_pitch_label[ np.digitize( data_id_dict['pitch'], class_pitch_bins) ]
    _class_dict['yaw'] = class_yaw_label[ np.digitize( data_id_dict['yaw'], class_yaw_bins) ]
    #
    data_id_dict['class'] = _class_dict
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


    # Break the stress test
    #----------------------------#
    if stop_at_fail_cases:
        if is_stress_test:
            if pass_count < pass_count_threshold:
                print("Fail, break the stress test!!")
                is_continuing_to_next_sample = False
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
