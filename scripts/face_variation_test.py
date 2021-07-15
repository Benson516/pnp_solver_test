import numpy as np
import copy
import time
import csv
import json
import heapq
#
import joblib
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
data_file_str = 'face_variation.txt'
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
#---------------------------#
workstate_dir_str = '/home/benson516/test_PnP_solver/pnp_solver_test/scripts/workstate/'
workstate_file_name_str = 'face_variation_work_state.pkl'
workstate_path_str = workstate_dir_str + workstate_file_name_str
#---------------------------#

# Behavior of this program
#---------------------------#
is_stress_test = True
# is_stress_test = False

# Data generation
# Pose
# is_random_pose = True
is_random_pose = False

# Pattern (face "shape")
is_randomly_perturbing_pattern = True
# is_randomly_perturbing_pattern = False

#
is_quantized = True
# is_quantized = False

# stop_at_fail_cases = True
stop_at_fail_cases = False

#
# DATA_COUNT = 3
# DATA_COUNT = 1000
# DATA_COUNT = 10000 # 1000
DATA_COUNT = 120000 # 60000 # 1000 # Estimated 30 miniute to run
#
verbose = True
# verbose = False
# Image display
is_showing_image = True
# is_showing_image = False
#
# Fail cases investigation
# is_storing_fail_case_image = True
is_storing_fail_case_image = False
# Note: pass_count >= pass_count_threshold --> passed!!
pass_count_threshold = 4 # 3 # Note: max=4. If there are less than (not included) pass_count_threshold pass items, store the image
#
# Statistic CSV file
is_statistic_csv_horizontal = True # class --> (right)
# is_statistic_csv_horizontal = False # class (|, down)
#---------------------------#

# Not to flush the screen
if is_stress_test:
    # is_random_pose = True
    is_randomly_perturbing_pattern = True
    is_showing_image = False
    if not stop_at_fail_cases:
        verbose = False
# if not is_random_pose:
#     DATA_COUNT = 1

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
point_3d_dict["eye_c_51"] = [0.0, 0.0, 0.0]
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
# Min heaps (priority queue)
heap_neg_abs_depth_err = list()
heap_neg_abs_roll_err = list()
heap_neg_abs_pitch_err = list()
heap_neg_abs_yaw_err = list()
#
failed_sample_filename_list = list()
failed_sample_count = 0
failed_sample_fit_error_count = 0
#--------------------------#

s_stamp = time.time()
delta_time = 0.0

# Loop, stress test
is_continuing_to_next_sample = True
sample_count = 0

# Load previous states
#-------------------------------#
workstate = None
try:
    workstate = joblib.load(workstate_path_str)
except Exception as e:
    print(e)
    print("\nCreate new workstate\n")
#
if workstate is not None:
    sample_count = workstate["sample_count"]
    delta_time = workstate["delta_time"]
    data_list = workstate["data_list"]
    result_list = workstate["result_list"]
    heap_neg_abs_depth_err = workstate["heap_neg_abs_depth_err"]
    heap_neg_abs_roll_err = workstate["heap_neg_abs_roll_err"]
    heap_neg_abs_pitch_err = workstate["heap_neg_abs_pitch_err"]
    heap_neg_abs_yaw_err = workstate["heap_neg_abs_yaw_err"]
else:
    workstate = dict()
print("\nsample_count = %d\n" % sample_count)
# time.sleep(10)
#-------------------------------#
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

    # Generate grund-truth pose
    #------------------------------------#
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
        _roll = 0.0 # deg.
        _pitch = 0.0 # deg.
        _yaw = 0.0 # deg.
        # _roll = 15.0 # deg.
        # _pitch = -15.0 # deg.
        # _yaw = 45.0 # deg.
        # _roll = 40.0 # deg.
        # _pitch = -60.0 # deg.
        # _yaw = 67.0 # deg.
        #
        _np_t_GT = np.zeros((3,1))
        _np_t_GT[0,0] = 0.0 # m
        _np_t_GT[1,0] = 0.0 # m
        _np_t_GT[2,0] = 1.0 # m
        # _np_t_GT[0,0] = 0.35 # m
        # _np_t_GT[1,0] = 0.35 # m
        # _np_t_GT[2,0] = 0.5 # m
        # _np_t_GT[0,0] = 3.5 # m
        # _np_t_GT[1,0] = -3.5 # m
        # _np_t_GT[2,0] = 0.1 # m
        # _np_t_GT[0,0] = 0.0 # m
        # _np_t_GT[1,0] = 0.0 # m
        # _np_t_GT[2,0] = 1.2 # m
    #------------------------------------#

    # Perturb the pattern (variate the face shape)
    #------------------------------------#
    GT_golden_pattern_id = 0
    np_pattern_perturbation_dict = dict()
    fixed_pattern_key = "eye_c_51"
    perturb_radius_point = 0.02 # m, currently uniform in all direction
    #
    if is_randomly_perturbing_pattern:
        _new_point_3d_dict = copy.deepcopy(point_3d_dict_GT_list[GT_golden_pattern_id])
        _new_pattern_scale = pattern_scale_GT_list[GT_golden_pattern_id]

        # Generate and normalize the perturbation
        #-------------------------------------#
        # (we want all the perturbation to have exactly the same
        #  magnitude for easy comparison)
        _perturnb_size = len(_new_point_3d_dict) - 1
        _perturb_matrix = random_gen.multivariate_normal( np.zeros((3,)), np.eye(3), _perturnb_size)
        # Normalization
        _perturb_flattened_vec = _perturb_matrix.reshape((_perturnb_size*3,))
        _perturb_flattened_unit_vec = pnp_solver_GT.unit_vec(_perturb_flattened_vec)
        _norm = np.linalg.norm(_perturb_flattened_unit_vec, ord=2)
        print("_norm = %f" % _norm)
        # Let each point to have the standard deviation of perturb_radius_point (we have _perturnb_size points)
        # _perturb_normalized_matrix_T = (perturb_radius_point*_perturnb_size) * ( _perturb_flattened_unit_vec.reshape((3,_perturnb_size)) )  # <-- wrong
        _perturb_normalized_matrix_T = perturb_radius_point * ( _perturb_flattened_unit_vec.reshape((3,_perturnb_size)) ) # Note: the extreme case will be point to a single axis of a point, thus only multiply 1x of radius.
        print("_perturb_normalized_matrix_T = \n%s" % _perturb_normalized_matrix_T)
        #-------------------------------------#
        # Add and store the perturbation
        _perturb_id = 0
        print("")
        for _key in _new_point_3d_dict:
            if _key == fixed_pattern_key:
                np_pattern_perturbation_dict[_key] = np.zeros((3,1))
            else:
                np_pattern_perturbation_dict[_key] = _perturb_normalized_matrix_T[:, _perturb_id:(_perturb_id+1)]
                _new_point_3d_dict[_key] = list(np.array(_new_point_3d_dict[_key]).reshape((3,1)) + np_pattern_perturbation_dict[_key])
                _perturb_id += 1
            print("np_pattern_perturbation_dict[%s].T = %s" % (_key, np_pattern_perturbation_dict[_key].T))
            # print("_new_point_3d_dict[%s] = %s" % (_key, _new_point_3d_dict[_key]))
        print("")
        # print("np_pattern_perturbation_dict = \n%s" % np_pattern_perturbation_dict)
        # Update the selected pattern
        pnp_solver_GT.update_the_selected_golden_pattern(GT_golden_pattern_id, _new_point_3d_dict, _new_pattern_scale)
    else:
        # pnp_solver_GT.update_the_selected_golden_pattern(GT_golden_pattern_id, point_3d_dict_GT_list[GT_golden_pattern_id], pattern_scale_GT_list[GT_golden_pattern_id])
        pass
    # time.sleep(10)
    #------------------------------------#


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
    data_id_dict['np_pattern_perturbation_dict'] = np_pattern_perturbation_dict
    # Test inputs
    pnp_solver_GT.set_golden_pattern_id(GT_golden_pattern_id) # Use the number 0 pattern to generate the LM
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
    #
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
    if pass_count < pass_count_threshold: # Note: pass_count >= pass_count_threshold --> passed!!
        failed_sample_count += 1
        if fitting_error > 1.5:
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

    # Pattern Perturbation
    _result_idx_dict["np_pattern_perturbation_dict"] = data_list[_idx]['np_pattern_perturbation_dict']
    #
    result_list.append(_result_idx_dict)
    #----------------------------#

    # Min heaps (priority queue)
    #----------------------------#
    # Push the element: (priority, _idx)
    heapq.heappush(heap_neg_abs_depth_err, (-_result_idx_dict["abs_depth_err"], _idx ))
    heapq.heappush(heap_neg_abs_roll_err, (-_result_idx_dict["abs_roll_err"], _idx ))
    heapq.heappush(heap_neg_abs_pitch_err, (-_result_idx_dict["abs_pitch_err"], _idx ))
    heapq.heappush(heap_neg_abs_yaw_err, (-_result_idx_dict["abs_yaw_err"], _idx ))
    #----------------------------#



    # Image Display
    #====================================================#
    if not (is_showing_image or is_storing_case_image):
        continue

    # Get the file name of the image
    #--------------------------------------------#
    _file_name = data_list[_idx]['file_name']
    _image_file_name_str = '_'.join(_file_name.split('_')[0:9]) + '.png'
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
        # time.sleep(3.0)
        # continue
        _scale = 3
        _width = 320 * _scale
        _height = 240 * _scale
        _intensity = 200
        _img = np.ones( (_height, _width, 3), dtype=np.uint8) * _intensity
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
        # Landmarks
        _center_pixel = (np_point_image_dict[_k][0:2,0] * LM_2_image_scale).astype('int')
        _radius = 3
        _color = _color_BLUE # BGR
        # _color = _color_RED # BGR
        cv2.circle(_img_LM, _center_pixel, _radius, _color, -1)
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

# delta_time = time.time() - s_stamp
delta_time += time.time() - s_stamp # Add to the value of the previous workstate

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


# Save the latest workstate
#--------------------------------------------------#
workstate["sample_count"] = sample_count
workstate["delta_time"] = delta_time
workstate["data_list"] = data_list
workstate["result_list"] = result_list
workstate["heap_neg_abs_depth_err"] = heap_neg_abs_depth_err
workstate["heap_neg_abs_roll_err"] = heap_neg_abs_roll_err
workstate["heap_neg_abs_pitch_err"] = heap_neg_abs_pitch_err
workstate["heap_neg_abs_yaw_err"] = heap_neg_abs_yaw_err
#
joblib.dump(workstate, workstate_path_str)
#--------------------------------------------------#

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


def get_classified_result(result_list, class_name='distance', include_all=True, approval_func=None):
    '''
    '''
    class_dict = dict()
    for _idx in range(len(result_list)):
        _label = result_list[_idx]['class'][class_name]
        if not _label in class_dict:
            class_dict[_label] = list()
        # Decide wether to record or not
        if (approval_func is None) or approval_func( result_list[_idx] ):
            class_dict[_label].append(result_list[_idx])
    #
    if include_all:
        # class_dict["all"] = result_list
        class_dict["all"] = copy.deepcopy(result_list)
    return class_dict





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
distance_class_dict = get_classified_result(result_list, class_name='distance', approval_func=None)
# distance_class_dict = get_classified_result(result_list, class_name='distance', approval_func=approval_func_small_angle)
# distance_class_dict = get_classified_result(result_list, class_name='distance', approval_func=approval_func_large_angle)


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






# Min heaps (priority queue)
#----------------------------#
# # Push the element: (priority, _idx)
# heapq.heappush(heap_neg_abs_depth_err, (-_result_idx_dict["abs_depth_err"], _idx ))
# heapq.heappush(heap_neg_abs_roll_err, (-_result_idx_dict["abs_roll_err"], _idx ))
# heapq.heappush(heap_neg_abs_pitch_err, (-_result_idx_dict["abs_pitch_err"], _idx ))
# heapq.heappush(heap_neg_abs_yaw_err, (-_result_idx_dict["abs_yaw_err"], _idx ))
#----------------------------#
ratio_most_significant_part = 0.1 # 10%
num_top_element = int(len(result_list) * ratio_most_significant_part)
#
abs_depth_err_top_value_idx_list = list()
abs_roll_err_top_value_idx_list = list()
abs_pitch_err_top_value_idx_list = list()
abs_yaw_err_top_value_idx_list = list()
# Store the top indexes as lists
for _idx in range(num_top_element):
    abs_depth_err_top_value_idx_list.append( list(heapq.heappop(heap_neg_abs_depth_err)) )
    abs_roll_err_top_value_idx_list.append( list(heapq.heappop(heap_neg_abs_roll_err)) )
    abs_pitch_err_top_value_idx_list.append( list(heapq.heappop(heap_neg_abs_pitch_err)) )
    abs_yaw_err_top_value_idx_list.append( list(heapq.heappop(heap_neg_abs_yaw_err)) )
    # The value is orginally the negative one inorder to get the maximun value from the min heap
    abs_depth_err_top_value_idx_list[-1][0] *= -1
    abs_roll_err_top_value_idx_list[-1][0]  *= -1
    abs_pitch_err_top_value_idx_list[-1][0] *= -1
    abs_yaw_err_top_value_idx_list[-1][0]   *= -1




# Find the most fragile/sensible pattern point and the perturbation direction on each point
def get_most_fragile_point_and_perturbation_direction(point_3d_dict_GT, result_list, top_value_idx_list, k_top_direction=5):
    fragile_point_count_dict = dict()
    _key_list = list(point_3d_dict_GT.keys())
    # Initialize the count of each key to be zero
    for _key in _key_list:
        fragile_point_count_dict[_key] = 0
    #
    # point_norm_dict_list = list()
    perturb_vec_list = list()
    _value_sum = 0.0
    for _value, _idx in top_value_idx_list:
        _value_sum += _value
        _result_idx_dict = result_list[_idx]
        np_pattern_perturbation_dict = _result_idx_dict["np_pattern_perturbation_dict"]
        # Loop through the dict by key
        _point_norm_dict = dict()
        _norm_max = -1.0 # Note: every norm must be non negative, thus this value will at least be replaced by the first item in the dict
        _norm_max_key = None
        _perturb_list = list()
        for _key in np_pattern_perturbation_dict:
            _perturb_i = np_pattern_perturbation_dict[_key]
            _perturb_list.append(_perturb_i)
            #
            _norm_i = np.linalg.norm(_perturb_i)
            _point_norm_dict[_key] = _norm_i
            if _norm_i > _norm_max:
                _norm_max = _norm_i
                _norm_max_key = _key
        # print("Most fragile point: [%s], _norm_max = %f" % (_norm_max_key, _norm_max))
        fragile_point_count_dict[_norm_max_key] += 1
        # point_norm_dict_list.append(_point_norm_dict)
        #
        _perturb_vec = np.vstack(_perturb_list) # 3n x 1 vector for n LMs
        perturb_vec_list.append(_perturb_vec)
    #---------------------------#
    perturb_matrix = np.hstack(perturb_vec_list).T # m x 3n matrix, where m is the number of perturbation
    #---------------------------#

    # Get the maximum and the average of the value
    value_max = top_value_idx_list[0][0]
    top_value_mean = _value_sum / float(len(top_value_idx_list))

    # Get the sorted list of the most significant point
    #---------------------------#
    fragile_point_sorted_list = [(fragile_point_count_dict[_k], _k) for _k in fragile_point_count_dict]
    fragile_point_sorted_list.sort(reverse=True)

    # Calculate the top k significant direction of perturbation
    #---------------------------#
    perturb_matrix_u, perturb_matrix_s, perturb_matrix_vh = np .linalg.svd(perturb_matrix)
    # print("perturb_matrix_s = \n%s" % perturb_matrix_s)
    # print("Top 5 significant direction: perturb_matrix_vh[:5,:] = \n%s" % perturb_matrix_vh[:5,:])
    #
    top_perturbation_list = list()
    for _i in range(k_top_direction):
        __perturb_direction_dict = dict()
        _perturb_vec_i = perturb_matrix_vh[_i,:]
        _j = 0
        for _key in _key_list:
            __perturb_direction_dict[_key] = _perturb_vec_i[_j:(_j+3)].reshape((3,1))
            _j += 3
        top_perturbation_list.append(__perturb_direction_dict)
    #
    top_similarity_list = list(perturb_matrix_s[:k_top_direction])
    #
    result_dict = dict()
    result_dict["fragile_point_count_dict"] = fragile_point_count_dict
    result_dict["fragile_point_sorted_list"] = fragile_point_sorted_list
    result_dict["top_perturbation_list"] = top_perturbation_list
    result_dict["top_similarity_list"] = top_similarity_list
    result_dict["value_max"] = value_max
    result_dict["top_value_mean"] = top_value_mean
    return result_dict

def print_point_3D_key(np_point_3D_dict_list):
    # Logging
    #---------------------------#
    str_out = ""
    np.set_printoptions(suppress=True, precision=4)
    for _key in np_point_3D_dict_list[0]:
        str_out += "%s:%s" % (   _key, " "*(12-len(_key)) )
        for _idx, _np_p_3D_dict in enumerate(np_point_3D_dict_list):
            str_out += "| p[%d]=%s.T " % (_idx, str(_np_p_3D_dict[_key].T) )
        str_out += "\n"
    np.set_printoptions(suppress=False, precision=8)
    #
    print("-"*35)
    print(str_out)
    print("-"*35)
    return str_out
    #---------------------------#


#
perturb_result_depth = get_most_fragile_point_and_perturbation_direction(point_3d_dict_GT_list[0], result_list, abs_depth_err_top_value_idx_list)
perturb_result_roll = get_most_fragile_point_and_perturbation_direction(point_3d_dict_GT_list[0], result_list, abs_roll_err_top_value_idx_list)
perturb_result_pitch = get_most_fragile_point_and_perturbation_direction(point_3d_dict_GT_list[0], result_list, abs_pitch_err_top_value_idx_list)
perturb_result_yaw = get_most_fragile_point_and_perturbation_direction(point_3d_dict_GT_list[0], result_list, abs_yaw_err_top_value_idx_list)

print("-"*70)
print("Depth: fragile_point_count_dict = \n%s" % perturb_result_depth["fragile_point_count_dict"])
print("Roll: fragile_point_count_dict = \n%s" % perturb_result_roll["fragile_point_count_dict"])
print("Pitch: fragile_point_count_dict = \n%s" % perturb_result_pitch["fragile_point_count_dict"])
print("Yaw: fragile_point_count_dict = \n%s" % perturb_result_yaw["fragile_point_count_dict"])
print("-"*70)
print("Depth: fragile_point_sorted_list = \n%s" % perturb_result_depth["fragile_point_sorted_list"])
print("Roll: fragile_point_sorted_list = \n%s" % perturb_result_roll["fragile_point_sorted_list"])
print("Pitch: fragile_point_sorted_list = \n%s" % perturb_result_pitch["fragile_point_sorted_list"])
print("Yaw: fragile_point_sorted_list = \n%s" % perturb_result_yaw["fragile_point_sorted_list"])
print("-"*70)
print("Depth: top_perturbation_list")
print("top_similarity_list: %s" % perturb_result_depth["top_similarity_list"])
print_point_3D_key(perturb_result_depth["top_perturbation_list"])
print("Roll: top_perturbation_list")
print("top_similarity_list: %s" % perturb_result_roll["top_similarity_list"])
print_point_3D_key(perturb_result_roll["top_perturbation_list"])
print("Pitch: top_perturbation_list")
print("top_similarity_list: %s" % perturb_result_pitch["top_similarity_list"])
print_point_3D_key(perturb_result_pitch["top_perturbation_list"])
print("Yaw: top_perturbation_list")
print("top_similarity_list: %s" % perturb_result_yaw["top_similarity_list"])
print_point_3D_key(perturb_result_yaw["top_perturbation_list"])
print("-"*70)
print("Depth(cm):   value_max = %f, top_value_mean = %f" % (perturb_result_depth["value_max"]*100, perturb_result_depth["top_value_mean"]*100))
print("Roll(deg.):  value_max = %f, top_value_mean = %f" % (perturb_result_roll["value_max"], perturb_result_roll["top_value_mean"]))
print("Pitch(deg.): value_max = %f, top_value_mean = %f" % (perturb_result_pitch["value_max"], perturb_result_pitch["top_value_mean"]))
print("Yaw(deg.):   value_max = %f, top_value_mean = %f" % (perturb_result_yaw["value_max"], perturb_result_yaw["top_value_mean"]))
print("-"*70)



def write_perturbation_result_to_csv(perturb_result, csv_path, inspected_value_name="depth", unit="m", unit_scale=1.0):
    '''
    perturb_result = (fragile_point_count_dict, fragile_point_sorted_list, top_perturbation_list, value_max, top_value_mean)

    Head:
        inspected_value_name
        value_max
        top_value_mean
    Column: keys() of the point_3d_dict_GT, i.e. name of each point
    Row:
        - Top-fragile-point count
        - rank
        - perturb_direction #1 | similarity
            - x
            - y
            - z
            ...
        - perturb_direction #n | similarity
            - x
            - y
            - z
    '''
    fragile_point_count_dict = perturb_result["fragile_point_count_dict"]
    fragile_point_sorted_list = perturb_result["fragile_point_sorted_list"]
    top_perturbation_list = perturb_result["top_perturbation_list"]
    top_similarity_list = perturb_result["top_similarity_list"]
    value_max = perturb_result["value_max"]
    top_value_mean = perturb_result["top_value_mean"]
    #
    header_0 = ("inspected prediction", inspected_value_name)
    header_1 = (("value_max(%s)" % unit), (value_max*unit_scale) )
    header_2 = (("top_value_mean(%s)" % unit), (top_value_mean*unit_scale) )
    _key_list = list(fragile_point_count_dict.keys())
    header_3 = [""] + _key_list
    #
    # row_0 = ["Top-fragile-point count"] + list(fragile_point_count_dict.values())
    row_0 = ["Top-fragile-point count"] + [fragile_point_count_dict[_k] for _k in _key_list]
    #
    _rank_dict = dict()
    for _idx, _t in enumerate(fragile_point_sorted_list):
        _rank_dict[ _t[1] ] = _idx + 1
    # row_1 = ["Rank of fragility"] + list(_rank_dict.values())
    row_1 = ["Rank of fragility"] + [_rank_dict[_k] for _k in _key_list]


    with open(csv_path, mode='w') as _csv_f:
        _csv_w = csv.writer(_csv_f)
        #
        _csv_w.writerow(header_0)
        _csv_w.writerow(header_1)
        _csv_w.writerow(header_2)
        _csv_w.writerow(header_3)
        #
        _csv_w.writerow(row_0)
        _csv_w.writerow(row_1)
        #
        for _idx, __perturb_direction_dict in enumerate(top_perturbation_list):
            _count = _idx+1
            _csv_w.writerow([])
            _row_idx_head = [("perturb_direction #%d" % _count), "similarity", top_similarity_list[_idx]]
            _csv_w.writerow(_row_idx_head)

            # _row_x = ["x[%d]" % _count]
            # _row_y = ["y[%d]" % _count]
            # _row_z = ["z[%d]" % _count]
            _row_x = ["x"]
            _row_y = ["y"]
            _row_z = ["z"]
            for _key in _key_list:
                _perturbation_i = __perturb_direction_dict[_key].reshape((3,))
                # _row_x += ["%f" % _perturbation_i[0]]
                # _row_y += ["%f" % _perturbation_i[1]]
                # _row_z += ["%f" % _perturbation_i[2]]
                _row_x.append( _perturbation_i[0])
                _row_y.append( _perturbation_i[1])
                _row_z.append( _perturbation_i[2])
            _csv_w.writerow(_row_x)
            _csv_w.writerow(_row_y)
            _csv_w.writerow(_row_z)


        print("\n*** Wrote the results to the csv file:\n\t[%s]\n" % csv_path)




# Save the result of perturbation analysis
inspected_value_name = 'depth'
unit = "cm"
unit_scale = 100.0
perturb_result = perturb_result_depth
#
perturbation_result_csv_path = result_csv_dir_str + result_statistic_txt_file_prefix_str + data_file_str[:-4] + ( "_perturbation_to_%s" % (inspected_value_name) ) + '.csv'
write_perturbation_result_to_csv(perturb_result, perturbation_result_csv_path, inspected_value_name=inspected_value_name, unit=unit, unit_scale=unit_scale)

inspected_value_name = 'roll'
unit = "deg."
unit_scale = 1.0
perturb_result = perturb_result_roll
#
perturbation_result_csv_path = result_csv_dir_str + result_statistic_txt_file_prefix_str + data_file_str[:-4] + ( "_perturbation_to_%s" % (inspected_value_name) ) + '.csv'
write_perturbation_result_to_csv(perturb_result, perturbation_result_csv_path, inspected_value_name=inspected_value_name, unit=unit, unit_scale=unit_scale)

inspected_value_name = 'pitch'
unit = "deg."
unit_scale = 1.0
perturb_result = perturb_result_pitch
#
perturbation_result_csv_path = result_csv_dir_str + result_statistic_txt_file_prefix_str + data_file_str[:-4] + ( "_perturbation_to_%s" % (inspected_value_name) ) + '.csv'
write_perturbation_result_to_csv(perturb_result, perturbation_result_csv_path, inspected_value_name=inspected_value_name, unit=unit, unit_scale=unit_scale)

inspected_value_name = 'yaw'
unit = "deg."
unit_scale = 1.0
perturb_result = perturb_result_yaw
#
perturbation_result_csv_path = result_csv_dir_str + result_statistic_txt_file_prefix_str + data_file_str[:-4] + ( "_perturbation_to_%s" % (inspected_value_name) ) + '.csv'
write_perturbation_result_to_csv(perturb_result, perturbation_result_csv_path, inspected_value_name=inspected_value_name, unit=unit, unit_scale=unit_scale)



# Draw the perturbation on the image
def draw_perturbation_on_image(np_perturbation_dict,
                                point_3d_dict_list, pattern_scale_list, golden_pattern_id,
                                image_dir_str, image_file_name, result_image_name_prefix,
                                pnp_solver_in,
                                Euler_GT_list, t_GT_list):
    '''
    Draw the perturbation on the image
    '''
    # Control parameters
    #-----------------------#
    is_mirrored_image = True
    is_showing_image = True
    #
    fixed_pattern_key = "eye_c_51"
    perturb_radius_point = 0.02 # m, currently uniform in all direction
    #-----------------------#

    #
    _pnp_solver = copy.deepcopy(pnp_solver_in)
    _pnp_solver.verbose = False
    #

    # Euler_GT_list = [roll, pitch, yaw]
    np_R_GT = _pnp_solver.get_rotation_matrix_from_Euler(Euler_GT_list[0], Euler_GT_list[2], Euler_GT_list[1], is_degree=True)
    np_t_GT = np.array(t_GT_list).reshape((3,1))

    # Reprojections
    #---------------------------------------------------#
    # Get a copy of nominal pattern
    #----------------------------#
    _new_point_3d_dict = copy.deepcopy(point_3d_dict_list[golden_pattern_id])
    _new_pattern_scale = pattern_scale_list[golden_pattern_id]
    _key_list = list(_new_point_3d_dict.keys())
    #----------------------------#

    # Nominal, golden pattern
    #----------------------------#
    _pnp_solver.update_the_selected_golden_pattern(golden_pattern_id, _new_point_3d_dict, _new_pattern_scale)
    np_point_image_dict_nominal = _pnp_solver.perspective_projection_golden_landmarks(np_R_GT, np_t_GT, is_quantized=False, is_pretrans_points=False)
    #----------------------------#

    # Perturbed
    #----------------------------#
    # Add the perturbation to the golden pattern
    for _key in _key_list:
        if _key != fixed_pattern_key:
            # Note: The perturbation vector (21 x 1) of np_perturbation_dict is normalized to length 1.0 (the direction)
            #       To reconstruct the perturbed pattern, scale it by perturb_radius_point
            _new_point_3d_dict[_key] = list(np.array(_new_point_3d_dict[_key]).reshape((3,1)) + perturb_radius_point * np_perturbation_dict[_key])
    #
    _pnp_solver.update_the_selected_golden_pattern(golden_pattern_id, _new_point_3d_dict, _new_pattern_scale)
    np_point_image_dict_perturbed = _pnp_solver.perspective_projection_golden_landmarks(np_R_GT, np_t_GT, is_quantized=False, is_pretrans_points=False)
    #----------------------------#
    #---------------------------------------------------#


    # Get the file name of the image
    #--------------------------------------------#
    result_image_file_name = result_image_name_prefix + image_file_name
    print('Image file name: [%s]' % image_file_name)
    print('Result image file name: [%s]' % result_image_file_name)
    #
    _image_ori_path_str = image_dir_str + image_file_name
    _image_result_unflipped_path_str = image_result_unflipped_dir_str + result_image_file_name
    _image_result_path_str = image_result_dir_str + result_image_file_name
    #--------------------------------------------#

    # Load the original image
    #--------------------------------------------#
    _img = cv2.imread(_image_ori_path_str)
    if _img is None:
        print("!! Error occured while loading the image !!\n")
        return
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

    # Landmarks
    #----------------------------------#
    # [[u,v,1]].T
    for _k in _key_list:
        # Reprojections of the golden pattern onto the image using estimated pose
        _center_pixel = (np_point_image_dict_nominal[_k][0:2,0] * LM_2_image_scale).astype('int')
        _radius = 10
        _color = (127, 127, 0) # BGR
        cv2.circle(_img_LM, _center_pixel, _radius, _color, -1)
        # Reprojections of the golden pattern onto the image using estimated pose
        _center_pixel = (np_point_image_dict_perturbed[_k][0:2,0] * LM_2_image_scale).astype('int')
        _radius = 12
        _color = (0, 50, 200) # BGR
        cv2.circle(_img_LM, _center_pixel, _radius, _color, -1)
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
        cv2.imshow(result_image_file_name, _img_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    #----------------------------------#



#
image_file_name = "left_000_030_pitch_u_000_yaw_000_image.png"
Euler_GT_list = [0.0, 0.0, 0.0] # rpy
t_GT_list = [0.00156207, 0.00977673, 0.3] # [x,y,z] m
# image_file_name = "right_000_040_pitch_u_000_yaw_040_image.png"
# Euler_GT_list = [0.0, 0.0, 40.0] # rpy
# t_GT_list = [0.01556171, 0.010149, 0.4] # [x,y,z] m
GT_golden_pattern_id = 0
#
np_perturbation_dict = perturb_result_depth["top_perturbation_list"][0]
result_image_name_prefix = "depth_"
draw_perturbation_on_image(np_perturbation_dict,
                            point_3d_dict_GT_list, pattern_scale_GT_list, GT_golden_pattern_id,
                            image_dir_str, image_file_name, result_image_name_prefix,
                            pnp_solver_GT,
                            Euler_GT_list, t_GT_list)

np_perturbation_dict = perturb_result_roll["top_perturbation_list"][0]
result_image_name_prefix = "roll_"
draw_perturbation_on_image(np_perturbation_dict,
                            point_3d_dict_GT_list, pattern_scale_GT_list, GT_golden_pattern_id,
                            image_dir_str, image_file_name, result_image_name_prefix,
                            pnp_solver_GT,
                            Euler_GT_list, t_GT_list)

np_perturbation_dict = perturb_result_pitch["top_perturbation_list"][0]
result_image_name_prefix = "pitch_"
draw_perturbation_on_image(np_perturbation_dict,
                            point_3d_dict_GT_list, pattern_scale_GT_list, GT_golden_pattern_id,
                            image_dir_str, image_file_name, result_image_name_prefix,
                            pnp_solver_GT,
                            Euler_GT_list, t_GT_list)

np_perturbation_dict = perturb_result_yaw["top_perturbation_list"][0]
result_image_name_prefix = "yaw_"
draw_perturbation_on_image(np_perturbation_dict,
                            point_3d_dict_GT_list, pattern_scale_GT_list, GT_golden_pattern_id,
                            image_dir_str, image_file_name, result_image_name_prefix,
                            pnp_solver_GT,
                            Euler_GT_list, t_GT_list)
