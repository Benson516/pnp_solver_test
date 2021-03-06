import numpy as np
import copy
import time
import csv
import os
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

# Make directories for storing the results if they do not exist.
#------------------------------------------------------------------#
os.makedirs(image_result_unflipped_dir_str, mode=0o777, exist_ok=True)
os.makedirs(image_result_dir_str, mode=0o777, exist_ok=True)
os.makedirs(result_csv_dir_str, mode=0o777, exist_ok=True)
os.makedirs(workstate_dir_str, mode=0o777, exist_ok=True)
#------------------------------------------------------------------#

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



# Data analysis and saving
#-----------------------------------#
TTBX.data_analysis_and_saving(result_list, result_csv_dir_str, result_csv_file_prefix_str, result_statistic_txt_file_prefix_str, data_file_str, is_statistic_csv_horizontal=is_statistic_csv_horizontal)
#-----------------------------------#








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
