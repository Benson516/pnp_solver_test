import numpy as np
import copy
import time
import csv
# import json
# import heapq
#
import cv2
#
import PNP_SOLVER_LIB as PNPS
# import joblib

# Integration tools
#----------------------------------------------#
def solving_center_point(p1,p2,p3,p4):
    '''
    p1   p2
       \/
       pc
       /\
    p4   p3
    '''
    # Transform to 2D arrays
    _n = np.array(p1).size
    _p1_shape = np.array(p1).shape
    _p1 = np.array(p1).reshape( (_n,1) )
    _p2 = np.array(p2).reshape( (_n,1) )
    _p3 = np.array(p3).reshape( (_n,1) )
    _p4 = np.array(p4).reshape( (_n,1) )
    #
    _d13 = _p3 - _p1
    _d24 = _p4 - _p2
    _A = np.hstack([_d13, _d24])
    _b = _p2 - _p1
    _uv = np.linalg.pinv(_A) @ _b
    _pc = _p1 + _uv[0,0] * _d13
    # reshape
    pc = _pc.reshape( _p1_shape )
    if type(p1) == type(list()):
        pc = list(pc)
    return pc

def convert_pixel_to_homo(pixel_xy, mirrored=False):
    '''
    pixel_xy: np array, shape=(2,)
    '''
    if mirrored:
        pixel_center_x = 320/2.0
        pixel_xy_mirrored = copy.deepcopy(pixel_xy)
        pixel_xy_mirrored[0] = -1.0 * (pixel_xy[0] - pixel_center_x) + pixel_center_x
        return np.array([pixel_xy_mirrored[0], pixel_xy_mirrored[1], 1.0]).reshape((3,1))
    else:
        return np.array([pixel_xy[0], pixel_xy[1], 1.0]).reshape((3,1))

def check_if_the_sample_passed(drpy_est_list, drpy_GT_list, drpy_error_bound_list):
    '''
    output
    [depth-passed, roll-passed, pitch-passed, yaw-passed], how many criterion passed
    '''
    drpy_pass_list = [ (np.abs( drpy_est_list[_i] - drpy_GT_list[_i] ) < drpy_error_bound_list[_i]) for _i in range(len(drpy_est_list))]
    pass_count = len([ _e for _e in drpy_pass_list if _e])
    return (drpy_pass_list, pass_count)
#----------------------------------------------#

# Golden patterns
#----------------------------------------------#
def get_golden_pattern(pattern_name="Alexander"):
    '''
    '''
    if pattern_name == 'Holly':
        # Holly
        # list: [x,y,z]
        point_3d_dict = dict()
        # Note: Each axis should exist at least 3 different values to make A_all full rank
        # Note: the Landmark definition in the pitcture in reversed
        point_3d_dict["eye_l_96"] = [ 0.028, 0.0, 0.0] # [ 0.035, 0.0, 0.0]
        point_3d_dict["eye_r_97"] = [-0.028, 0.0, 0.0] # [ 0.035, 0.0, 0.0]
        point_3d_dict["eye_c_51"] = [0.0, 0.0, 0.0]
        point_3d_dict["mouse_l_76"] = [ 0.025, 0.060, 0.0] # [ 0.025, 0.085, 0.0]
        point_3d_dict["mouse_r_82"] = [ -0.025, 0.060, 0.0] # [ -0.025, 0.085, 0.0]
        point_3d_dict["nose_t_54"] = [ 0.00, 0.039, -0.03] # [ 0.0, 0.0455, 0.03] # [ 0.0, 0.046, 0.03]
        point_3d_dict["chin_t_16"] = [0.0, 0.098, 0.0]
        point_3d_dict["brow_cl_35"] = [ 0.035, -0.0228, 0.0]
        point_3d_dict["brow_il_37"] = [ 0.0135, -0.017, 0.0]
        point_3d_dict["brow_ir_42"] = [ -0.0135, -0.017, 0.0]
        point_3d_dict["brow_cr_44"] = [ -0.035, -0.0228, 0.0]
        point_3d_dict["eye_lo_60"] = [0.046, 0.0, 0.0] # [ 0.035, 0.0, 0.0]
        point_3d_dict["eye_li_64"] = [0.01, 0.0, 0.0] # [ 0.035, 0.0, 0.0]
        point_3d_dict["eye_ro_72"] = [-0.046, 0.0, 0.0] # [ 0.035, 0.0, 0.0]
        point_3d_dict["eye_ri_68"] = [-0.01, 0.0, 0.0] # [ 0.035, 0.0, 0.0]
        #
        # point_3d_dict["face_c"] = solving_center_point(
        #                         point_3d_dict["eye_r_97"],
        #                         point_3d_dict["eye_l_96"],
        #                         point_3d_dict["mouse_l_76"],
        #                         point_3d_dict["mouse_r_82"]
        #                         )
        #
    else: # default
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
        point_3d_dict["brow_cl_35"] = [ 0.035, -0.0228, 0.0]
        point_3d_dict["brow_il_37"] = [ 0.0135, -0.017, 0.0]
        point_3d_dict["brow_ir_42"] = [ -0.0135, -0.017, 0.0]
        point_3d_dict["brow_cr_44"] = [ -0.035, -0.0228, 0.0]
        point_3d_dict["eye_lo_60"] = [0.05, 0.0, 0.0] # [ 0.035, 0.0, 0.0]
        point_3d_dict["eye_li_64"] = [0.014, 0.0, 0.0] # [ 0.035, 0.0, 0.0]
        point_3d_dict["eye_ro_72"] = [-0.05, 0.0, 0.0] # [ 0.035, 0.0, 0.0]
        point_3d_dict["eye_ri_68"] = [-0.014, 0.0, 0.0] # [ 0.035, 0.0, 0.0]
        #
        # point_3d_dict["face_c"] = solving_center_point(
        #                         point_3d_dict["eye_r_97"],
        #                         point_3d_dict["eye_l_96"],
        #                         point_3d_dict["mouse_l_76"],
        #                         point_3d_dict["mouse_r_82"]
        #                         )
        #
    return point_3d_dict
#----------------------------------------------#

# Ground truth classification
#----------------------------------------------#
def get_classification_parameters(drpy_class_format="drpy_expand"):
    '''
    '''
    is_classified_by_label = False
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
    #
    if is_classified_by_label:
        class_drpy_param_dict = None
    else:
        class_drpy_param_dict = dict()
        class_drpy_param_dict["labels"] = dict()
        class_drpy_param_dict["bins"] = dict()
        #
        class_drpy_param_dict["labels"]["depth"] = class_depth_label
        class_drpy_param_dict["labels"]["roll"]  = class_roll_label
        class_drpy_param_dict["labels"]["pitch"] = class_pitch_label
        class_drpy_param_dict["labels"]["yaw"]   = class_yaw_label
        #
        class_drpy_param_dict["bins"]["depth"] = class_depth_bins
        class_drpy_param_dict["bins"]["roll"]  = class_roll_bins
        class_drpy_param_dict["bins"]["pitch"] = class_pitch_bins
        class_drpy_param_dict["bins"]["yaw"]   = class_yaw_bins
    #
    return class_drpy_param_dict

def classify_drpy(class_drpy_param_dict, value, class_name="depth", is_classified_by_label=False):
    '''
    '''
    if is_classified_by_label or (class_drpy_param_dict is None): # Use its value as the class name
        return ("%d" % int(value) )
    else:
        labes = class_drpy_param_dict["labels"][class_name]
        bins = class_drpy_param_dict["bins"][class_name]
        return labes[ np.digitize(value, bins) ]
#----------------------------------------------#

# Landmark error distances
#----------------------------------------------#
def cal_LM_error_distances(np_point_dict_1, np_point_dict_2):
    '''
    Inputs
    - np_point_dict_1
    - np_point_dict_2
    Primary outputs
    - average_error
    - max_error
    - max_error_key
    Minor outputs
    - np_error_vec_dict
    - error_norm_dict
    '''
    # Calculate the pixel error of the LMs and the ground-truth projection of golden pattern
    np_error_vec_dict = dict()
    error_norm_dict = dict()
    LM_GT_error_total = 0.0
    max_error = 0.0
    max_error_key = ""
    for _k in np_point_dict_1:
        np_error_vec_dict[_k] = np_point_dict_1[_k] - np_point_dict_2[_k]
        error_norm_dict[_k] = np.linalg.norm( np_error_vec_dict[_k] )
        LM_GT_error_total += error_norm_dict[_k]
        if error_norm_dict[_k] > max_error:
            max_error = error_norm_dict[_k]
            max_error_key = _k
    average_error = LM_GT_error_total / len(np_point_dict_1)
    # Output
    output_dict = dict()
    output_dict["average_error"] = average_error
    output_dict["max_error"] = max_error
    output_dict["max_error_key"] = max_error_key
    output_dict["np_error_vec_dict"] = np_error_vec_dict
    output_dict["error_norm_dict"] = error_norm_dict
    return output_dict
#----------------------------------------------#

# Compare the result with GT and generate result_dict
#----------------------------------------------#
def compare_result_and_generate_result_dict(
                            pnp_solver,
                            data_list_idx,
                            np_R_est, np_t_est, rpy_est,
                            np_R_GT, np_t_GT_est, rpy_GT,
                            res_norm,
                            fail_count, pass_count, drpy_pass_list,
                            np_point_image_dict=None,
                            verbose=True):
    '''
    '''

    #----------------------------#
    roll_est, pitch_est, yaw_est = rpy_est
    roll_GT, pitch_GT, yaw_GT = rpy_GT
    #
    t3_est = np_t_est[2,0] # m
    distance_GT = np_t_GT_est[2,0] # m
    #----------------------------#

    # Reprojections
    np_homo_point_reproject_golden_dict = pnp_solver.perspective_projection_golden_landmarks(np_R_est, np_t_est, is_quantized=False, is_pretrans_points=False)
    # np_homo_point_reproject_golden_dict = pnp_solver.perspective_projection_golden_landmarks(np_R_ca_est, np_t_ca_est, is_quantized=False, is_pretrans_points=True)
    np_homo_point_GT_golden_dict = pnp_solver.perspective_projection_golden_landmarks(np_R_GT, np_t_GT_est, is_quantized=False, is_pretrans_points=False)

    #----------------------------#
    if np_point_image_dict is None:
        # Fake this
        np_point_image_dict = np_homo_point_GT_golden_dict
    #----------------------------#

    # Reprojection errors
    #-----------------------------------------------------------#
    # Calculate the pixel error of the LMs and the ground-truth projection of golden pattern
    LM_GT_ed_dict = cal_LM_error_distances(np_point_image_dict, np_homo_point_GT_golden_dict)
    LM_GT_error_average = LM_GT_ed_dict["average_error"]
    LM_GT_error_max = LM_GT_ed_dict["max_error"]
    LM_GT_error_max_key = LM_GT_ed_dict["max_error_key"]
    print("(LM_GT_error_average, LM_GT_error_max, LM_GT_error_max_key) = (%f, %f, %s)" % (LM_GT_error_average, LM_GT_error_max, LM_GT_error_max_key))

    # Calculate the pixel error of the LMs and the reprojections of golden pattern
    predict_LM_ed_dict = cal_LM_error_distances(np_homo_point_reproject_golden_dict, np_point_image_dict)
    predict_LM_error_average = predict_LM_ed_dict["average_error"]
    predict_LM_error_max = predict_LM_ed_dict["max_error"]
    predict_LM_error_max_key = predict_LM_ed_dict["max_error_key"]
    print("(predict_LM_error_average, predict_LM_error_max, predict_LM_error_max_key) = (%f, %f, %s)" % (predict_LM_error_average, predict_LM_error_max, predict_LM_error_max_key))

    # Calculate the pixel error of the LMs and the reprojections of golden pattern
    predict_GT_ed_dict = cal_LM_error_distances(np_homo_point_reproject_golden_dict, np_homo_point_GT_golden_dict)
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
                str(np_homo_point_reproject_golden_dict[_k].T),
                str((np_homo_point_reproject_golden_dict[_k]-np_point_image_dict[_k]).T)
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
    print("distance = %f cm" % data_list_idx['distance'])
    print("(roll_GT, yaw_GT, pitch_GT) \t\t= %s" % str(  [ data_list_idx['roll'], data_list_idx['yaw'], data_list_idx['pitch'] ]) )
    if verbose:
        print("np_Gamma_GT = \n%s" % str( np_R_GT / np_t_GT_est[2,0] ))
    print("np_R_GT = \n%s" % str(np_R_GT))
    print("np_t_GT_est = \n%s" % str(np_t_GT_est))
    print("-"*30 + " The End " + "-"*30)
    print()
    #----------------------------#



    # Store the error for statistic
    #----------------------------#
    _result_idx_dict = dict()
    _result_idx_dict['idx'] = data_list_idx['idx']
    _result_idx_dict["file_name"] = data_list_idx['file_name']
    _result_idx_dict["drpy"] = (distance_GT, roll_GT, pitch_GT, yaw_GT)
    _result_idx_dict["class"] = data_list_idx['class']
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
    return (_result_idx_dict, np_homo_point_GT_golden_dict, np_homo_point_reproject_golden_dict)
#----------------------------------------------#


# Visualization
#----------------------------------------------#
def plot_LMs_and_axies(
            image_file_name_str,
            image_dir_str,
            image_result_unflipped_dir_str, image_result_dir_str,
            pnp_solver,
            result_idx_dict,
            np_homo_point_LM_dict,
            np_homo_point_GT_golden_dict,
            np_homo_point_reproject_golden_dict,
            np_homo_point_LM_frontal_view_dict=None,
            is_ploting_LMs=True,
            is_mirrored_image=False,
            LM_img_width=320,
            is_showing_image=True):
    '''
    LM_img_width: The width of the original image scale for representing the landmark coordinate

    result_idx_dict:
    - np_R_GT
    - np_t_GT_est
    #
    - np_R_est
    - np_t_est
    #
    - roll_GT
    - roll_est
    - roll_err
    #
    - pitch_GT
    - pitch_est
    - pitch_err
    #
    - yaw_GT
    - yaw_est
    - yaw_err

    '''
    # Get the file name of the image
    #--------------------------------------------#
    print('image file name: [%s]' % image_file_name_str)
    _image_ori_path_str = image_dir_str + image_file_name_str
    _image_result_unflipped_path_str = image_result_unflipped_dir_str + image_file_name_str
    _image_result_path_str = image_result_dir_str + image_file_name_str
    #--------------------------------------------#

    # Load the original image
    #--------------------------------------------#
    _img = cv2.imread(_image_ori_path_str)
    if _img is None:
        print("!! Error occured while loading the image !!\n")
        # time.sleep(3.0)
        # continue
        # Fake one
        #---------------------#
        _scale = 3
        _width = 320 * _scale
        _height = 240 * _scale
        _intensity = 200
        _img = np.ones( (_height, _width, 3), dtype=np.uint8) * _intensity
        #---------------------#
    # Resize the image
    #----------------------#
    _desired_img_display_width = 960 # 2048
    if (_img.shape[1] != _desired_img_display_width):
        _aspext_ratio_img_in = float(_img.shape[1]) / float(_img.shape[0]) # width / height
        _desired_dim = ( int(_desired_img_display_width), int(_desired_img_display_width / _aspext_ratio_img_in) )
        _img = cv2.resize(_img, _desired_dim, interpolation=cv2.INTER_AREA) # Resize 1.5 times
    #----------------------#
    _img_shape = _img.shape
    print("_img.shape = %s" % str(_img_shape))
    LM_2_image_scale = _img_shape[1] / LM_img_width # Note: 320x240 is the image scale used by LM
    #--------------------------------------------#


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
    # Ground truth (R,t)
    np_R_GT = result_idx_dict["np_R_GT"]
    np_t_GT_est = result_idx_dict["np_t_GT_est"]
    # Estimated (R,t)
    np_R_est = result_idx_dict["np_R_est"]
    np_t_est = result_idx_dict["np_t_est"]

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
    _key_list = list( np_homo_point_reproject_golden_dict.keys() )
    for _k in _key_list:
        #
        if is_ploting_LMs:
            # Landmarks
            _center_pixel = (np_homo_point_LM_dict[_k][0:2,0] * LM_2_image_scale).astype('int')
            _radius = 3
            _color = _color_BLUE # BGR
            # _color = _color_RED # BGR
            cv2.circle(_img_LM, _center_pixel, _radius, _color, -1)
        #
        # Reprojections of golden pattern onto image using grund truth pose
        _center_pixel = (np_homo_point_GT_golden_dict[_k][0:2,0] * LM_2_image_scale).astype('int')
        _radius = 2
        _color = (127, 127, 0) # BGR
        # _color = _color_RED # BGR
        cv2.circle(_img_LM, _center_pixel, _radius, _color, -1)
        # Reprojections of the golden pattern onto the image using estimated pose
        _center_pixel = (np_homo_point_reproject_golden_dict[_k][0:2,0] * LM_2_image_scale).astype('int')
        _radius = 1
        _color = _color_RED # BGR
        # _color = _color_BLUE # BGR
        cv2.circle(_img_LM, _center_pixel, _radius, _color, -1)
        if np_homo_point_LM_frontal_view_dict is not None:
            # Frontal view
            _center_pixel = (np_homo_point_LM_frontal_view_dict[_k][0:2,0] * LM_2_image_scale).astype('int')
            _radius = 5
            _color = (0, 180, 0) # BGR
            # _color = _color_BLUE # BGR
            cv2.circle(_img_LM, _center_pixel, _radius, _color, -1)
    #----------------------------------#

    # Text
    #----------------------------------#
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL # = 5
    fontScale = 1.5
    thickness = 2
    # _text = "Hello world! 012345"
    _est_text = "drpy_est = (%.2f, %.2f, %.2f, %.2f)" % (result_idx_dict["t3_est"]*100.0,
                                            result_idx_dict["roll_est"],
                                            result_idx_dict["pitch_est"],
                                            result_idx_dict["yaw_est"])
    _GT_text = "drpy_GT = (%.2f, %.2f, %.2f, %.2f)" % (result_idx_dict["distance_GT"]*100.0,
                                            result_idx_dict["roll_GT"],
                                            result_idx_dict["pitch_GT"],
                                            result_idx_dict["yaw_GT"])
    _err_text = "drpy_err = (%.2f, %.2f, %.2f, %.2f)" % (result_idx_dict["depth_err"]*100.0,
                                            result_idx_dict["roll_err"],
                                            result_idx_dict["pitch_err"],
                                            result_idx_dict["yaw_err"])
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
        cv2.imshow(image_file_name_str, _img_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    #----------------------------------#
#----------------------------------------------#


# File writing
#----------------------------------------------#
def write_result_to_csv(result_list, csv_path):
    '''
    '''
    with open(csv_path, mode='w') as _csv_f:
        fieldnames = result_list[0].keys()
        # fieldnames = ["idx", "file_name", "drpy", "t3_est", "roll_est", "pitch_est", "yaw_est", "res_norm", "distance_GT", "roll_GT", "pitch_GT", "yaw_GT"]
        # fieldnames = ["idx", "file_name", "drpy", "t3_est", "distance_GT", "roll_est", "roll_GT", "pitch_est", "pitch_GT", "yaw_est", "yaw_GT", "res_norm"]
        # fieldnames = ["idx", "file_name", "drpy", "distance_GT", "t3_est", "depth_err", "roll_GT", "roll_est", "roll_err", "pitch_GT", "pitch_est", "pitch_err", "yaw_GT", "yaw_est", "yaw_err", "res_norm"]
        # fieldnames = ["idx", "file_name", "drpy", "distance_GT", "t3_est", "depth_err", "roll_GT", "roll_est", "roll_err", "pitch_GT", "pitch_est", "pitch_err", "yaw_GT", "yaw_est", "yaw_err", "res_norm"]
        _csv_w = csv.DictWriter(_csv_f, fieldnames=fieldnames, extrasaction='ignore')

        _csv_w.writeheader()
        _csv_w.writerows(result_list)
        # for _e_dict in result_list:
        #     _csv_w.writerow(_e_dict)
        print("\n*** Wrote the results to the csv file:\n\t[%s]\n" % csv_path)

def _class_order_func(e):
    '''
    For sorting the key of class
    Note: "all" class is placed specifically at the top.
    '''
    # return ( (-1) if (e == "all") else int(e))
    return ( float("-inf") if (e == "all") else float(e))

def write_statistic_to_txt(class_statistic_dict, statistic_txt_path, class_name="distance", statistic_data_name="depth"):
    '''
    '''
    #---------------------#
    _statistic_str_out = '\nStatistic of [%s] for each [%s] class:\n' % (statistic_data_name, class_name)
    # def value_of_string(e):
    #   return int(e)
    _label_list = list(class_statistic_dict.keys())
    # _label_list.sort(key=int) # Using the integer value of string to sort the list
    _label_list.sort(key=_class_order_func) # Using the integer value of string to sort the list
    for _label in _label_list:
        _s_data_dict = class_statistic_dict[_label] # New version, changed to dict
        _statistic_str_out += "[%s]: " % _label
        #
        for _idx, _matric_l in enumerate(_s_data_dict):
            if _idx > 0:
                _statistic_str_out += " | "
            # Write data
            _statistic_str_out += "%s=%f" % (_matric_l, _s_data_dict[_matric_l]) # Note: _matric_l already include the unit and scale
            # else: # Apply init_scale and print unit
            #     _statistic_str_out += "%s=%f %s" % (_matric_l, _s_data_dict[_matric_l]*unit_scale, unit)
        # _statistic_str_out += "m_ratio=%f" % (_s_data_dict[0])
        # _statistic_str_out += " | mean=%f %s" % (_s_data_dict[1]*unit_scale, unit)
        # _statistic_str_out += " | stddev=%f %s" % (_s_data_dict[2]*unit_scale, unit)
        # _statistic_str_out += " | MAE_2_GT=%f %s" % (_s_data_dict[3]*unit_scale, unit)
        # _statistic_str_out += " | MAE_2_mean=%f %s" % (_s_data_dict[4]*unit_scale, unit)
        _statistic_str_out += "\n"
    #
    print(_statistic_str_out)
    #
    with open(statistic_txt_path, "w") as _f:
        _f.write(_statistic_str_out)
        print("\n*** Wrote the statistic to the txt file:\n\t[%s]\n" % statistic_txt_path)
    #---------------------#
    return _statistic_str_out

def write_statistic_to_csv(class_statistic_dict, statistic_csv_path, class_name="distance", statistic_data_name="depth", is_horizontal=True):
    '''
    Horizontal:
            class -->
    matrics
    |
    |/

    Vertical:
            matrics -->
    class
    |
    |/
    '''
    # Preorganize data
    #--------------------------------#
    # Class labels
    _label_list = list(class_statistic_dict.keys())
    # _label_list.sort(key=int) # Using the integer value of string to sort the list
    _label_list.sort(key=_class_order_func) # Using the integer value of string to sort the list
    # Matric labels
    _matric_list = list(class_statistic_dict[ _label_list[0] ].keys())
    #--------------------------------#

    if is_horizontal:
        # Horizontal class
        #--------------------------------#
        fieldnames = ['_'] + _label_list
        row_dict_list = list()
        for _matric_l in _matric_list: # Key
            _row_dict = dict()
            # 1st column
            _row_dict['_'] = _matric_l
            # 2nd column and so on
            for _label in _label_list: # Key
                _s_data_dict = class_statistic_dict[_label] # _s_data_dict is a dict of statistic values
                _row_dict[_label] = _s_data_dict[_matric_l] # _matric_l matric
            #
            row_dict_list.append(_row_dict)
        #--------------------------------#
    else:
        # Vertical class
        #--------------------------------#
        fieldnames = ['_'] + _matric_list
        row_dict_list = list()
        for _label in _label_list: # Key
            _row_dict = dict()
            # 1st column
            _row_dict['_'] = _label
            # 2nd column and so on
            for _matric_l in _matric_list: # Index
                _s_data_dict = class_statistic_dict[_label] # _s_data_dict is a dict of statistic values
                _row_dict[ _matric_l ] = _s_data_dict[_matric_l] # _matric_l matric
            #
            row_dict_list.append(_row_dict)
        #--------------------------------#

    with open(statistic_csv_path, mode='w') as _csv_f:
        _csv_w = csv.DictWriter(_csv_f, fieldnames=fieldnames, extrasaction='ignore')
        #
        _csv_w.writeheader()
        _csv_w.writerows(row_dict_list)
        # for _e_dict in row_dict_list:
        #     _csv_w.writerow(_e_dict)
        _hv_adv_str = 'horizontally' if is_horizontal else 'vertically'
        print("\n*** Wrote the statistic results [%s] to the csv file:\n\t[%s]\n" % ( _hv_adv_str, statistic_csv_path))
    return row_dict_list

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

    with open(csv_path, mode='w') as _csv_f:
        _csv_w = csv.DictWriter(_csv_f, fieldnames=fieldnames, extrasaction='ignore')
        #
        _csv_w.writeheader()
        _csv_w.writerows(row_dict_list)
        # for _e_dict in row_dict_list:
        #     _csv_w.writerow(_e_dict)
        print("\n*** Wrote the drpy statistic results to the csv file:\n\t[%s]\n" % ( csv_path))
#----------------------------------------------#

# Statistic analysis
#----------------------------------------------#
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
#----------------------------------------------#

# Possible approval functions for get_classified_result()
#----------------------------------------------#
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
#----------------------------------------------#

# drpy (depth, roll, pitch, yaw) analysis
#----------------------------------------------#
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
#----------------------------------------------#


# Headpose estimation data analysis
#----------------------------------------------#
def data_analysis_and_saving(result_list, result_csv_dir_str, result_csv_file_prefix_str, result_statistic_txt_file_prefix_str, data_file_str, is_statistic_csv_horizontal=True):
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








    # Get overall statistic data
    # This part is simply for logging
    #-----------------------------------#
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
    #-----------------------------------#


    # Write to result CSV file
    csv_path = result_csv_dir_str + result_csv_file_prefix_str + data_file_str[:-4] + '.csv'
    write_result_to_csv(result_list, csv_path)




    # Get statistic result in each class, filter (by approval func) if required
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
    write_statistic_to_txt(class_statistic_dict, statistic_txt_path, class_name=class_name, statistic_data_name=statistic_data_name)
    write_statistic_to_csv(class_statistic_dict, statistic_csv_path, class_name=class_name, statistic_data_name=statistic_data_name, is_horizontal=is_statistic_csv_horizontal)
    #
    statistic_data_name = "roll" # Just the name as the info. to the reader
    class_statistic_dict = dist_2_roll_statistic_dict
    statistic_txt_path = result_csv_dir_str + result_statistic_txt_file_prefix_str + data_file_str[:-4] + ( "_%s_to_%s" % (class_name, statistic_data_name) ) + '.txt'
    statistic_csv_path = result_csv_dir_str + result_statistic_txt_file_prefix_str + data_file_str[:-4] + ( "_%s_to_%s" % (class_name, statistic_data_name) ) + '.csv'
    write_statistic_to_txt(class_statistic_dict, statistic_txt_path, class_name=class_name, statistic_data_name=statistic_data_name)
    write_statistic_to_csv(class_statistic_dict, statistic_csv_path, class_name=class_name, statistic_data_name=statistic_data_name, is_horizontal=is_statistic_csv_horizontal)
    #
    statistic_data_name = "pitch" # Just the name as the info. to the reader
    class_statistic_dict = dist_2_pitch_statistic_dict
    statistic_txt_path = result_csv_dir_str + result_statistic_txt_file_prefix_str + data_file_str[:-4] + ( "_%s_to_%s" % (class_name, statistic_data_name) ) + '.txt'
    statistic_csv_path = result_csv_dir_str + result_statistic_txt_file_prefix_str + data_file_str[:-4] + ( "_%s_to_%s" % (class_name, statistic_data_name) ) + '.csv'
    write_statistic_to_txt(class_statistic_dict, statistic_txt_path, class_name=class_name, statistic_data_name=statistic_data_name)
    write_statistic_to_csv(class_statistic_dict, statistic_csv_path, class_name=class_name, statistic_data_name=statistic_data_name, is_horizontal=is_statistic_csv_horizontal)
    #
    statistic_data_name = "yaw" # Just the name as the info. to the reader
    class_statistic_dict = dist_2_yaw_statistic_dict
    statistic_txt_path = result_csv_dir_str + result_statistic_txt_file_prefix_str + data_file_str[:-4] + ( "_%s_to_%s" % (class_name, statistic_data_name) ) + '.txt'
    statistic_csv_path = result_csv_dir_str + result_statistic_txt_file_prefix_str + data_file_str[:-4] + ( "_%s_to_%s" % (class_name, statistic_data_name) ) + '.csv'
    write_statistic_to_txt(class_statistic_dict, statistic_txt_path, class_name=class_name, statistic_data_name=statistic_data_name)
    write_statistic_to_csv(class_statistic_dict, statistic_csv_path, class_name=class_name, statistic_data_name=statistic_data_name, is_horizontal=is_statistic_csv_horizontal)











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

#----------------------------------------------#
