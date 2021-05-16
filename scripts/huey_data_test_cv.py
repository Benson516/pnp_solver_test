import numpy as np
import copy
import time
#
import cv2
#
import PNP_SOLVER_LIB as PNPS

#---------------------------#
# Landmark (LM) dataset
data_dir_str = '/home/benson516/test_PnP_solver/dataset/Huey_face_landmarks_pose/'
data_file_str = 'test_Alexander.txt'
# data_file_str = 'test_Alexey.txt'
#---------------------------#
# Original image
image_dir_str = '/home/benson516/test_PnP_solver/dataset/images/alexander_SZ/'
# The image used for analysis
image_result_unflipped_dir_str = '/home/benson516/test_PnP_solver/dataset/images/alexander_SZ_result_unflipped/'
# The same as the original image
image_result_dir_str = '/home/benson516/test_PnP_solver/dataset/images/alexander_SZ_result/'
#---------------------------#



# Behavior of this program
#---------------------------#
# Data
is_limiting_line_count = True
# is_limiting_line_count = False
# DATA_START_ID = 0
# DATA_START_ID = 658
DATA_START_ID = 379 # (0, 0, 0)
DATA_COUNT =  3
#
verbose = True
# verbose = False
# Image display
is_showing_image = True
#---------------------------#

# Parameters of the data
#---------------------------#
is_h_mirrored_image = True
pattern_scale = 1.0 # 0.85 # Multiply onto the golden pattern
#---------------------------#


# Loading Data
#----------------------------------------------------------#
data_path_str = data_dir_str + data_file_str
#
data_str_list_list = list()
data_name_split_list_list = list()
with open(data_path_str, 'r') as _f:
    # Read and print the entire file line by line
    _line = _f.readline()
    _count = 0
    while (_line != '') and ((not is_limiting_line_count) or (_count < (DATA_START_ID+DATA_COUNT) ) ):  # The EOF char is an empty string
        if _count >= DATA_START_ID:
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
        _count += 1
#
print(data_str_list_list[0][0:5]) # [data_idx][column in line of file]
print(data_name_split_list_list[0][9:12]) # [data_idx][column in file name split]
#----------------------------------------------------------#

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
# f_camera = 188.55 # 175.0
f_camera = 225.68717584155982
#
fx_camera = f_camera
# fx_camera = (-f_camera) if is_h_mirrored_image else f_camera # Note: mirrored image LM features
fy_camera = f_camera
xo_camera = 320/2.0
yo_camera = 240/2.0
# np_K_camera_GT = np.array([[fx_camera, 0.0, xo_camera], [0.0, fy_camera, yo_camera], [0.0, 0.0, 1.0]]) # Grund truth
np_K_camera_est = np.array([[fx_camera, 0.0, xo_camera], [0.0, fy_camera, yo_camera], [0.0, 0.0, 1.0]]) # Estimated
# print("np_K_camera_GT = \n%s" % str(np_K_camera_GT))
print("np_K_camera_est = \n%s" % str(np_K_camera_est))

# 3D landmark point - local coordinate
#----------------------------------------#
# list: [x,y,z]
point_3d_dict = dict()
# Note: Each axis should exist at least 3 different values to make A_all full rank
# Note: the Landmark definition in the pitcture in reversed
point_3d_dict["eye_l_96"] = [ 0.032, 0.0, 0.0] # [ 0.035, 0.0, 0.0]
point_3d_dict["eye_r_97"] = [-0.032, 0.0, 0.0] # [ 0.035, 0.0, 0.0]
point_3d_dict["mouse_l_76"] = [ 0.027, 0.070, 0.0] # [ 0.025, 0.085, 0.0]
point_3d_dict["mouse_r_82"] = [ -0.027, 0.070, 0.0] # [ -0.025, 0.085, 0.0]
point_3d_dict["nose_t_54"] = [ -0.005, 0.0455, -0.03] # [ 0.0, 0.0455, 0.03] # [ 0.0, 0.046, 0.03]
point_3d_dict["eye_c_51"] = [0.0, 0.0, 0.0]
point_3d_dict["chin_t_16"] = [0.0, 0.12, 0.0]
# point_3d_dict["face_c"] = [ 0.0, 0.035, 0.0]
# point_3d_dict["chin"] = [ 0.0, 0.08, -0.005]
# point_3d_dict["far"] = [ 0.0, 0.0, -0.5]

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
pnp_solver = PNPS.PNP_SOLVER_A2_M3(np_K_camera_est, point_3d_dict, pattern_scale=pattern_scale, verbose=verbose)


def convert_pixel_to_homo(pixel_xy, mirrored=True):
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

# Loop through data
#-------------------------------------------------------#
distance_error_list = list()
distance_ratio_list = list()
for _idx in range(len(data_list)):
    print("\n-------------- (idx = %d)--------------\n" % _idx)
    print('file file_name: [%s]' % data_list[_idx]['file_name'])


    LM_pixel_data_matrix = data_list[_idx]['LM_pixel'] # [LM_id] --> [x,y]
    np_point_image_dict = dict()
    # [x,y,1].T, shape: (3,1)
    np_point_image_dict["eye_l_96"] = convert_pixel_to_homo(LM_pixel_data_matrix[96])
    np_point_image_dict["eye_r_97"] = convert_pixel_to_homo(LM_pixel_data_matrix[97])
    np_point_image_dict["mouse_l_76"] = convert_pixel_to_homo(LM_pixel_data_matrix[76])
    np_point_image_dict["mouse_r_82"] = convert_pixel_to_homo(LM_pixel_data_matrix[82])
    np_point_image_dict["nose_t_54"] = convert_pixel_to_homo(LM_pixel_data_matrix[54])
    np_point_image_dict["eye_c_51"] = convert_pixel_to_homo(LM_pixel_data_matrix[51])
    np_point_image_dict["chin_t_16"] = convert_pixel_to_homo(LM_pixel_data_matrix[16])
    #
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
    np_R_ca_est = pnp_solver.np_R_c_a_est
    np_t_ca_est = pnp_solver.np_t_c_a_est
    # np_R_ca_est = copy.deepcopy(pnp_solver.np_R_c_a_est)
    # np_t_ca_est = copy.deepcopy(pnp_solver.np_t_c_a_est)

    # Compare result
    #-----------------------------#
    print("Result from the solver:\n")

    # Grund truth (R,t)
    np_R_GT = pnp_solver.get_rotation_matrix_from_Euler( data_list[_idx]['roll'], data_list[_idx]['yaw'], data_list[_idx]['pitch'], is_degree=True )
    # _det = np.linalg.det(np_R_GT)
    # print("np_R_GT = \n%s" % str(np_R_GT))
    # print("_det = %f" % _det)
    distance_GT = data_list[_idx]['distance'] *0.01 # cm --> m
    np_t_GT_est = (np_t_est/t3_est) * distance_GT


    # Reprojections
    # np_point_image_dict_reproject = pnp_solver.perspective_projection_golden_landmarks(np_R_est, np_t_est, is_quantized=False, is_pretrans_points=False)
    np_point_image_dict_reproject = pnp_solver.perspective_projection_golden_landmarks(np_R_ca_est, np_t_ca_est, is_quantized=False, is_pretrans_points=True)
    np_point_image_dict_reproject_GT_ori_golden_patern = pnp_solver.perspective_projection_golden_landmarks(np_R_GT, np_t_GT_est, is_quantized=False, is_pretrans_points=False)
    #
    print("2D points on image (re-projection):")
    # print("2D points on image (is_h_mirrored_image=%s):" % str(is_h_mirrored_image))
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
    print("(roll_est, yaw_est, pitch_est) \t\t= %s" % str( np.rad2deg( (roll_est, yaw_est, pitch_est) ) ) )
    print("np_R_est = \n%s" % str(np_R_est))
    print("np_t_est = \n%s" % str(np_t_est))
    # Grund truth
    print("-"*28 + " Grund Truth " + "-"*28)
    print("distance = %f cm" % data_list[_idx]['distance'])
    print("(roll_GT, yaw_GT, pitch_GT) \t\t= %s" % str(  [ data_list[_idx]['roll'], data_list[_idx]['yaw'], data_list[_idx]['pitch'] ]) )
    print("np_R_GT = \n%s" % str(np_R_GT))
    print("np_t_GT_est = \n%s" % str(np_t_GT_est))
    print("-"*30 + " The End " + "-"*30)
    print()
    #----------------------------#

    # Store the error for statistic
    #----------------------------#
    distance_ratio_list.append( (t3_est*100.0 / data_list[_idx]['distance']))
    distance_error_list.append( (t3_est*100.0 - data_list[_idx]['distance']))
    #----------------------------#


    # Image Display
    #====================================================#
    if not is_showing_image:
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
        time.sleep(3.0)
        continue
    _img_shape = _img.shape
    print("_img.shape = %s" % str(_img_shape))
    LM_2_image_scale = _img_shape[1] / 320.0

    # Flip the image if needed
    #----------------------------------#
    if is_h_mirrored_image:
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
    uv_o, dir_x, dir_y, dir_z = pnp_solver.perspective_projection_obj_axis(np_R_GT, np_t_GT_est) # Note: use the estimated t since we don't have the grund truth.
    print("(uv_o, dir_x, dir_y, dir_z) = %s" % str((uv_o, dir_x, dir_y, dir_z)))
    vector_scale = 0.2
    _pixel_uv_o = (LM_2_image_scale*uv_o).astype('int')
    _pixel_uv_x1 = (LM_2_image_scale*(uv_o+dir_x*vector_scale)).astype('int')
    _pixel_uv_y1 = (LM_2_image_scale*(uv_o+dir_y*vector_scale)).astype('int')
    _pixel_uv_z1 = (LM_2_image_scale*(uv_o+dir_z*vector_scale)).astype('int')
    # Draw lines
    _line_width = 2
    cv2.line(_img_LM, _pixel_uv_o, _pixel_uv_x1, (0,0,127), _line_width)
    cv2.line(_img_LM, _pixel_uv_o, _pixel_uv_y1, (0,127,0), _line_width)
    cv2.line(_img_LM, _pixel_uv_o, _pixel_uv_z1, (127,0,0), _line_width)

    # Estimated axes
    uv_o, dir_x, dir_y, dir_z = pnp_solver.perspective_projection_obj_axis(np_R_est, np_t_est)
    print("(uv_o, dir_x, dir_y, dir_z) = %s" % str((uv_o, dir_x, dir_y, dir_z)))
    vector_scale = 0.2
    _pixel_uv_o = (LM_2_image_scale*uv_o).astype('int')
    _pixel_uv_x1 = (LM_2_image_scale*(uv_o+dir_x*vector_scale)).astype('int')
    _pixel_uv_y1 = (LM_2_image_scale*(uv_o+dir_y*vector_scale)).astype('int')
    _pixel_uv_z1 = (LM_2_image_scale*(uv_o+dir_z*vector_scale)).astype('int')
    # Draw lines
    _line_width = 1
    cv2.line(_img_LM, _pixel_uv_o, _pixel_uv_x1, _color_RED, _line_width)
    cv2.line(_img_LM, _pixel_uv_o, _pixel_uv_y1, _color_GREEN, _line_width)
    cv2.line(_img_LM, _pixel_uv_o, _pixel_uv_z1, _color_BLUE, _line_width)
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

    # Dtermine the final image
    #-------------------------#
    # _img_result = _img
    # _img_result = _img_preprocessed
    _img_result = _img_LM
    #-------------------------#

    # Flip the result image if needed
    #----------------------------------#
    if is_h_mirrored_image:
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

np_distance_ratio_vec = np.vstack(distance_ratio_list)
np_distance_error_vec = np.vstack(distance_error_list)

mean_distance_ratio = np.average(np_distance_ratio_vec)
mean_distance_error = np.average(np_distance_error_vec)
error_distance_MAE = np.linalg.norm(np_distance_error_vec, ord=1)/(np_distance_error_vec.shape[0])
print("mean_distance_ratio = %f" % mean_distance_ratio)
print("mean_distance_error = %f" % mean_distance_error)
print("error_distance_MAE = %f" % error_distance_MAE)
