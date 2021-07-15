import numpy as np
import copy
import time
import csv
#
import cv2
#
import PNP_SOLVER_LIB as PNPS

#---------------------------#
# Landmark (LM) dataset
data_dir_str = '/home/benson516/test_PnP_solver/dataset/Real_face_LM_1/'
data_file_str = 'FD_result_jpg640.csv'
#---------------------------#
# Image of Alexander
# Original image
image_dir_str = '/home/benson516/test_PnP_solver/dataset/Real_face_LM_1/images/original/'
# The image used for analysis
image_result_unflipped_dir_str = '/home/benson516/test_PnP_solver/dataset/Real_face_LM_1/images/result_unflipped/'
# The same as the original image
image_result_dir_str = '/home/benson516/test_PnP_solver/dataset/Real_face_LM_1/images/result/'
#---------------------------#
# Result CSV file
result_csv_dir_str = '/home/benson516/test_PnP_solver/dataset/Real_face_LM_1/result_CSVs/'
result_csv_file_prefix_str = "result_csv_"
result_statistic_txt_file_prefix_str = "statistic_"

# Behavior of this program
#---------------------------#
is_run_through_all_data = True
# is_run_through_all_data = False
# Data
is_limiting_line_count = True
# is_limiting_line_count = False
# DATA_START_ID = 0
# DATA_START_ID = 658
# DATA_START_ID = 379 # (0, 0, 0)
DATA_START_ID = 926 # (0, -20, 0)
# DATA_START_ID = 1070 # (0, 40, 0)
# DATA_START_ID = 146 # 1203 # 1205 # 1124 # 616 # 487 # 379 # 934 # 893 # 540 # 512 # 775 # (0, 0, 0), d=220~20
DATA_COUNT =  3
#
verbose = True
# verbose = False
# Image display
is_showing_image = True
# is_showing_image = False
#
# Statistic CSV file
# is_statistic_csv_horizontal = True # class --> (right)
is_statistic_csv_horizontal = False # class (|, down)
#---------------------------#

# Not to flush the screen
if is_run_through_all_data:
    DATA_START_ID = 0
    is_limiting_line_count = False
    verbose = False
    is_showing_image = False
#

# Parameters of the data
#---------------------------#
# is_mirrored_image = True
is_mirrored_image = False
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
    _line = _f.readline() # Ignore the first line
    _line = _f.readline()
    _idx = 0
    while (_line != '') and ((not is_limiting_line_count) or (_idx < (DATA_START_ID+DATA_COUNT) ) ):  # The EOF char is an empty string
        if _idx >= DATA_START_ID:
            data_idx_list.append(_idx)
            # print(_line, end='')
            _line_split_list = _line.split(",") # CSV file
            # print(_line_split_list)
            data_str_list_list.append(_line_split_list)
            #
            _name = _line_split_list[0]
            if _name.rfind('.jpg') == (len(_name) - 4):
                _name = _name[:-4]
            data_name_split_list = _name.split('_')
            # print("data_name_split_list = %s" % str(data_name_split_list))
            data_name_split_list_list.append( data_name_split_list )
        # Update
        _line = _f.readline()
        _idx += 1
#
print(data_idx_list[0])
print(data_str_list_list[0][0:5]) # [data_idx][column in line of file]
print(data_name_split_list_list[0]) # [data_idx][column in file name split]
# print(data_name_split_list_list[0][9:12]) # [data_idx][column in file name split]
#----------------------------------------------------------#

# Convert the original data to structured data_list
#-------------------------------------------------------#
data_list = list()
for _idx in range(len(data_str_list_list)):
    data_id_dict = dict()
    # File info
    data_id_dict['idx'] = data_idx_list[_idx]
    data_id_dict['file_name'] = data_str_list_list[_idx][0]
    data_id_dict['person_name'] = data_name_split_list_list[_idx][0]
    # "Label" of classes, type: string
    _class_dict = dict()
    _class_dict['distance'] = data_name_split_list_list[_idx][1]
    _class_dict['pitch'] = "NA"
    _class_dict['roll'] = "NA"
    _class_dict['yaw'] = "NA"
    data_id_dict['class'] = _class_dict
    # Grund truth
    data_id_dict['distance'] = float(data_name_split_list_list[_idx][1])
    data_id_dict['pitch'] = 0.0 # Actually, None
    data_id_dict['roll'] = 0.0 # Actually, None
    data_id_dict['yaw'] = 0.0 # Actually, None
    # Test inputs
    data_id_dict['box_xy'] = np.array(data_name_split_list_list[_idx][3:5]).astype(np.float) # np array, shape=(2,)
    data_id_dict['box_h'] = float(data_name_split_list_list[_idx][5]) #
    norm_box_h = 112 # pixel
    data_id_dict['LM_local_norm'] = (np.array([data_str_list_list[_idx][1::2], data_str_list_list[_idx][2::2]]).T).astype(np.float) / float(norm_box_h) # np array, shape=(2,)
    #
    data_id_dict['LM_pixel'] = data_id_dict['LM_local_norm'] * data_id_dict['box_h'] + data_id_dict['box_xy'].reshape((1,2))
    # Sppend to the total data list
    data_list.append(data_id_dict)
#
print(data_list[0])
#-------------------------------------------------------#




# ============= Start testing ================
#-------------------------------------------------------#
# Parameters and data
# Camera intrinsic matrix (Ground truth)
# f_camera = 188.55 # 175.0
# f_camera = 225.68717584155982 # / 1.15 0.864246
f_camera = 269.2140797750142 # = 225.68717584155982 / 0.864246 / 0.97 # The FOV of the real camera
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
# list: [x,y,z]
point_3d_dict = dict()
# Note: Each axis should exist at least 3 different values to make A_all full rank
# Note: the Landmark definition in the pitcture in reversed
point_3d_dict["eye_l_9"] = [ 0.032, 0.0, 0.0] # [ 0.035, 0.0, 0.0]
point_3d_dict["eye_r_8"] = [-0.032, 0.0, 0.0] # [ 0.035, 0.0, 0.0]
point_3d_dict["eye_c_6"] = [0.0, 0.0, 0.0]
point_3d_dict["mouse_l_12"] = [ 0.027, 0.070, 0.0] # [ 0.025, 0.085, 0.0]
point_3d_dict["mouse_r_11"] = [ -0.027, 0.070, 0.0] # [ -0.025, 0.085, 0.0]
point_3d_dict["nose_t_7"] = [ -0.005, 0.0455, -0.03] # [ 0.0, 0.0455, 0.03] # [ 0.0, 0.046, 0.03]
point_3d_dict["chin_t_10"] = [0.0, 0.12, 0.0]
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
pnp_solver = PNPS.PNP_SOLVER(np_K_camera_est, point_3d_dict, pattern_scale=pattern_scale, verbose=verbose)




def convert_pixel_to_homo(pixel_xy, mirrored=is_mirrored_image):
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

# Collect the result
#--------------------------#
result_list = list()
#--------------------------#

# Loop thrugh data
for _idx in range(len(data_list)):
    print("\n-------------- data_idx = %d (process idx = %d)--------------\n" % (data_list[_idx]['idx'], _idx))
    print('file file_name: [%s]' % data_list[_idx]['file_name'])


    LM_pixel_data_matrix = data_list[_idx]['LM_pixel'] # [LM_id] --> [x,y]
    np_point_image_dict = dict()
    # [x,y,1].T, shape: (3,1)
    np_point_image_dict["eye_l_9"] = convert_pixel_to_homo(LM_pixel_data_matrix[9])
    np_point_image_dict["eye_r_8"] = convert_pixel_to_homo(LM_pixel_data_matrix[8])
    np_point_image_dict["eye_c_6"] = convert_pixel_to_homo(LM_pixel_data_matrix[6])
    np_point_image_dict["mouse_l_12"] = convert_pixel_to_homo(LM_pixel_data_matrix[12])
    np_point_image_dict["mouse_r_11"] = convert_pixel_to_homo(LM_pixel_data_matrix[11])
    np_point_image_dict["nose_t_7"] = convert_pixel_to_homo(LM_pixel_data_matrix[7])
    np_point_image_dict["chin_t_10"] = convert_pixel_to_homo(LM_pixel_data_matrix[10])
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
    print("np_R_GT = \n%s" % str(np_R_GT))
    print("np_t_GT_est = \n%s" % str(np_t_GT_est))
    print("-"*30 + " The End " + "-"*30)
    print()
    #----------------------------#

    # Store the error for statistic
    #----------------------------#
    _result_idx_dict = dict()
    _result_idx_dict['idx'] = data_list[_idx]['idx']
    _result_idx_dict["file_name"] = data_list[_idx]['file_name']
    _result_idx_dict["drpy"] = (distance_GT, roll_GT, pitch_GT, yaw_GT)
    _result_idx_dict["class"] = data_list[_idx]['class']
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
    #
    result_list.append(_result_idx_dict)
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


# # Store the error for statistic
# #----------------------------#
# _result_idx_dict = dict()
# _result_idx_dict['idx'] = data_list[_idx]['idx']
# _result_idx_dict["file_name"] = data_list[_idx]['file_name']
# _result_idx_dict["drpy"] = (distance_GT, roll_GT, pitch_GT, yaw_GT)
# #-------------------------------#
# # R, t, depth, roll, pitch, yaw, residual
# #---#
# # GT
# # Result
# # Error, err = (est - GT)
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
# # roll
# _result_idx_dict["roll_GT"] = roll_GT
# _result_idx_dict["roll_est"] = roll_est
# _result_idx_dict["roll_err"] = roll_est - roll_GT
# # pitch
# _result_idx_dict["pitch_GT"] = pitch_GT
# _result_idx_dict["pitch_est"] = pitch_est
# _result_idx_dict["pitch_err"] = pitch_est - pitch_GT
# # yaw
# _result_idx_dict["yaw_GT"] = yaw_GT
# _result_idx_dict["yaw_est"] = yaw_est
# _result_idx_dict["yaw_err"] = yaw_est - yaw_GT
# # residual
# _result_idx_dict["res_norm"] = res_norm
# #
# result_list.append(_result_idx_dict)
# #----------------------------#



#-------------------------------------------------------#
def get_statistic_of_result(result_list, class_name='all', class_label='all', data_est_key="t3_est", data_GT_key="distance_GT", unit="m", unit_scale=1.0, verbose=True):
    '''
    '''
    if len(result_list) == 0:
        return (1.0, 0.0, 0.0, 0.0, 0.0)
    data_est_vec = np.vstack( [ _d[ data_est_key ] for _d in result_list] )
    data_GT_vec = np.vstack( [ _d[ data_GT_key ] for _d in result_list] )
    _np_data_ratio_vec = data_est_vec / data_GT_vec
    _np_data_error_vec = data_est_vec - data_GT_vec

    ratio_mean = np.average(_np_data_ratio_vec)
    error_mean = np.average(_np_data_error_vec)
    error_variance = (np.linalg.norm( (_np_data_error_vec - error_mean), ord=2)**2)  / (_np_data_error_vec.shape[0])
    error_stddev = error_variance**0.5
    MAE_2_GT = np.linalg.norm(_np_data_error_vec, ord=1)/(_np_data_error_vec.shape[0])
    MAE_2_mean = np.linalg.norm((_np_data_error_vec - error_mean), ord=1)/(_np_data_error_vec.shape[0])
    #
    if verbose:
        print("class: [%s], class_label: [%s]" % (class_name, class_label))
        print("ratio_mean (estimated/actual) = %f" % ratio_mean)
        print("error_mean = %f %s" % (error_mean*unit_scale, unit))
        print("error_stddev = %f %s" % (error_stddev*unit_scale, unit))
        print("MAE_2_GT = %f %s" % (MAE_2_GT*unit_scale, unit))
        print("MAE_2_mean = %f %s" % (MAE_2_mean*unit_scale, unit))
        print("\n")
    # return (ratio_mean, error_mean, error_stddev, MAE_2_GT, MAE_2_mean)

    # Capsulate the statistic results as a dict, which contains all the matric names.
    statis_dict = dict()
    statis_dict['m_ratio'] = ratio_mean
    statis_dict['mean(%s)' % unit] = error_mean * unit_scale
    statis_dict['stddev(%s)' % unit] = error_stddev * unit_scale
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
    return ( (-1) if (e == "all") else int(e))

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

def write_statistic_to_csv(class_statistic_dict, statistic_csv_path, class_name="distance", statistic_data_name="depth", is_horizontal=is_statistic_csv_horizontal):
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
        print("\n*** Wrote the statistic results [%s] to the csv file:\n\t[%s]\n" % ( _hv_adv_str, csv_path))
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
write_result_to_csv(result_list, csv_path)




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
write_statistic_to_txt(class_statistic_dict, statistic_txt_path, class_name=class_name, statistic_data_name=statistic_data_name)
write_statistic_to_csv(class_statistic_dict, statistic_csv_path, class_name=class_name, statistic_data_name=statistic_data_name)
#
statistic_data_name = "roll" # Just the name as the info. to the reader
class_statistic_dict = dist_2_roll_statistic_dict
statistic_txt_path = result_csv_dir_str + result_statistic_txt_file_prefix_str + data_file_str[:-4] + ( "_%s_to_%s" % (class_name, statistic_data_name) ) + '.txt'
statistic_csv_path = result_csv_dir_str + result_statistic_txt_file_prefix_str + data_file_str[:-4] + ( "_%s_to_%s" % (class_name, statistic_data_name) ) + '.csv'
write_statistic_to_txt(class_statistic_dict, statistic_txt_path, class_name=class_name, statistic_data_name=statistic_data_name)
write_statistic_to_csv(class_statistic_dict, statistic_csv_path, class_name=class_name, statistic_data_name=statistic_data_name)
#
statistic_data_name = "pitch" # Just the name as the info. to the reader
class_statistic_dict = dist_2_pitch_statistic_dict
statistic_txt_path = result_csv_dir_str + result_statistic_txt_file_prefix_str + data_file_str[:-4] + ( "_%s_to_%s" % (class_name, statistic_data_name) ) + '.txt'
statistic_csv_path = result_csv_dir_str + result_statistic_txt_file_prefix_str + data_file_str[:-4] + ( "_%s_to_%s" % (class_name, statistic_data_name) ) + '.csv'
write_statistic_to_txt(class_statistic_dict, statistic_txt_path, class_name=class_name, statistic_data_name=statistic_data_name)
write_statistic_to_csv(class_statistic_dict, statistic_csv_path, class_name=class_name, statistic_data_name=statistic_data_name)
#
statistic_data_name = "yaw" # Just the name as the info. to the reader
class_statistic_dict = dist_2_yaw_statistic_dict
statistic_txt_path = result_csv_dir_str + result_statistic_txt_file_prefix_str + data_file_str[:-4] + ( "_%s_to_%s" % (class_name, statistic_data_name) ) + '.txt'
statistic_csv_path = result_csv_dir_str + result_statistic_txt_file_prefix_str + data_file_str[:-4] + ( "_%s_to_%s" % (class_name, statistic_data_name) ) + '.csv'
write_statistic_to_txt(class_statistic_dict, statistic_txt_path, class_name=class_name, statistic_data_name=statistic_data_name)
write_statistic_to_csv(class_statistic_dict, statistic_csv_path, class_name=class_name, statistic_data_name=statistic_data_name)
