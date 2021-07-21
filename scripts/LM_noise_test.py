import numpy as np
import copy
import time
import csv
import os
#
import cv2
import matplotlib.pyplot as plt
#
import PNP_SOLVER_LIB as PNPS
import TEST_TOOLBOX as TTBX

# ctrl+c
from signal import signal, SIGINT


#---------------------------#
# Landmark (LM) dataset
data_name = 'LM_noise'
result_root_dir_str = '/home/benson516/test_PnP_solver/dataset/' + data_name + '/'
#---------------------------#
# Plots
# The same as the original image
result_plot_dir_str = result_root_dir_str + 'plots/'
#---------------------------#
# Result CSV file
result_data_dir_str = result_root_dir_str + 'result_data/'


# Make directories for storing the results if they do not exist.
#------------------------------------------------------------------#
os.makedirs(result_plot_dir_str, mode=0o777, exist_ok=True)
os.makedirs(result_plot_dir_str, mode=0o777, exist_ok=True)
os.makedirs(result_data_dir_str, mode=0o777, exist_ok=True)
#------------------------------------------------------------------#

# Behavior of this program
#---------------------------#

bbox_resolution = 112.0
# bbox_resolution = 224.0

# Data generation
is_quantized = True
# is_quantized = False

#
# verbose = True
verbose = False

#---------------------------#



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









# ============= Generating Data ================
#----------------------------------------------------------#



# Generating random perturbations for all test
#-------------------------------------#
# Random generator
random_seed = 42
# random_seed = None
random_gen = np.random.default_rng(seed=random_seed)
#
n_noise = 3 # 10 # 300 # The number of noise pattern for different sameples
anchor_point_key = "eye_c_51"
n_point_to_perturb = len(point_3d_dict_list[0]) - 1
LM_noise_set = random_gen.multivariate_normal( np.zeros((2,)), np.eye(2), (n_noise, n_point_to_perturb)) # shape = (n_noise, n_point_to_perturb, 2)
print("LM_noise_set.shape = %s" % str(LM_noise_set.shape))
#-------------------------------------#


# Control variables list
#-------------------------------------#
n_ctrl_yaw = 10 # 15
n_ctrl_noise_norm = 3 # 15
#
ctrl_bound_yaw = (0.0, 50.0) # deg
ctrl_bound_noise_stddev = (0.0, 5.0) # pixel, in bbox local coordinate
#
# Generate value list
test_ctrl_yaw_list = np.linspace(ctrl_bound_yaw[0], ctrl_bound_yaw[1], num=n_ctrl_yaw) # deg.
test_ctrl_noise_stddev_list = np.linspace(ctrl_bound_noise_stddev[0], ctrl_bound_noise_stddev[1], num=n_ctrl_noise_norm)
#-------------------------------------#


# Fixed conditions
#-------------------------------------#
fixed_depth = 1.0 # m
fixed_roll = 0.0 # deg.
fixed_pitch = 0.0 # deg.
#
bbox_size = 30 # pixel @ 1m
#-------------------------------------#

def convert_bbox_scale_to_global(value, bbox_size_local=112, bbox_size_global=30):
    # The function for converting the local value in a bounding box back to global image.
    return (value / float(bbox_size_local) * float(bbox_size_global))





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
sample_count = 0


# Result mesh
mesh_yn_yaw_error_mean = np.zeros( (n_ctrl_yaw, n_ctrl_noise_norm) )
mesh_yn_yaw_error_stddev = np.zeros( (n_ctrl_yaw, n_ctrl_noise_norm) )
mesh_yn_yaw_MAE = np.zeros( (n_ctrl_yaw, n_ctrl_noise_norm) )
#
for ctrl_noise_idx, noise_stddev_i in enumerate(test_ctrl_noise_stddev_list):
    for ctrl_yaw_idx, yaw_i in enumerate(test_ctrl_yaw_list):
        # at (yaw_i, noise_stddev_i)
        print("(yaw_i, noise_stddev_i) = (%f, %f)" % (yaw_i, noise_stddev_i))

        # Generate the landmarks (quantized, or not)
        #------------------------------------#
        np_R_GT = pnp_solver_GT.get_rotation_matrix_from_Euler(fixed_roll, yaw_i, fixed_pitch, is_degree=True)
        np_t_GT = np.array([0.0, 0.0, fixed_depth]).reshape((3,1))
        pnp_solver_GT.set_golden_pattern_id(0) # Use the number 0 pattern to generate the LM
        quantize_q = convert_bbox_scale_to_global(1.0, bbox_size_local=bbox_resolution, bbox_size_global=bbox_size)
        LM_pixel_dict = pnp_solver_GT.perspective_projection_golden_landmarks(np_R_GT, np_t_GT, is_quantized=is_quantized, quantize_q=quantize_q, is_pretrans_points=False, is_returning_homogeneous_vec=False) # image (u, v)
        #------------------------------------#


        # Convert noise to global image scale
        #------------------------------------#
        noise_stddev_i_global = convert_bbox_scale_to_global(noise_stddev_i, bbox_size_local=bbox_resolution, bbox_size_global=bbox_size)
        print("noise_stddev_i_global = %f" % noise_stddev_i_global)
        #------------------------------------#


        # Run the statistic test over noise patterns under this condition
        yaw_error_list = list()
        for noise_idx in range(LM_noise_set.shape[0]):
            _LM_pixel_polluted_dict = copy.deepcopy(LM_pixel_dict)
            noise_i = noise_stddev_i_global * LM_noise_set[noise_idx, :, :] # shape = (n_point_to_perturb, 2)

            # Add noise to LMs
            #------------------------------------#
            _LM_idx = 0
            for _k in _LM_pixel_polluted_dict:
                if _k != anchor_point_key:
                    _LM_pixel_polluted_dict[_k] += noise_i[_LM_idx, :].reshape((2,))
                    _LM_idx += 1
            #------------------------------------#

            # Convert to homogeneous coordinate
            #------------------------------------#
            np_point_image_dict = dict()
            for _k in _LM_pixel_polluted_dict:
                np_point_image_dict[_k] = np.vstack( (_LM_pixel_polluted_dict[_k].reshape((2,1)), 1.0) )
            # # Print
            # print("-"*35)
            # print("2D points on image:")
            # for _k in np_point_image_dict:
            #     print("%s:\n%s.T" % (_k, str(np_point_image_dict[_k].T)))
            # print("-"*35)
            #------------------------------------#

            # Solve
            #------------------------------------#
            np_R_est, np_t_est, t3_est, roll_est, yaw_est, pitch_est, res_norm = pnp_solver.solve_pnp(np_point_image_dict)
            #------------------------------------#

            # Calculate error
            #------------------------------------#
            yaw_err = yaw_est - yaw_i
            yaw_error_list.append(yaw_err)
            # print("yaw_err = %f" % yaw_err)
            #------------------------------------#

            # Terminate
            #------------------------------------#
            if received_SIGINT:
                break
            #------------------------------------#
        #

        # Calculate the error
        #------------------------------------#
        # yaw_error_list
        yaw_error_mean = np.mean(yaw_error_list)
        yaw_error_stddev = np.std(yaw_error_list)
        yaw_MAE = np.mean(np.abs(yaw_error_list))
        print("yaw_error_mean = %f" % yaw_error_mean)
        print("yaw_error_stddev = %f" % yaw_error_stddev)
        print("yaw_MAE = %f" % yaw_MAE)
        #------------------------------------#

        # Store the result into the meshes
        #------------------------------------#
        mesh_yn_yaw_error_mean[ctrl_yaw_idx, ctrl_noise_idx] = yaw_error_mean
        mesh_yn_yaw_error_stddev[ctrl_yaw_idx, ctrl_noise_idx] = yaw_error_stddev
        mesh_yn_yaw_MAE[ctrl_yaw_idx, ctrl_noise_idx] = yaw_MAE
        #------------------------------------#




# is_transparent = True
is_transparent = False

def plot_yaw_2_yaw_error(x, Y, title, y_label, test_ctrl_noise_stddev_list, result_plot_dir_str, is_transparent=False):
    plt.figure(title)
    plt.plot(x, Y)
    plt.title(title)
    plt.xlabel('Yaw (deg.)')
    plt.ylabel(y_label)
    plt.legend([r"$\sigma$ = %.1f deg." % e for e in test_ctrl_noise_stddev_list])
    file_path = result_plot_dir_str + '_'.join( title.split(" ") ) + '.png'
    plt.savefig( file_path, transparent=is_transparent)

def plot_noise_2_yaw_error(x, Y, title, y_label, test_ctrl_yaw_list, result_plot_dir_str, is_transparent=False):
    plt.figure(title)
    plt.plot(x, Y[::3, :].T)
    plt.title(title)
    plt.xlabel(r'Noise stddev $\sigma$ (deg.)')
    plt.ylabel(y_label)
    plt.legend(["Yaw = %.1f deg." % e for e in test_ctrl_yaw_list])
    file_path = result_plot_dir_str + '_'.join( title.split(" ") ) + '.png'
    plt.savefig( file_path, transparent=is_transparent)

# Mean
title = "Yaw to yaw error mean"
plt.figure(title)
plt.plot(test_ctrl_yaw_list, mesh_yn_yaw_error_mean)
plt.title(title)
plt.xlabel('Yaw (deg.)')
plt.ylabel('Yaw MAE (deg.)')
plt.legend([r"$\sigma$ = %.1f deg." % e for e in test_ctrl_noise_stddev_list])
file_path = result_plot_dir_str + '_'.join( title.split(" ") ) + '.png'
plt.savefig( file_path, transparent=is_transparent)
# stddev
title = "Yaw to yaw error stddev"
plt.figure(title)
plt.plot(test_ctrl_yaw_list, mesh_yn_yaw_error_stddev)
plt.title(title)
plt.xlabel('Yaw (deg.)')
plt.ylabel('Yaw MAE (deg.)')
plt.legend([r"$\sigma$ = %.1f deg." % e for e in test_ctrl_noise_stddev_list])
file_path = result_plot_dir_str + '_'.join( title.split(" ") ) + '.png'
plt.savefig( file_path, transparent=is_transparent)
# MAE
title = "Yaw to yaw MAE"
plt.figure(title)
plt.plot(test_ctrl_yaw_list, mesh_yn_yaw_MAE)
plt.title(title)
plt.xlabel('Yaw (deg.)')
plt.ylabel('Yaw MAE (deg.)')
plt.legend([r"$\sigma$ = %.1f deg." % e for e in test_ctrl_noise_stddev_list])
file_path = result_plot_dir_str + '_'.join( title.split(" ") ) + '.png'
plt.savefig( file_path, transparent=is_transparent)


# Mean
title = "Noise stddev to yaw error mean"
plt.figure(title)
plt.plot(test_ctrl_noise_stddev_list, mesh_yn_yaw_error_mean[::3, :].T)
plt.title(title)
plt.xlabel(r'Noise stddev $\sigma$ (deg.)')
plt.ylabel('Yaw MAE (deg.)')
plt.legend(["Yaw = %.1f deg." % e for e in test_ctrl_yaw_list])
file_path = result_plot_dir_str + '_'.join( title.split(" ") ) + '.png'
plt.savefig( file_path, transparent=is_transparent)
# stddev
title = "Noise stddev to yaw error stddev"
plt.figure(title)
plt.plot(test_ctrl_noise_stddev_list, mesh_yn_yaw_error_stddev[::3, :].T)
plt.title(title)
plt.xlabel(r'Noise stddev $\sigma$ (deg.)')
plt.ylabel('Yaw MAE (deg.)')
plt.legend(["Yaw = %.1f deg." % e for e in test_ctrl_yaw_list])
file_path = result_plot_dir_str + '_'.join( title.split(" ") ) + '.png'
plt.savefig( file_path, transparent=is_transparent)
# MAE
title = "Noise stddev to yaw MAE"
plt.figure(title)
plt.plot(test_ctrl_noise_stddev_list, mesh_yn_yaw_MAE[::3, :].T)
plt.title(title)
plt.xlabel(r'Noise stddev $\sigma$ (deg.)')
plt.ylabel('Yaw MAE (deg.)')
plt.legend(["Yaw = %.1f deg." % e for e in test_ctrl_yaw_list])
file_path = result_plot_dir_str + '_'.join( title.split(" ") ) + '.png'
plt.savefig( file_path, transparent=is_transparent)

plt.show()
