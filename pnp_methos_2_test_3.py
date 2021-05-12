import numpy as np
import scipy.linalg as sp_lin

# Mode
#-----------------------------#
# is_quantized = True
is_quantized = False
print("is_quantized = %s" % str(is_quantized))
#-----------------------------#


#-----------------------------------------------------------#
def print_matrix_and_eigen_value(m_name, m_in, is_printing_eig_vec=False):
    w, v = np.linalg.eig(m_in)
    #
    np.set_printoptions(suppress=True)
    print()
    print("==== (Start) Eigen of %s ===" % m_name)
    print("%s = \n%s" % (m_name, str(m_in)))
    print("%s_eig_value = \n%s" % (m_name, str(w)))
    if is_printing_eig_vec:
        print("%s_eig_vec = \n%s" % (m_name, str(v)))
    print("====  (End)  Eigen of %s ===" % m_name)
    print()
    np.set_printoptions(suppress=False)
    return (w, v)

def print_matrix_and_SVD(m_name, m_in, is_printing_u_vh=False):
    u, s, vh = np.linalg.svd(m_in)
    _norm = np.linalg.norm(m_in)
    #
    np.set_printoptions(suppress=True)
    print()
    print("==== (Start) SVD of %s ===" % m_name)
    print("%s = \n%s" % (m_name, str(m_in)))
    print("||%s|| = %f" % (m_name, _norm))
    if is_printing_u_vh:
        print("%s_U = \n%s" % (m_name, str(u)))
    print("%s_S = \n%s" % (m_name, str(s)))
    if is_printing_u_vh:
        print("%s_Vh = \n%s" % (m_name, str(vh)))
    print("====  (End)  SVD of %s ===" % m_name)
    print()
    np.set_printoptions(suppress=False)
    return (u, s, vh)
#-----------------------------------------------------------#

#-----------------------------------------------------------#
def get_symmetry(A_in):
    return 0.5*(A_in + A_in.T)

def get_skew(A_in):
    return 0.5*(A_in - A_in.T)

def unit_vec(vec_in):
    _norm = np.linalg.norm(vec_in)
    if np.abs(_norm) <= 10**-7:
         _norm_inv = 1.0
         print("_norm = %f" % _norm)
         print("_norm approaches zeros!!")
    else:
         _norm_inv = 1.0/_norm
    return (vec_in * _norm_inv)
#-----------------------------------------------------------#


#-----------------------------------------------------------#
def fix_R_svd(R_in):
    # print("R_in = \n%s" % str(R_in))
    G_u, G_s, G_vh = np.linalg.svd(R_in)
    # print("G_u = \n%s" % str(G_u))
    # print("G_s = \n%s" % str(G_s))
    # print("G_vh = \n%s" % str(G_vh))
    G_D = np.linalg.det(G_u @ G_vh)
    # print("G_D = %f" % G_D)
    # Reconstruct R
    np_R_est = G_u @ np.diag([1.0, 1.0, G_D]) @ G_vh
    # print("np_R_est = \n%s" % str(np_R_est))
    return np_R_est
#-----------------------------------------------------------#

#-----------------------------------------------------------#
def get_rotation_matrix_from_Euler(roll, yaw, pitch):
    '''
    roll, yaw, pitch --> R
    '''
    c1 = np.cos(roll)
    s1 = np.sin(roll)
    c2 = np.cos(yaw)
    s2 = np.sin(yaw)
    c3 = np.cos(pitch)
    s3 = np.sin(pitch)
    R_roll_0 = np.array([[ c1*c2, (s1*c3 + c1*s2*s3), (s1*s3 - c1*s2*c3)]])
    R_roll_1 = np.array([[-s1*c2, (c1*c3 - s1*s2*s3), (c1*s3 + s1*s2*c3)]])
    R_roll_2 = np.array([[ s2,    -c2*s3,              c2*c3            ]])
    # print("np.cross(R_roll_0, R_roll_1) = %s" % str(np.cross(R_roll_0, R_roll_1)))
    # print("R_roll_2 = %s" % str(R_roll_2))
    _R01 = np.concatenate((R_roll_0, R_roll_1), axis=0)
    R = np.concatenate((_R01, R_roll_2), axis=0)
    return R

def get_Euler_from_rotation_matrix(R_in):
    '''
    R --> roll, yaw, pitch
    '''
    _eps = 10.0**-7
    # if np.linalg.norm([R_in[2,1], R_in[2,2]]) <= _eps: # Note: abs(c2) = np.linalg.norm([R_in[2,1], R_in[2,2]])
    if np.abs(np.pi/2.0 - np.arcsin(np.abs(R_in[2,0]))) <= _eps: # Faster
        # Singularity, "gimbal locked"
        yaw = np.sign(R_in[2,0]) * (np.pi/2.0) # theta_2
        # Assume pitch = 0.0 (theta_3 = 0.0)
        pitch = 0.0 # theta_3
        roll = np.arctan2(R_in[0,1], R_in[1,1]) # theta_1
    else:
        # Normal case
        roll = np.arctan2(-R_in[1,0], R_in[0,0]) # theta_1
        pitch = np.arctan2(-R_in[2,1], R_in[2,2]) # theta_3
        #
        c1 = np.cos(roll)
        c3 = np.cos(pitch)
        if np.abs(c1) > np.abs(c3):
            print("Set c2 as (r11/c1)")
            c2 = R_in[0,0]/c1
        else:
            print("Set c2 as (r33/c3)")
            c2 = R_in[2,2]/c3
        yaw = np.arctan2(R_in[2,0], c2) # theta_2
    return (roll, yaw, pitch)
#-----------------------------------------------------------#

# Parameters and data
# Camera intrinsic matrix (Ground truth)
fx_camera = 111.0 # 201.0 # 111.0
fy_camera = fx_camera # 111.0
xo_camera = 100.0
yo_camera = 100.0
np_K_camera_GT = np.array([[fx_camera, 0.0, xo_camera], [0.0, fy_camera, yo_camera], [0.0, 0.0, 1.0]])
print("np_K_camera = \n%s" % str(np_K_camera_GT))

# 3D landmark point - local coordinate
#----------------------------------------#
# list: [x,y,z]
point_3d_dict = dict()
# Note: Each axis should exist at least 3 different values to make A_all full rank
point_3d_dict["eye_l"] = [ 0.05, 0.0, 0.0]
point_3d_dict["eye_r"] = [-0.05, 0.0, 0.0]
point_3d_dict["mouse_l"] = [ 0.03, 0.07, 0.0]
point_3d_dict["mouse_r"] = [ -0.03, 0.07, 0.0]
point_3d_dict["nose_t"] = [ 0.0, 0.035, 0.015]
point_3d_dict["face_c"] = [ 0.0, 0.035, 0.0]
# Convert to numpy vector, shape: (3,1)
np_point_3d_dict = dict()
print("-"*35)
print("3D points in local coordinate:")
for _k in point_3d_dict:
    np_point_3d_dict[_k] = np.array(point_3d_dict[_k]).reshape((3,1))
    print("%s:\n%s" % (_k, str(np_point_3d_dict[_k])))
print("-"*35)
# print(np_point_3d_dict)
#----------------------------------------#

# Ground truth (R, t)
#---------------------------------------#
# Rotation
roll_GT, yaw_GT, pitch_GT = np.deg2rad(15), np.deg2rad(30), np.deg2rad(25)
# roll_GT, yaw_GT, pitch_GT = np.deg2rad(15), np.deg2rad(10), np.deg2rad(-30)
print("(roll_GT, yaw_GT, pitch_GT) \t= %s" % str( np.rad2deg((roll_GT, yaw_GT, pitch_GT)) ) )
np_R_GT = get_rotation_matrix_from_Euler(roll_GT, yaw_GT, pitch_GT)
print("np_R_GT = \n%s" % str(np_R_GT))
_result = get_Euler_from_rotation_matrix(np_R_GT)
print("(roll, yaw, pitch) \t\t= %s" % str( np.rad2deg(_result) ) )
# Translation (x,y,z) in camera frame
np_t_GT = np.array([0.2, -0.05, 1.2]).reshape((3,1))
print("np_t_GT = \n%s" % str(np_t_GT))
#---------------------------------------#

# Generate projection sample
#--------------------------------------#
#
np_point_image_dict = dict()
# [x,y,1].T, shape: (3,1)
print("-"*35)
print("is_quantized = %s" % str(is_quantized))
print("2D points on image:")
for _k in np_point_3d_dict:
    _ray = np_K_camera_GT @ (np_R_GT @ np_point_3d_dict[_k] + np_t_GT)
    # normalize
    _projection_i = _ray/_ray[2,0]
    # Quantize
    if is_quantized:
        np_point_image_dict[_k] = np.around(_projection_i) # with quantization
    else:
        np_point_image_dict[_k] = _projection_i # no quantization
    print("%s:\n%s" % (_k, str(np_point_image_dict[_k])))
print("-"*35)
# print("np_point_image_dict = \n%s" % str(np_point_image_dict))
#--------------------------------------#

# Solution (approach 2)
#-----------------------------------------------------------#
def get_Delta_i(theta_i, phi_3_est, id=None):
    '''
    theta_i=[[x,y,z]].T, shape=(3,1)
    phi_3_est = [[xu, yu, zu]].T, shape=(3,1)
    '''
    _theta_i_T = theta_i.reshape((1,3)) # Incase it's 1-D array, use reshape instead of theta_i.T
    _phi_3 = phi_3_est.reshape((3,1))
    Delta_i = (_theta_i_T @ _phi_3 + 1.0)
    _eps = 10**-7
    if np.abs(Delta_i) <= _eps:
        print("Delta[%d] is too close to zero!!" % (id if id is not None else -1) )
        Delta_i = _eps
    return Delta_i

def get_A_i(theta_i, Delta_i):
    '''
    theta_i=[[x,y,z]].T, shape=(3,1)
    Delta_i=(theta_i.T @ phi_3 + 1.0), scalar
    '''
    _theta_i_T = theta_i.reshape((1,3)) # Incase it's 1-D array, use reshape instead of theta_i.T
    #
    _theta_i_T_div_Delta_i = _theta_i_T / Delta_i
    #
    A_i_c0 = np.concatenate( (_theta_i_T_div_Delta_i ,  np.zeros((1,3))         ), axis=0)
    A_i_c1 = np.concatenate( ( np.zeros((1,3))       ,  _theta_i_T_div_Delta_i  ), axis=0)
    #
    A_i_c01 = np.concatenate((A_i_c0, A_i_c1), axis=1)
    A_i = np.concatenate((A_i_c01, np.eye(2)), axis=1)
    return A_i

def get_Delta_A_all(np_point_3d_dict_in, np_point_image_dict_in, phi_3_est):
    '''
    '''
    # To be more realistic, use the image point to search
    A_i_list = list()
    Delta_i_list = list()
    for _id, _k in enumerate(np_point_image_dict_in):
        Delta_i = get_Delta_i(np_point_3d_dict_in[_k], phi_3_est, id=_id)
        Delta_i_list.append( Delta_i )
        A_i_list.append( get_A_i(np_point_3d_dict_in[_k], Delta_i) )
    Delta_all = np.vstack(Delta_i_list)
    A_all = np.vstack(A_i_list)
    return (Delta_all, A_all)

def get_B_all(np_point_image_dict_in, K_in):
    '''
    '''
    _K_inv = np.linalg.inv(K_in)
    # To be more realistic, use the image point to search
    B_i_list = list()
    for _k in np_point_image_dict_in:
        nu_i = _K_inv @ np_point_image_dict_in[_k] # shape = (3,1)
        B_i_list.append( nu_i[0:2,:] )
    B_all = np.vstack(B_i_list)
    return B_all






# Form the problem for solving
B_all = get_B_all(np_point_image_dict, np_K_camera_GT)
print("B_all = \n%s" % str(B_all))
print("B_all.shape = %s" % str(B_all.shape))

# print("A_all = \n%s" % str(A_all))
# print("A_all.shape = %s" % str(A_all.shape))
#
# rank_A_all = np.linalg.matrix_rank(A_all)
# print("rank_A_all = %d" % rank_A_all)
#
# A_u, A_s, A_vh = np.linalg.svd(A_all)
# print()
# # np.set_printoptions(suppress=True, precision=4)
# print("A_s = \n%s" % str(A_s))
# # np.set_printoptions(suppress=False, precision=8)
# print()
#
# # basis of the null space of A_all
# null_A_all_basis = A_vh[rank_A_all:,:].T
# np.set_printoptions(suppress=True)
# print("null_A_all_basis = \n%s" % str(null_A_all_basis))
# np.set_printoptions(suppress=False)
#
#
# # Solve phi
# # phi_est = np.linalg.inv(A_all.T @ A_all) @ (A_all.T) @ B_all # Note: This equation only apply when A_all is full rank (i.e. rank(A_all) == 11), or it will fail
# phi_est = np.linalg.pinv(A_all) @ B_all
# print("phi_est = \n%s" % str(phi_est))



# Solve by iteration
#--------------------------------------#
# Initial guess, not neccessaryly unit vector!!
phi_3_est = np.array([-1.0, -1.0, -1.0]).reshape((3,1))
step_alpha = 1.0 # 0.5
num_it = 3
# Iteration
k_it = 0
print("---")
while k_it < num_it:
     k_it += 1
     print("!!!!!!!!!!!!!!!!!!!!!!>>>>> k_it = %d" % k_it)
     # Generate Delta_i(k-1) and A(k-1)
     Delta_all, A_all = get_Delta_A_all(np_point_3d_dict, np_point_image_dict, phi_3_est)
     #-------------------------#
     # print("A_all = \n%s" % str(A_all))
     print("A_all.shape = %s" % str(A_all.shape))
     rank_A_all = np.linalg.matrix_rank(A_all)
     print("rank_A_all = %d" % rank_A_all)
     A_u, A_s, A_vh = np.linalg.svd(A_all)
     # np.set_printoptions(suppress=True, precision=4)
     print("A_s = \n%s" % str(A_s))
     # np.set_printoptions(suppress=False, precision=8)
     #-------------------------#

     # Solve for phi
     #-------------------------#
     phi_est = np.linalg.pinv(A_all) @ B_all
     print("phi_est = \n%s" % str(phi_est))
     # residule
     _res = (A_all @ phi_est) - B_all
     # print("_res = \n%s" % str(_res))
     print("norm(_res) = %f" % np.linalg.norm(_res))
     #-------------------------#

     #-------------------------#
     phi_1_est = phi_est[0:3,:]
     phi_2_est = phi_est[3:6,:]
     print("phi_1_est = \n%s" % str(phi_1_est))
     print("phi_2_est = \n%s" % str(phi_2_est))
     norm_phi_1_est = np.linalg.norm(phi_1_est)
     norm_phi_2_est = np.linalg.norm(phi_2_est)
     print("norm_phi_1_est = %f" % norm_phi_1_est)
     print("norm_phi_2_est = %f" % norm_phi_2_est)
     #-------------------------#
     # Update phi_3_est
     phi_3_est_uni = unit_vec( (1.0-step_alpha)*phi_3_est + step_alpha*unit_vec( np.cross(phi_est[0:3,:].T, phi_est[3:6,:].T).T ))
     norm_phi_3_est = 0.5*(norm_phi_1_est + norm_phi_2_est)
     phi_3_est = norm_phi_3_est * phi_3_est_uni
     print("phi_3_est = \n%s" % str(phi_3_est))
     print("norm_phi_3_est = %f" % norm_phi_3_est)

     # Test
     #---------------------------------#
     phi_1_est = phi_est[0:3,:]
     phi_2_est = phi_est[3:6,:]
     Gamma_list = [phi_1_est.T, phi_2_est.T, phi_3_est.T]
     np_Gamma_est = np.vstack(Gamma_list)
     print("np_Gamma_est = \n%s" % str(np_Gamma_est))
     G_u, G_s, G_vh = np.linalg.svd(np_Gamma_est)
     # print("G_u = \n%s" % str(G_u))
     print("G_s = \n%s" % str(G_s))
     # print("G_vh = \n%s" % str(G_vh))
     G_D = np.linalg.det(G_u @ G_vh)
     # print("G_D = %f" % G_D)
     # Reconstruct R
     np_R_est = G_u @ np.diag([1.0, 1.0, G_D]) @ G_vh
     print("np_R_est = \n%s" % str(np_R_est))
     # Convert to Euler angle
     Euler_angle_est = get_Euler_from_rotation_matrix(np_R_est)
     print("(roll, yaw, pitch) \t\t= %s" % str( np.rad2deg(Euler_angle_est) ) )
     # roll_est, yaw_est, pitch_est = Euler_angle_est
     # Reconstruct t vector
     # t3_est = 1.0 / G_s[0]
     t3_est = 1.0 / G_s[1] # Accurate?
     # t3_est = 1.0 / np.average(G_s)
     print("t3_est = %f" % t3_est)
     np_t_est = np.vstack((phi_est[6:8,:], 1.0)) * t3_est
     print("np_t_est = \n%s" % str(np_t_est))
     #---------------------------------#
     # end Test
     print("---")

#--------------------------------------#

print()
print("phi_est = \n%s" % str(phi_est))
print("phi_3_est = \n%s" % str(phi_3_est))
print()

# #
# _x_hat, _res, _rank, _sv = np.linalg.lstsq(A_all, B_all, rcond=None) # Solve the linear equation#
# print("_x_hat = \n%s" % str(_x_hat))
# print("_res = \n%s" % str(_res))
# print("_rank = \n%s" % str(_rank))
# print("_sv = \n%s" % str(_sv))
# print()
# #

phi_1_est = phi_est[0:3,:]
phi_2_est = phi_est[3:6,:]
# phi_3_est = phi_3_est
print("phi_1_est = \n%s" % str(phi_1_est))
print("phi_2_est = \n%s" % str(phi_2_est))
print("phi_3_est = \n%s" % str(phi_3_est))
print()
norm_phi_1_est = np.linalg.norm(phi_1_est)
norm_phi_2_est = np.linalg.norm(phi_2_est)
norm_phi_3_est = np.linalg.norm(phi_3_est)
print("norm_phi_1_est = %f" % norm_phi_1_est)
print("norm_phi_2_est = %f" % norm_phi_2_est)
print("norm_phi_3_est = %f" % norm_phi_3_est)
print()
phi_1_est_uni = phi_1_est / norm_phi_1_est
phi_2_est_uni = phi_2_est / norm_phi_2_est
phi_3_est_uni = phi_3_est / norm_phi_3_est
print("phi_1_est_uni = \n%s" % str(phi_1_est_uni))
print("phi_2_est_uni = \n%s" % str(phi_2_est_uni))
print("phi_3_est_uni = \n%s" % str(phi_3_est_uni))
print()

# # Fix phi_3
# phi_3_est_uni_fix = np.cross(phi_1_est_uni.T, phi_2_est_uni.T).T
# # phi_3_est = phi_3_est_uni_fix * (0.5*(norm_phi_1_est + norm_phi_2_est))
# # phi_3_est = phi_3_est_uni_fix * (norm_phi_1_est + norm_phi_2_est + norm_phi_3_est)/3.0
# phi_3_est = phi_3_est_uni_fix * norm_phi_3_est
# # phi_3_est = 0.5*phi_3_est + 0.5*phi_3_est_uni_fix*(0.5*(norm_phi_1_est + norm_phi_2_est))
# print("phi_3_est_uni_fix (Fix by phi_1_est and phi_2_est) = \n%s" % str(phi_3_est_uni_fix))
# print("phi_3_est (Fix by phi_1_est and phi_2_est) = \n%s" % str(phi_3_est))
# print()

# Gamma_list = [phi_est[0:3,:].T, phi_est[3:6,:].T, phi_est[6:9,:].T]
Gamma_list = [phi_1_est.T, phi_2_est.T, phi_3_est.T]
np_Gamma_est = np.vstack(Gamma_list)
print("np_Gamma_est = \n%s" % str(np_Gamma_est))

G_u, G_s, G_vh = np.linalg.svd(np_Gamma_est)
print("G_u = \n%s" % str(G_u))
print("G_s = \n%s" % str(G_s))
print("G_vh = \n%s" % str(G_vh))
G_D = np.linalg.det(G_u @ G_vh)
print("G_D = %f" % G_D)

# np_Gamma_est_div_G_s_1 = np_Gamma_est / G_s[1]
# print("np_Gamma_est_div_G_s_1 = \n%s" % str(np_Gamma_est_div_G_s_1))



def cay_trans(A_in):
    I = np.eye(3)
    return ( np.linalg.inv(I + A_in) @ (I - A_in) )

def cay_cay_op(R_in):
    return cay_trans(cay_trans(R_in))

def fix_R_cay_skew_cay(R_in):
    return cay_trans(get_skew(cay_trans(R_in)))

def fix_R_polar_decomposition(R_in):
    return (R_in @ np.linalg.inv(sp_lin.sqrtm(R_in.T @ R_in)))


# Reconstruct R
#--------------------------------------------------------#
np_R_est = G_u @ np.diag([1.0, 1.0, G_D]) @ G_vh
# np_R_est = cay_cay_op(np_Gamma_est)
# np_R_est = fix_R_cay_skew_cay(np_Gamma_est)
# np_R_est = fix_R_polar_decomposition(np_Gamma_est)
print("np_R_est = \n%s" % str(np_R_est))
# Convert to Euler angle
Euler_angle_est = get_Euler_from_rotation_matrix(np_R_est)
print("(roll, yaw, pitch) \t\t= %s" % str( np.rad2deg(Euler_angle_est) ) )
roll_est, yaw_est, pitch_est = Euler_angle_est
#--------------------------------------------------------#

# Reconstruct t vector
# t3_est = 1.0 / G_s[0]
t3_est = 1.0 / G_s[1] # Accurate?
# t3_est = 1.0 / np.average(G_s)
print("t3_est = %f" % t3_est)
np_t_est = np.vstack((phi_est[9:11,:], 1.0)) * t3_est
print("np_t_est = \n%s" % str(np_t_est))

# Compare with ground truth
#-------------------------------------------------#
print()
print("-"*35)
print("np_R_GT = \n%s" % str(np_R_GT))
print("(roll_GT, yaw_GT, pitch_GT) \t= %s" % str( np.rad2deg((roll_GT, yaw_GT, pitch_GT)) ) )
print("np_t_GT = \n%s" % str(np_t_GT))
print("-"*35)

# Error
#------------------------#
print()
print("-"*60)
# Depth
np_t_err = np_t_est - np_t_GT
depth_err = np_t_err[2,0]
print("np_t_err = \n%s" % str(np_t_err))
print("depth_err = %f" % depth_err)
# Orientation
roll_err = roll_est - roll_GT
yaw_err = yaw_est - yaw_GT
pitch_err = pitch_est - pitch_GT
print("(roll_err, yaw_err, pitch_err) \t= %s" % str( np.rad2deg((roll_err, yaw_err, pitch_err)) ) )
print("-"*60)
#------------------------#
