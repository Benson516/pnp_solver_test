import numpy as np
# import scipy.linalg as sp_lin
import copy


class PNP_SOLVER_A2_M3(object):
    '''
    '''
    def __init__(self, np_K_camera_est, point_3d_dict, verbose=False):
        '''
        '''
        self.verbose = verbose
        self.np_K_camera_est = copy.deepcopy(np_K_camera_est)

        # LM in face local frame
        self.point_3d_dict = copy.deepcopy(point_3d_dict)
        # Convert to numpy vector, shape: (3,1)
        self.np_point_3d_dict = dict()
        self.lib_print("-"*35)
        self.lib_print("3D points in local coordinate:")
        for _k in point_3d_dict:
            self.np_point_3d_dict[_k] = np.array(point_3d_dict[_k]).reshape((3,1))
            self.lib_print("%s:\n%s" % (_k, str(self.np_point_3d_dict[_k])))
        self.lib_print("-"*35)
        # self.lib_print(self.np_point_3d_dict)

    def lib_print(self, str=''):
        if self.verbose:
            print(str)

    # Solution
    #-----------------------------------------------------------#
    def solve_pnp(self, np_point_image_dict):
        '''
        For each image frame,
        return (np_R_est, np_t_est, t3_est, roll_est, yaw_est, pitch_est )
        '''
        # Form the problem for solving
        B_all = self.get_B_all(np_point_image_dict, self.np_K_camera_est)
        self.lib_print("B_all = \n%s" % str(B_all))
        self.lib_print("B_all.shape = %s" % str(B_all.shape))

        # Solve by iteration
        #--------------------------------------#
        # Initial guess, not neccessaryly unit vector!!
        # phi_3_est = np.array([-1.0, -1.0, -1.0]).reshape((3,1))
        phi_3_est = np.array([0.0, 0.0, 1.0]).reshape((3,1))
        phi_3_est_new = copy.deepcopy(phi_3_est)
        step_alpha = 1.0 # 0.5
        num_it = 3
        #
        # W_all_diag = np.ones((B_all.shape[0],))
        # # W_all_diag[8:10] *= 10**-10
        # W_all = np.diag(W_all_diag)
        # self.lib_print("W_all_diag = \n%s" % str(W_all_diag))
        #
        update_phi_3_method = 1
        # update_phi_3_method = 2
        # Iteration
        k_it = 0
        self.lib_print("---")
        res_norm = 10*3
        while k_it < num_it:
             k_it += 1
             self.lib_print("!!!!!!!!!!!!!!!!!!!!!!>>>>> k_it = %d" % k_it)
             # Generate Delta_i(k-1) and A(k-1)
             Delta_all, A_all = self.get_Delta_A_all(self.np_point_3d_dict, np_point_image_dict, phi_3_est)
             #-------------------------#
             # self.lib_print("A_all = \n%s" % str(A_all))
             self.lib_print("A_all.shape = %s" % str(A_all.shape))
             rank_A_all = np.linalg.matrix_rank(A_all)
             self.lib_print("rank_A_all = %d" % rank_A_all)
             A_u, A_s, A_vh = np.linalg.svd(A_all)
             # np.set_printoptions(suppress=True, precision=4)
             self.lib_print("A_s = \n%s" % str(A_s))
             # np.set_printoptions(suppress=False, precision=8)
             #-------------------------#

             # Solve for phi
             #-------------------------#
             # phi_est = np.linalg.inv(A_all.T @ A_all) @ A_all.T @ B_all
             # phi_est = np.linalg.inv(A_all.T @ W_all @ A_all) @ A_all.T @ W_all @ B_all
             phi_est = np.linalg.pinv(A_all) @ B_all
             self.lib_print("phi_est = \n%s" % str(phi_est))
             # residule
             # _res = (A_all @ phi_est) - B_all
             _res = B_all - (A_all @ phi_est)
             # self.lib_print("_res = \n%s" % str(_res))
             # _res_delta = _res - np_quantization_error_world_space_vec
             # self.lib_print("_res_delta = \n%s" % str(_res_delta))
             res_norm = np.linalg.norm(_res)
             self.lib_print("norm(_res) = %f" % res_norm)
             #-------------------------#

             #-------------------------#
             phi_1_est = phi_est[0:3,:]
             phi_2_est = phi_est[3:6,:]
             self.lib_print("phi_1_est = \n%s" % str(phi_1_est))
             self.lib_print("phi_2_est = \n%s" % str(phi_2_est))
             norm_phi_1_est = np.linalg.norm(phi_1_est)
             norm_phi_2_est = np.linalg.norm(phi_2_est)
             self.lib_print("norm_phi_1_est = %f" % norm_phi_1_est)
             self.lib_print("norm_phi_2_est = %f" % norm_phi_2_est)
             #
             self.lib_print("phi_3_est = \n%s" % str(phi_3_est_new))
             #-------------------------#

             if update_phi_3_method == 1:
                 # First update phi_3_est
                 phi_3_est_new, norm_phi_3_est = self.update_phi_3_est_m1(phi_1_est, norm_phi_1_est, phi_2_est, norm_phi_2_est, phi_3_est, step_alpha)
                 # Then, test (not necessary)
                 #---------------------------------#
                 # np_R_est, np_t_est, t3_est = reconstruct_R_t_m1(phi_est, phi_3_est)
                 np_R_est, np_t_est, t3_est = self.reconstruct_R_t_m1(phi_est, phi_3_est_new)
                 # np_R_est, np_t_est, t3_est = reconstruct_R_t_m3(phi_est, phi_3_est_new)

                 #---------------------------------#
                 # end Test
             else: # update_phi_3_method == 2
                 # First reconstructing R, necessary for this method
                 np_R_est, np_t_est, t3_est = self.reconstruct_R_t_m2(phi_est, phi_3_est)
                 # Then, update phi_3_est
                 phi_3_est_new, norm_phi_3_est = self.update_phi_3_est_m2(np_R_est, t3_est)
             # Real update of phi_3_est
             phi_3_est = copy.deepcopy(phi_3_est_new)
             self.lib_print("---")

        #--------------------------------------#

        self.lib_print()
        self.lib_print("phi_est = \n%s" % str(phi_est))
        self.lib_print("phi_3_est = \n%s" % str(phi_3_est))
        self.lib_print()
        # Reconstruct (R, t)
        #--------------------------------------------------------#
        if update_phi_3_method == 1:
            np_R_est, np_t_est, t3_est = self.reconstruct_R_t_m1(phi_est, phi_3_est)
            # np_R_est, np_t_est, t3_est = reconstruct_R_t_m3(phi_est, phi_3_est)
        else:
            np_R_est, np_t_est, t3_est = self.reconstruct_R_t_m2(phi_est, phi_3_est)
        # self.lib_print("np_R_est = \n%s" % str(np_R_est))
        # Convert to Euler angle
        Euler_angle_est = self.get_Euler_from_rotation_matrix(np_R_est, verbose=False)
        self.lib_print("(roll, yaw, pitch) \t\t= %s" % str( np.rad2deg(Euler_angle_est) ) )
        roll_est, yaw_est, pitch_est = Euler_angle_est
        #
        # self.lib_print("t3_est = %f" % t3_est)
        # self.lib_print("np_t_est = \n%s" % str(np_t_est))
        #--------------------------------------------------------#
        return (np_R_est, np_t_est, t3_est, roll_est, yaw_est, pitch_est, res_norm)

    #-----------------------------------------------------------#

    # Components of the solution
    #-----------------------------------------------------------#
    def get_Delta_i(self, theta_i, phi_3_est, id=None):
        '''
        theta_i=[[x,y,z]].T, shape=(3,1)
        phi_3_est = [[xu, yu, zu]].T, shape=(3,1)
        '''
        _theta_i_T = theta_i.reshape((1,3)) # Incase it's 1-D array, use reshape instead of theta_i.T
        _phi_3 = phi_3_est.reshape((3,1))
        Delta_i = (_theta_i_T @ _phi_3 + 1.0)
        _eps = 10**-7
        if np.abs(Delta_i) <= _eps:
            self.lib_print("Delta[%d] is too close to zero!!" % (id if id is not None else -1) )
            Delta_i = _eps
        return Delta_i

    def get_A_i(self, theta_i, Delta_i):
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
        A_i = np.concatenate((A_i_c01, np.eye(2)/Delta_i), axis=1)
        return A_i

    def get_Delta_A_all(self, np_point_3d_dict_in, np_point_image_dict_in, phi_3_est):
        '''
        '''
        # To be more realistic, use the image point to search
        A_i_list = list()
        Delta_i_list = list()
        for _id, _k in enumerate(np_point_image_dict_in):
            Delta_i = self.get_Delta_i(np_point_3d_dict_in[_k], phi_3_est, id=_id)
            Delta_i_list.append( Delta_i )
            A_i_list.append( self.get_A_i(np_point_3d_dict_in[_k], Delta_i) )
        Delta_all = np.vstack(Delta_i_list)
        A_all = np.vstack(A_i_list)
        return (Delta_all, A_all)

    def get_B_all(self, np_point_image_dict_in, K_in):
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

    def update_phi_3_est_m1(self, phi_1_est, norm_phi_1_est, phi_2_est, norm_phi_2_est, phi_3_est, step_alpha=1.0):
        '''
        '''
        # Update phi_3_est
        phi_3_est_uni = self.unit_vec( (1.0-step_alpha)*phi_3_est + step_alpha*self.unit_vec( np.cross(phi_1_est.T, phi_2_est.T).T ))
        norm_phi_3_est = 0.5*(norm_phi_1_est + norm_phi_2_est)
        # norm_phi_3_est = min( norm_phi_1_est, norm_phi_2_est)
        # norm_phi_3_est = 0.83333333333 # Ground truth
        phi_3_est_new = norm_phi_3_est * phi_3_est_uni
        self.lib_print("phi_3_est_new = \n%s" % str(phi_3_est_new))
        self.lib_print("norm_phi_3_est = %f" % norm_phi_3_est)
        return (phi_3_est_new, norm_phi_3_est)

    def update_phi_3_est_m2(self, np_R_est, t3_est):
        '''
        '''
        # Update phi_3_est
        phi_3_est_uni = np_R_est[2,:].reshape((3,1))
        # norm_phi_3_est = 0.5*(norm_phi_1_est + norm_phi_2_est)
        # norm_phi_3_est = min( norm_phi_1_est, norm_phi_2_est)
        norm_phi_3_est = 1.0/t3_est
        phi_3_est_new = norm_phi_3_est * phi_3_est_uni
        self.lib_print("phi_3_est_new = \n%s" % str(phi_3_est_new))
        self.lib_print("norm_phi_3_est = %f" % norm_phi_3_est)
        return (phi_3_est_new, norm_phi_3_est)

    def reconstruct_R_t_m1(self, phi_est, phi_3_est):
        '''
        - Use phi_3_est
        - Use G_2[2]
        '''
        # Test
        #---------------------------------#
        phi_1_est = phi_est[0:3,:]
        phi_2_est = phi_est[3:6,:]
        Gamma_list = [phi_1_est.T, phi_2_est.T, phi_3_est.T]
        # Gamma_list = [phi_1_est.T, phi_2_est.T, np.zeros((1,3))]
        np_Gamma_est = np.vstack(Gamma_list)
        self.lib_print("np_Gamma_est = \n%s" % str(np_Gamma_est))
        G_u, G_s, G_vh = np.linalg.svd(np_Gamma_est)
        # self.lib_print("G_u = \n%s" % str(G_u))
        self.lib_print("G_s = \n%s" % str(G_s))
        # self.lib_print("G_vh = \n%s" % str(G_vh))
        G_D = np.linalg.det(G_u @ G_vh)
        # self.lib_print("G_D = %f" % G_D)
        # Reconstruct R
        # np_R_est = np_Gamma_est
        np_R_est = G_u @ np.diag([1.0, 1.0, G_D]) @ G_vh
        self.lib_print("np_R_est = \n%s" % str(np_R_est))
        # Convert to Euler angle
        Euler_angle_est = self.get_Euler_from_rotation_matrix(np_R_est)
        self.lib_print("(roll, yaw, pitch) \t\t= %s" % str( np.rad2deg(Euler_angle_est) ) )
        # roll_est, yaw_est, pitch_est = Euler_angle_est
        # Reconstruct t vector
        # Get the "value" of G
        G_col_norm_vec = np.linalg.norm(np_Gamma_est, axis=0)
        G_row_norm_vec = np.linalg.norm(np_Gamma_est, axis=1)
        self.lib_print("G_col_norm_vec = %s" % str(G_col_norm_vec))
        self.lib_print("G_row_norm_vec = %s" % str(G_row_norm_vec))
        # value_G = G_s[0] # Note: G_s[0] = np.linalg.norm(np_Gamma_est, ord=2)
        # value_G = G_s[1] # Accurate?
        # value_G = G_s[2] # Accurate? Note: G_s[2] = np.linalg.norm(np_Gamma_est, ord=-2)
        # value_G = np.average(G_s)
        value_G = np.linalg.norm(np_Gamma_est, ord=-2)
        self.lib_print("value_G = %f" % value_G)
        #
        t3_est = 1.0 / value_G
        # t3_est = 1.0 / np.average(G_s)
        self.lib_print("t3_est = %f" % t3_est)
        np_t_est = np.vstack((phi_est[6:8,:], 1.0)) * t3_est
        self.lib_print("np_t_est = \n%s" % str(np_t_est))
        #---------------------------------#
        # end Test
        return (np_R_est, np_t_est, t3_est)

    def reconstruct_R_t_m2(self, phi_est, phi_3_est):
        '''
        - Not using phi_3_est
        - Use G_s[2]
        '''
        # Test
        #---------------------------------#
        phi_1_est = phi_est[0:3,:]
        phi_2_est = phi_est[3:6,:]
        # Gamma_list = [phi_1_est.T, phi_2_est.T, phi_3_est.T]
        Gamma_list = [phi_1_est.T, phi_2_est.T, np.zeros((1,3))]
        np_Gamma_est = np.vstack(Gamma_list)
        self.lib_print("np_Gamma_est = \n%s" % str(np_Gamma_est))
        G_u, G_s, G_vh = np.linalg.svd(np_Gamma_est)
        # self.lib_print("G_u = \n%s" % str(G_u))
        self.lib_print("G_s = \n%s" % str(G_s))
        # self.lib_print("G_vh = \n%s" % str(G_vh))
        G_D = np.linalg.det(G_u @ G_vh)
        # self.lib_print("G_D = %f" % G_D)
        # Reconstruct R
        np_R_est = G_u @ np.diag([1.0, 1.0, G_D]) @ G_vh
        self.lib_print("np_R_est = \n%s" % str(np_R_est))
        # Convert to Euler angle
        Euler_angle_est = self.get_Euler_from_rotation_matrix(np_R_est)
        self.lib_print("(roll, yaw, pitch) \t\t= %s" % str( np.rad2deg(Euler_angle_est) ) )
        # roll_est, yaw_est, pitch_est = Euler_angle_est
        # Reconstruct t vector
        # t3_est = 1.0 / G_s[0]
        t3_est = 1.0 / G_s[1] # Accurate?
        # t3_est = 1.0 / G_s[2] # Accurate?
        # t3_est = 1.0 / np.average(G_s)
        self.lib_print("t3_est = %f" % t3_est)
        np_t_est = np.vstack((phi_est[6:8,:], 1.0)) * t3_est
        self.lib_print("np_t_est = \n%s" % str(np_t_est))
        #---------------------------------#
        # end Test
        return (np_R_est, np_t_est, t3_est)

    def reconstruct_R_t_m3(self, phi_est, phi_3_est):
        '''
        - Use phi_3_est
        - Use G_2[2]
        '''
        # Test
        #---------------------------------#
        phi_1_est = phi_est[0:3,:]
        phi_2_est = phi_est[3:6,:]
        Gamma_list = [phi_1_est.T, phi_2_est.T, phi_3_est.T]
        # Gamma_list = [phi_1_est.T, phi_2_est.T, np.zeros((1,3))]
        np_Gamma_est = np.vstack(Gamma_list)
        # Get the "value" of G
        G_col_norm_vec = np.linalg.norm(np_Gamma_est, axis=0)
        G_row_norm_vec = np.linalg.norm(np_Gamma_est, axis=1)
        self.lib_print("G_col_norm_vec = %s" % str(G_col_norm_vec))
        self.lib_print("G_row_norm_vec = %s" % str(G_row_norm_vec))
        value_G = np.average(G_row_norm_vec)
        self.lib_print("value_G = %f" % value_G)

        # Normalize Gamma
        np_Gamma_est /= (G_row_norm_vec.reshape((3,1)))
        #
        self.lib_print("np_Gamma_est = \n%s" % str(np_Gamma_est))
        G_u, G_s, G_vh = np.linalg.svd(np_Gamma_est)
        # self.lib_print("G_u = \n%s" % str(G_u))
        self.lib_print("G_s = \n%s" % str(G_s))
        # self.lib_print("G_vh = \n%s" % str(G_vh))
        G_D = np.linalg.det(G_u @ G_vh)
        # self.lib_print("G_D = %f" % G_D)
        # Reconstruct R
        # np_R_est = np_Gamma_est
        np_R_est = G_u @ np.diag([1.0, 1.0, G_D]) @ G_vh
        self.lib_print("np_R_est = \n%s" % str(np_R_est))
        # Convert to Euler angle
        Euler_angle_est = self.get_Euler_from_rotation_matrix(np_R_est)
        self.lib_print("(roll, yaw, pitch) \t\t= %s" % str( np.rad2deg(Euler_angle_est) ) )
        # roll_est, yaw_est, pitch_est = Euler_angle_est
        # Reconstruct t vector
        t3_est = 1.0 / value_G
        # t3_est = 1.0 / np.average(G_s)
        self.lib_print("t3_est = %f" % t3_est)
        np_t_est = np.vstack((phi_est[6:8,:], 1.0)) * t3_est
        self.lib_print("np_t_est = \n%s" % str(np_t_est))
        #---------------------------------#
        # end Test
        return (np_R_est, np_t_est, t3_est)
    #-----------------------------------------------------------#

    #-----------------------------------------------------------#
    def get_rotation_matrix_from_Euler(self, roll, yaw, pitch):
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
        # self.lib_print("np.cross(R_roll_0, R_roll_1) = %s" % str(np.cross(R_roll_0, R_roll_1)))
        # self.lib_print("R_roll_2 = %s" % str(R_roll_2))
        _R01 = np.concatenate((R_roll_0, R_roll_1), axis=0)
        R = np.concatenate((_R01, R_roll_2), axis=0)
        return R

    def get_Euler_from_rotation_matrix(self, R_in, verbose=True):
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
                if verbose:
                    self.lib_print("Set c2 as (r11/c1)")
                c2 = R_in[0,0]/c1
            else:
                if verbose:
                    self.lib_print("Set c2 as (r33/c3)")
                c2 = R_in[2,2]/c3
            yaw = np.arctan2(R_in[2,0], c2) # theta_2
        return (roll, yaw, pitch)
    #-----------------------------------------------------------#

    def perspective_projection(self, np_point_3d, np_K_camera, np_R, np_t, is_quantized=False, is_returning_homogeneous_vec=True):
        '''
        input:
        - np_point_3d: [[x,y,z]].T
        output:
            if is_returning_homogeneous_vec:
                - np_point_image:  [[u, v, 1.0]].T
                - projection_no_q: [[u, v, 1.0]].T
            else:
                - np_point_image:  [u, v]
                - projection_no_q: [u, v]
        '''
        _np_point_3d = np_point_3d.reshape((3,1))
        _ray = np_K_camera @ (np_R @ _np_point_3d + np_t)
        # normalize
        projection_no_q = _ray/_ray[2,0]
        # Quantize
        if is_quantized:
            np_point_image = np.around(projection_no_q) # with quantization
        else:
            np_point_image = projection_no_q # no quantization
        if is_returning_homogeneous_vec:
            return (np_point_image, projection_no_q)
        else:
            return (np_point_image[0:2,0], projection_no_q[0:2,0])


    def perspective_projection_obj_axis(self, np_R, np_t):
        '''
        Project the unit vector of each axis onto the image space.
        output:
        - uv_o: [uo, vo]
        - dir_x: [dx_u, dx_y]
        - dir_y: [dy_u, dy_y]
        - dir_z: [dz_u, dz_y]
        '''
        # Calculate the control points of each axis (i.e. the point at distance 1.0 on each axis)
        # Note: For uv_0, this operation is eqivelent to "uv_o = (K @ np_t)[0:2,0] "
        uv_o, _ = self.perspective_projection(np.array([[0.0, 0.0, 0.0]]), self.np_K_camera_est, np_R, np_t, is_quantized=False, is_returning_homogeneous_vec=False)
        uv_x1, _ = self.perspective_projection(np.array([[1.0, 0.0, 0.0]]), self.np_K_camera_est, np_R, np_t, is_quantized=False, is_returning_homogeneous_vec=False)
        uv_y1, _ = self.perspective_projection(np.array([[0.0, 1.0, 0.0]]), self.np_K_camera_est, np_R, np_t, is_quantized=False, is_returning_homogeneous_vec=False)
        uv_z1, _ = self.perspective_projection(np.array([[0.0, 0.0, 1.0]]), self.np_K_camera_est, np_R, np_t, is_quantized=False, is_returning_homogeneous_vec=False)
        # Calculate the direction vector for each axis in the image space
        # Note: This operation is eqivelent to "dir_x = self.unit_vec(K @ np_R[:,0])"
        dir_x = self.unit_vec(uv_x1 - uv_o)
        dir_y = self.unit_vec(uv_y1 - uv_o)
        dir_z = self.unit_vec(uv_z1 - uv_o)
        return (uv_o, dir_x, dir_y, dir_z)


    def perspective_projection_golden_landmarks(self, np_R, np_t, is_quantized=False):
        '''
        Project the golden landmarks onto the image space using the given camera intrinsic.
        '''
        # [x,y,1].T, shape: (3,1)
        np_point_image_dict = dict()
        np_point_image_no_q_err_dict = dict()
        np_point_quantization_error_dict = dict()

        # Perspective Projection + quantization
        #--------------------------------------------------#
        for _k in self.np_point_3d_dict:
            _np_point_image, _projection_no_q = self.perspective_projection(self.np_point_3d_dict[_k], self.np_K_camera_est, np_R, np_t, is_quantized=is_quantized)
            np_point_image_dict[_k] = _np_point_image
            np_point_quantization_error_dict[_k] = (_np_point_image - _projection_no_q)
            np_point_image_no_q_err_dict[_k] = _projection_no_q
            # print("%s:\n%s" % (_k, str(np_point_image_dict[_k])))
            # print("%s:\n%s" % (_k, str(np_point_quantization_error_dict[_k])))
        #--------------------------------------------------#
        # Print
        # print("-"*35)
        # print("is_quantized = %s" % str(is_quantized))
        # print("2D points on image:")
        # for _k in np_point_image_dict:
        #     print("%s:%sp=%s.T | p_no_q_err=%s.T | q_e=%s.T" % (_k, " "*(12-len(_k)), str(np_point_image_dict[_k].T), str(np_point_image_no_q_err_dict[_k].T), str(np_point_quantization_error_dict[_k].T) ))
        #     # print("%s:\n%s" % (_k, str(np_point_image_dict[_k])))
        #     # print("%s:\n%s" % (_k, str(np_point_quantization_error_dict[_k])))
        # print("-"*35)
        return np_point_image_dict

    #-----------------------------------------------------------#
    def print_matrix_and_eigen_value(self, m_name, m_in, is_printing_eig_vec=False):
        w, v = np.linalg.eig(m_in)
        #
        np.set_printoptions(suppress=True)
        self.lib_print()
        self.lib_print("==== (Start) Eigen of %s ===" % m_name)
        self.lib_print("%s = \n%s" % (m_name, str(m_in)))
        self.lib_print("%s_eig_value = \n%s" % (m_name, str(w)))
        if is_printing_eig_vec:
            self.lib_print("%s_eig_vec = \n%s" % (m_name, str(v)))
        self.lib_print("====  (End)  Eigen of %s ===" % m_name)
        self.lib_print()
        np.set_printoptions(suppress=False)
        return (w, v)

    def print_matrix_and_SVD(self, m_name, m_in, is_printing_u_vh=False):
        u, s, vh = np.linalg.svd(m_in)
        _norm = np.linalg.norm(m_in)
        #
        np.set_printoptions(suppress=True)
        self.lib_print()
        self.lib_print("==== (Start) SVD of %s ===" % m_name)
        self.lib_print("%s = \n%s" % (m_name, str(m_in)))
        self.lib_print("||%s|| = %f" % (m_name, _norm))
        if is_printing_u_vh:
            self.lib_print("%s_U = \n%s" % (m_name, str(u)))
        self.lib_print("%s_S = \n%s" % (m_name, str(s)))
        if is_printing_u_vh:
            self.lib_print("%s_Vh = \n%s" % (m_name, str(vh)))
        self.lib_print("====  (End)  SVD of %s ===" % m_name)
        self.lib_print()
        np.set_printoptions(suppress=False)
        return (u, s, vh)
    #-----------------------------------------------------------#

    #-----------------------------------------------------------#
    def get_symmetry(self, A_in):
        return 0.5*(A_in + A_in.T)

    def get_skew(self, A_in):
        return 0.5*(self, A_in - A_in.T)

    def unit_vec(self, vec_in):
        _norm = np.linalg.norm(vec_in)
        if np.abs(_norm) <= 10**-7:
             _norm_inv = 1.0
             self.lib_print("_norm = %f" % _norm)
             self.lib_print("_norm approaches zeros!!")
        else:
             _norm_inv = 1.0/_norm
        return (vec_in * _norm_inv)
    #-----------------------------------------------------------#


    #-----------------------------------------------------------#
    def fix_R_svd(self, R_in):
        # self.lib_print("R_in = \n%s" % str(R_in))
        G_u, G_s, G_vh = np.linalg.svd(R_in)
        # self.lib_print("G_u = \n%s" % str(G_u))
        # self.lib_print("G_s = \n%s" % str(G_s))
        # self.lib_print("G_vh = \n%s" % str(G_vh))
        G_D = np.linalg.det(G_u @ G_vh)
        # self.lib_print("G_D = %f" % G_D)
        # Reconstruct R
        np_R_est = G_u @ np.diag([1.0, 1.0, G_D]) @ G_vh
        # self.lib_print("np_R_est = \n%s" % str(np_R_est))
        return np_R_est

    def cay_trans(self, A_in):
        I = np.eye(3)
        return ( np.linalg.inv(I + A_in) @ (I - A_in) )

    def cay_cay_op(self, R_in):
        return self.cay_trans(self.cay_trans(R_in))

    def fix_R_cay_skew_cay(self, R_in):
        return self.cay_trans(self.get_skew(self.cay_trans(R_in)))

    def fix_R_polar_decomposition(self, R_in):
        return (R_in @ np.linalg.inv(sp_lin.sqrtm(R_in.T @ R_in)))
    #-----------------------------------------------------------#
