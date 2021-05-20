import numpy as np
# import scipy.linalg as sp_lin
import copy


class PNP_SOLVER_A2_M3(object):
    '''
    '''
    def __init__(self, np_K_camera_est, point_3d_dict, pattern_scale=1.0, verbose=False):
        '''
        '''
        self.verbose = True
        self.np_K_camera_est = copy.deepcopy(np_K_camera_est)

        # LM in face local frame
        self.pattern_scale = pattern_scale
        self.point_3d_dict = copy.deepcopy(point_3d_dict)
        # Convert to numpy vector, shape: (3,1)
        # Applying the scale as well
        self.np_point_3d_dict = dict()
        for _k in self.point_3d_dict:
            self.np_point_3d_dict[_k] = np.array(self.point_3d_dict[_k]).reshape((3,1))
            self.np_point_3d_dict[_k] *= pattern_scale # Multiply the scale
            # # Tilting or other correction
            # _R_correct = self.get_rotation_matrix_from_Euler(0.0, 0.0, 10.0, is_degree=True) # 15 deg up
            # _t_correct = np.array([[0.0, 0.0, 0.0]]).T
            # self.np_point_3d_dict[_k] = self.transform_3D_point(self.np_point_3d_dict[_k], _R_correct, _t_correct)
        # self.lib_print(self.np_point_3d_dict)

        # Backup
        self.np_point_3d_pretransfer_dict = copy.deepcopy(self.np_point_3d_dict)

        # test, pre-transfer
        #---------------------------#
        self.is_using_pre_transform = False
        # self.is_using_pre_transform = True
        self.pre_trans_R_a_h = np.eye(3)
        self.pre_trans_t_a_h = np.array([[0.0, 0.0, -0.5]]).T
        # self.pre_trans_R_a_h = self.get_rotation_matrix_from_Euler(0.0, 0.0, 45.0, is_degree=True)
        # self.pre_trans_t_a_h = np.array([[0.0, 0.0, 0.0]]).T
        if self.is_using_pre_transform:
            for _k in self.np_point_3d_pretransfer_dict:
                self.np_point_3d_pretransfer_dict[_k] = self.transform_3D_point(self.np_point_3d_pretransfer_dict[_k], self.pre_trans_R_a_h, self.pre_trans_t_a_h)
        # For storing the estimated result of R_ca and t_ca
        self.np_R_c_a_est = np.eye(3)
        self.np_t_c_a_est = np.zeros((3,1))
        #---------------------------#


        self.lib_print("-"*35)
        self.lib_print("3D points in local coordinate:")
        self.lib_print("pattern_scale = %f" % pattern_scale)
        for _k in self.point_3d_dict:
            # self.lib_print("%s:\n%s" % (_k, str(self.np_point_3d_dict[_k])))
            # self.lib_print("%s:\n%s" % (_k, str(self.np_point_3d_pretransfer_dict[_k])))
            np.set_printoptions(suppress=True, precision=2)
            print("%s:%sp_3D=%s.T | p_3D_pretrans=%s.T" %
                (   _k,
                    " "*(12-len(_k)),
                    str(self.np_point_3d_dict[_k].T),
                    str(self.np_point_3d_pretransfer_dict[_k].T)
                )
            )
            np.set_printoptions(suppress=False, precision=8)
        self.lib_print("-"*35)
        #

        self.verbose = verbose

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
        # update_phi_3_method = 1
        # update_phi_3_method = 2
        update_phi_3_method = 0
        # Iteration
        k_it = 0
        self.lib_print("---")
        #
        W_all_diag_old = np.ones((B_all.shape[0],))
        res_old = np.ones(B_all.shape)
        res_norm = 10*3
        #
        while k_it < num_it:
            k_it += 1
            self.lib_print("!!!!!!!!!!!!!!!!!!!!!!>>>>> k_it = %d" % k_it)
            # Generate Delta_i(k-1) and A(k-1)
            # Delta_all, A_all = self.get_Delta_A_all(self.np_point_3d_dict, np_point_image_dict, phi_3_est)
            Delta_all, A_all = self.get_Delta_A_all(self.np_point_3d_pretransfer_dict, np_point_image_dict, phi_3_est)
            #-------------------------#
            # self.lib_print("A_all = \n%s" % str(A_all))
            self.lib_print("A_all.shape = %s" % str(A_all.shape))
            rank_A_all = np.linalg.matrix_rank(A_all)
            self.lib_print("rank_A_all = %d" % rank_A_all)
            A_u, A_s, A_vh = np.linalg.svd(A_all)
            # np.set_printoptions(suppress=True, precision=4)
            self.lib_print("A_s = %s" % str(A_s))
            # np.set_printoptions(suppress=False, precision=8)
            #-------------------------#

            # Solve for phi
            #-------------------------#
            # phi_est = np.linalg.inv(A_all.T @ A_all) @ A_all.T @ B_all
            #
            # # W_all_diag = np.ones((B_all.shape[0],))
            # _res_old_unit = self.unit_vec(res_old)
            # # W_all_diag = 1.0/(0.001 +  np.squeeze(_res_old_unit)**2)
            # # W_all_diag = 0.5 * W_all_diag_old + 0.5 / (0.001 +  np.squeeze(_res_old_unit)**2)
            # _res_MAE = np.average(np.abs(_res_old_unit))
            # W_all_diag = 1.0/(0.001 +  np.squeeze(np.abs(_res_old_unit) - _res_MAE)**2)
            # #
            # W_all_diag_old = copy.deepcopy(W_all_diag)
            # self.lib_print("W_all_diag = \n%s" % str(W_all_diag))
            # W_all = np.diag(W_all_diag)
            # # self.lib_print("W_all = \n%s" % str(W_all))
            # phi_est = np.linalg.inv(A_all.T @ W_all @ A_all) @ A_all.T @ W_all @ B_all
            #
            phi_est = np.linalg.pinv(A_all) @ B_all
            self.lib_print("phi_est = \n%s" % str(phi_est))
            # residule
            # _res = (A_all @ phi_est) - B_all
            _res = B_all - (A_all @ phi_est)
            res_old = copy.deepcopy(_res)
            self.lib_print("_res = %s.T" % str(_res.T))
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
                if self.verbose:
                    # np_R_est, np_t_est, t3_est = reconstruct_R_t_m1(phi_est, phi_3_est)
                    np_R_est, np_t_est, t3_est = self.reconstruct_R_t_m1(phi_est, phi_3_est_new)
                    # np_R_est, np_t_est, t3_est = reconstruct_R_t_m3(phi_est, phi_3_est_new)
                    # Save the computing power
                else:
                    pass # Save the computing power
                #---------------------------------#
                # end Test
            elif update_phi_3_method == 2: # update_phi_3_method == 2
                # First reconstructing R, necessary for this method
                np_R_est, np_t_est, t3_est = self.reconstruct_R_t_m2(phi_est, phi_3_est)
                # Then, update phi_3_est
                phi_3_est_new, norm_phi_3_est = self.update_phi_3_est_m2(np_R_est, t3_est, phi_3_est, step_alpha)
            else: # update_phi_3_method == 0
                # First reconstructing R
                np_R_est, np_t_est, t3_est = self.reconstruct_R_t_block_reconstruction(phi_est, phi_3_est)
                # Then, update phi_3_est
                phi_3_est_new, norm_phi_3_est = self.update_phi_3_est_m2(np_R_est, t3_est, phi_3_est, step_alpha)
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
        elif update_phi_3_method == 2:
            np_R_est, np_t_est, t3_est = self.reconstruct_R_t_m2(phi_est, phi_3_est)
        else:
            np_R_est, np_t_est, t3_est = self.reconstruct_R_t_block_reconstruction(phi_est, phi_3_est)
        # self.lib_print("np_R_est = \n%s" % str(np_R_est))

        # test, pre-transfer
        #---------------------------#
        self.np_R_c_a_est = copy.deepcopy(np_R_est)
        self.np_t_c_a_est = copy.deepcopy(np_t_est)
        if self.is_using_pre_transform:
            np_R_c_h_est = np_R_est @ self.pre_trans_R_a_h # R_ch = R_ca @ R_ah
            np_t_c_h = np_R_est @ self.pre_trans_t_a_h + np_t_est # t_ch = R_ca @ t_ah + t_ca
            # Overwrite output
            np_R_est = copy.deepcopy(np_R_c_h_est)
            np_t_est = copy.deepcopy(np_t_c_h)
            t3_est = np_t_est[2,0]
        #---------------------------#

        # Convert to Euler angle
        Euler_angle_est = self.get_Euler_from_rotation_matrix(np_R_est, verbose=False, is_degree=True)
        # self.lib_print("(roll, yaw, pitch) \t\t= %s" % str( np.rad2deg(Euler_angle_est) ) )
        self.lib_print("(roll, yaw, pitch) \t\t= %s" % str( Euler_angle_est )  ) # Already in degree
        roll_est, yaw_est, pitch_est = Euler_angle_est
        #
        # self.lib_print("t3_est = %f" % t3_est)
        # self.lib_print("np_t_est = \n%s" % str(np_t_est))
        #--------------------------------------------------------#

        # Note: Euler angles are in degree
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
            if Delta_i < 0:
                Delta_i = -1.0*_eps
            else: # Note: np.sign(0.0) --> 0.0, which is not preferred.
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
        # # test, add the perpendicular condition
        # #---------------------------------#
        # A_i_list.append( np.hstack([phi_3_est.T, np.zeros((1,3)), np.zeros((1,2))]))
        # A_i_list.append( np.hstack([np.zeros((1,3)), phi_3_est.T, np.zeros((1,2))]))
        # #---------------------------------#
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
        # # test, add the perpendicular condition
        # #---------------------------------#
        # B_i_list.append( np.zeros((2,1)))
        # #---------------------------------#
        B_all = np.vstack(B_i_list)
        return B_all

    def update_phi_3_est_m1(self, phi_1_est, norm_phi_1_est, phi_2_est, norm_phi_2_est, phi_3_est, step_alpha=1.0):
        '''
        '''
        # Update phi_3_est
        phi_3_est_uni = self.unit_vec( (1.0-step_alpha)*phi_3_est + step_alpha*self.unit_vec( np.cross(phi_1_est.T, phi_2_est.T).T ))
        # norm_phi_3_est = 0.5*(norm_phi_1_est + norm_phi_2_est)
        # norm_phi_3_est = (0.5*(norm_phi_1_est**2 + norm_phi_2_est**2))**0.5
        # norm_phi_3_est = min( norm_phi_1_est, norm_phi_2_est)
        norm_phi_3_est = max( norm_phi_1_est, norm_phi_2_est)
        # norm_phi_3_est = 0.83333333333 # Ground truth
        phi_3_est_new = norm_phi_3_est * phi_3_est_uni
        self.lib_print("phi_3_est_new = \n%s" % str(phi_3_est_new))
        self.lib_print("norm_phi_3_est = %f" % norm_phi_3_est)
        return (phi_3_est_new, norm_phi_3_est)

    def update_phi_3_est_m2(self, np_R_est, t3_est, phi_3_est, step_alpha=1.0):
        '''
        '''
        # Update phi_3_est
        phi_3_est_uni = np_R_est[2,:].reshape((3,1))
        # norm_phi_3_est = 0.5*(norm_phi_1_est + norm_phi_2_est)
        # norm_phi_3_est = min( norm_phi_1_est, norm_phi_2_est)
        norm_phi_3_est = 1.0/t3_est
        phi_3_est_new = norm_phi_3_est * phi_3_est_uni
        # if step_alpha != 1.0:
        #     # Asymptotically update
        #     phi_3_est_new = phi_3_est + step_alpha * (phi_3_est_new - phi_3_est)
        #
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
        Euler_angle_est = self.get_Euler_from_rotation_matrix(np_R_est, is_degree=True)
        # self.lib_print("(roll, yaw, pitch) \t\t= %s" % str( np.rad2deg(Euler_angle_est) ) )
        self.lib_print("(roll, yaw, pitch) \t\t= %s" % str( Euler_angle_est ) ) # Note: Euler angles are in degree.
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
        # value_G = np.linalg.norm(np_Gamma_est, ord=-2)
        value_G = np.linalg.norm(np_Gamma_est, ord=2)
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
        Euler_angle_est = self.get_Euler_from_rotation_matrix(np_R_est, is_degree=True)
        # self.lib_print("(roll, yaw, pitch) \t\t= %s" % str( np.rad2deg(Euler_angle_est) ) )
        self.lib_print("(roll, yaw, pitch) \t\t= %s" % str( Euler_angle_est ) ) # Note: Euler angles are in degree
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
        Euler_angle_est = self.get_Euler_from_rotation_matrix(np_R_est, is_degree=True)
        # self.lib_print("(roll, yaw, pitch) \t\t= %s" % str( np.rad2deg(Euler_angle_est) ) )
        self.lib_print("(roll, yaw, pitch) \t\t= %s" % str( Euler_angle_est ) ) # Note: Euler angles are in degree.
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

    def reconstruct_R_t_block_reconstruction(self, phi_est, phi_3_est):
    # def reconstruct_R_t_block_reconstruction(self, phi_est):
        '''
        Reconstruct the G = gamma*R using only the 2x2 block of the element in G
        while solving the scale: gamma at the same time.

        G = | K         beta |
            | alpha.T   c    |
          = gamma * R
        '''
        self.lib_print()


        # Note: In current version, we still construct the full Gamma matrix for investigation purpose.
        #       This action is actually redundant, and the phi_3_est is not required in the following reconstruction process.
        phi_1_est = phi_est[0:3,:]
        phi_2_est = phi_est[3:6,:]
        Gamma_list = [phi_1_est.T, phi_2_est.T, phi_3_est.T]
        # Gamma_list = [phi_1_est.T, phi_2_est.T, np.zeros((1,3))]
        np_Gamma_est = np.vstack(Gamma_list)
        self.lib_print("np_Gamma_est = \n%s" % str(np_Gamma_est))
        # Balance
        # _gc0 = np_Gamma_est[:,0].reshape((3,1))
        # _gc1 = np_Gamma_est[:,1].reshape((3,1))
        # _gc2 = np_Gamma_est[:,2].reshape((3,1))
        # _ngc0 = np.linalg.norm(_gc0)
        # _ngc1 = np.linalg.norm(_gc1)
        # _ngc2 = np.linalg.norm(_gc2)
        # _ngca = np.average([_ngc0, _ngc1, _ngc2])
        # # _gc0_n = _gc0 * _ngca/_ngc0
        # # _gc1_n = _gc1 * _ngca/_ngc1
        # # _gc2_n = _gc2 * _ngca/_ngc2
        # _gc0_n = _gc0 + (_ngca-_ngc0)/_ngca * _gc2
        # _gc1_n = _gc1 + (_ngca-_ngc1)/_ngca * _gc2
        # _gc2_n = _gc2
        # np_Gamma_est = np.hstack( [_gc0_n, _gc1_n, _gc2_n] )
        # self.lib_print("np_Gamma_est = \n%s" % str(np_Gamma_est))

        '''
        Current version: Fixed on left-top block of G ( <-- the "capital Gamma" matrix)
        TODO:
        - Find a method to choose the most significant 4 elements.
        - Use row/column permutation to construct a matrix with left-top block is known.
        - After solving the G matrix, permute back to form the final matrix.
        '''
        _K = np_Gamma_est[0:2,0:2] # left-top block, the known
        # _K = np.vstack( (phi_1_est[0:2,:].T, phi_2_est[0:2,:].T) ) # left-top block, the known
        _KTK = _K.T @ _K
        self.lib_print("_KTK = \n%s" % str(_KTK))
        _k1 = _KTK[0,0]
        _k2 = _KTK[0,1]
        _k3 = _KTK[1,1]

        # Solve the _gamma
        #--------------------------------#
        _D = (_k1 - _k3)**2 + 4*_k2**2
        self.lib_print("_D = %f" % _D)
        _gamma_2 = 0.5*( (_k1 + _k3) + np.sqrt(_D) )
        _gamma = np.sqrt(_gamma_2) # This, in our case, is positive.
        self.lib_print("(_gamma_2, _gamma) = %s" % str((_gamma_2, _gamma)) )
        #--------------------------------#

        # Reconstruct the _beta_se vector
        #--------------------------------#
        _e_2 = _k2**2 / (_gamma_2 - _k3)
        _e_se = np.sqrt(_e_2) # !! What is teh sign?
        self.lib_print("(_e_2, _e_se) = %s" % str((_e_2, _e_se)) )
        _d_se = -(_k2/_e_se)
        self.lib_print("_d_se = %f" % _d_se )
        _beta_se = np.array([[_e_se, _d_se]]).T
        self.lib_print("_beta_se = \n%s" % str(_beta_se))
        #--------------------------------#

        # Solve the c and _alpha_se
        #--------------------------------#
        _delta_se = np.linalg.inv(_K.T) @ _beta_se
        self.lib_print("_delta_se = \n%s" % str(_delta_se))
        # _c_2 = _gamma_2 / (_delta_se.T @ _delta_se + 1.0)
        # _c_sc = np.sqrt(_c_2) # !! What is teh sign?
        # self.lib_print("(_c_2, _c_sc) = %s" % str((_c_2, _c_sc)) )
        _c = np.linalg.det(_K) / _gamma # Note: no sign issue here!!
        self.lib_print("_c = %f" % _c )
        _alpha_se = -(_c * _delta_se)
        self.lib_print("_alpha_se = \n%s" % str(_alpha_se))
        #--------------------------------#

        # Determin the sign of _e (se) from the original phi_est
        #--------------------------------#
        # if ( np.sign(_alpha_se[0]) == np.sign(np_Gamma_est[0,2]) ):
        #     _se = 1.0
        # else:
        #     _se = -1.0

        # See which one is closer to the leat-squared solution, _alpha_se or (-_alpha_se)
        _alpha_lsq = np_Gamma_est[0:2, 2]
        _se = np.sign(_alpha_se[:,0].dot(_alpha_lsq))
        # if (_alpha_se[:,0].dot(_alpha_lsq)) < 0.0:
        #     _se = -1.0
        # else:
        #     _se = 1.0
        #
        _alpha = _se * _alpha_se
        _beta = _se * _beta_se
        self.lib_print("_se = %f" % _se )
        self.lib_print("_alpha = \n%s" % str(_alpha))
        self.lib_print("_beta = \n%s" % str(_beta))
        #--------------------------------#

        # Reconstruct the G and R, using _K, _alpha_se, _beta_se, _c, _gamma
        #--------------------------------#
        # _G_row_0_hyp1 = np.hstack( (_K, _alpha_se) )
        # _G_row_1_hyp1 = np.hstack( (_beta_se.T, [[_c]]) )
        # _G_hyp1 = np.vstack( (_G_row_0_hyp1, _G_row_1_hyp1) )
        # # _det_G_hyp1 = np.linalg.det(_G_hyp1)
        # # self.lib_print("_det_G_hyp1 = %f" % _det_G_hyp1 )
        # _G_row_0_hyp2 = np.hstack( (_K, -_alpha_se) )
        # _G_row_1_hyp2 = np.hstack( (-_beta_se.T, [[_c]]) )
        # _G_hyp2 = np.vstack( (_G_row_0_hyp2, _G_row_1_hyp2) )
        # # _det_G_hyp2 = np.linalg.det(_G_hyp2)
        # # self.lib_print("_det_G_hyp2 = %f" % _det_G_hyp2 )
        # # np_Gamma_reconstruct = _G_hyp1
        # np_Gamma_reconstruct = _G_hyp2
        # self.lib_print("np_Gamma_reconstruct = \n%s" % str(np_Gamma_reconstruct))

        # Reconstruct the Gamma
        _G_row_0 = np.hstack( (_K, _alpha) )
        _G_row_1 = np.hstack( (_beta.T, [[_c]]) )
        np_Gamma_reconstruct = np.vstack( (_G_row_0, _G_row_1) )
        self.lib_print("np_Gamma_reconstruct = \n%s" % str(np_Gamma_reconstruct))
        #
        # G_u, G_s, G_vh = np.linalg.svd(np_Gamma_reconstruct)
        # self.lib_print("G_s = \n%s" % str(G_s))
        # G_D = np.linalg.det(G_u @ G_vh)
        # # self.lib_print("G_D = %f" % G_D)
        # np_R_est = G_u @ np.diag([1.0, 1.0, G_D]) @ G_vh # SVD reconstruct
        # self.lib_print("(svd) np_R_est = \n%s" % str(np_R_est))

        # Note: Since we solved the G,T G = gamma^2 equation, the reconstructed G is in good shape
        #       Reconstructing it again by SVD actually results in the same matrix.
        np_R_est = np_Gamma_reconstruct / _gamma # Method 1
        self.lib_print("[G/gamma] np_R_est = \n%s" % str(np_R_est))

        #
        # Convert to Euler angle
        Euler_angle_est = self.get_Euler_from_rotation_matrix(np_R_est, is_degree=True)
        # self.lib_print("(roll, yaw, pitch) \t\t= %s" % str( np.rad2deg(Euler_angle_est) ) )
        self.lib_print("(roll, yaw, pitch) \t\t= %s" % str( Euler_angle_est ) ) # Note: Euler angles are in degree.
        # roll_est, yaw_est, pitch_est = Euler_angle_est
        #--------------------------------#

        # The value_G is exactly equal to the _gamma
        value_G = _gamma
        self.lib_print("value_G = %f" % value_G)
        #

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
    # def get_rotation_matrix_from_Euler(self, roll, yaw, pitch, is_degree=False):
    #     '''
    #     roll, yaw, pitch --> R
    #
    #     is_degree - True: the angle unit is degree, False: the angle unit is rad
    #     '''
    #     # Mirror correction term
    #     #------------------------------#
    #     yaw = -yaw
    #     #------------------------------#
    #     if is_degree:
    #         roll = np.deg2rad(roll)
    #         yaw = np.deg2rad(yaw)
    #         pitch = np.deg2rad(pitch)
    #     c1 = np.cos(roll)
    #     s1 = np.sin(roll)
    #     c2 = np.cos(yaw)
    #     s2 = np.sin(yaw)
    #     c3 = np.cos(pitch)
    #     s3 = np.sin(pitch)
    #     R_roll_0 = np.array([[ c1*c2, (s1*c3 + c1*s2*s3), (s1*s3 - c1*s2*c3)]])
    #     R_roll_1 = np.array([[-s1*c2, (c1*c3 - s1*s2*s3), (c1*s3 + s1*s2*c3)]])
    #     R_roll_2 = np.array([[ s2,    -c2*s3,              c2*c3            ]])
    #     # self.lib_print("np.cross(R_roll_0, R_roll_1) = %s" % str(np.cross(R_roll_0, R_roll_1)))
    #     # self.lib_print("R_roll_2 = %s" % str(R_roll_2))
    #     _R01 = np.concatenate((R_roll_0, R_roll_1), axis=0)
    #     R = np.concatenate((_R01, R_roll_2), axis=0)
    #     return R



    # def get_Euler_from_rotation_matrix(self, R_in, verbose=True, is_degree=False):
    #     '''
    #     R --> roll, yaw, pitch
    #
    #     is_degree - True: the angle unit is degree, False: the angle unit is rad
    #     '''
    #     _eps = 10.0**-7
    #     # if np.linalg.norm([R_in[2,1], R_in[2,2]]) <= _eps: # Note: abs(c2) = np.linalg.norm([R_in[2,1], R_in[2,2]])
    #     if np.abs(np.pi/2.0 - np.arcsin(np.abs(R_in[2,0]))) <= _eps: # Faster
    #         # Singularity, "gimbal locked"
    #         yaw = np.sign(R_in[2,0]) * (np.pi/2.0) # theta_2
    #         # Assume pitch = 0.0 (theta_3 = 0.0)
    #         pitch = 0.0 # theta_3
    #         roll = np.arctan2(R_in[0,1], R_in[1,1]) # theta_1
    #     else:
    #         # Normal case
    #         roll = np.arctan2(-R_in[1,0], R_in[0,0]) # theta_1
    #         pitch = np.arctan2(-R_in[2,1], R_in[2,2]) # theta_3
    #         #
    #         c1 = np.cos(roll)
    #         c3 = np.cos(pitch)
    #         if np.abs(c1) > np.abs(c3):
    #             if verbose:
    #                 self.lib_print("Set c2 as (r11/c1)")
    #             c2 = R_in[0,0]/c1
    #         else:
    #             if verbose:
    #                 self.lib_print("Set c2 as (r33/c3)")
    #             c2 = R_in[2,2]/c3
    #         yaw = np.arctan2(R_in[2,0], c2) # theta_2
    #     # Convert to degree if needed
    #     if is_degree:
    #         roll = np.rad2deg(roll)
    #         yaw = np.rad2deg(yaw)
    #         pitch = np.rad2deg(pitch)
    #     # Mirror correction term
    #     #------------------------------#
    #     yaw = -yaw
    #     #------------------------------#
    #     return (roll, yaw, pitch)

    def get_rotation_matrix_from_Euler(self, roll, yaw, pitch, is_degree=False):
        '''
        (x) roll, yaw, pitch --> R
        roll, pitch, yaw --> R

        is_degree - True: the angle unit is degree, False: the angle unit is rad
        '''
        # Mirror correction term
        #------------------------------#
        yaw = -yaw
        #------------------------------#
        if is_degree:
            roll = np.deg2rad(roll)
            yaw = np.deg2rad(yaw)
            pitch = np.deg2rad(pitch)
        c1 = np.cos(roll)
        s1 = np.sin(roll)
        c2 = np.cos(yaw)
        s2 = np.sin(yaw)
        c3 = np.cos(pitch)
        s3 = np.sin(pitch)

        E_roll = np.array([[c1, s1, 0.0], [-s1, c1, 0.0], [0.0, 0.0, 1.0]])
        E_yaw = np.array([[c2, 0.0, -s2], [0.0, 1.0, 0.0], [s2, 0.0, c2]])
        E_pitch = np.array([[1.0, 0.0, 0.0], [0.0, c3, s3], [0.0, -s3, c3]])
        # Test the sequence
        # R = E_roll @ E_yaw @ E_pitch # version 1, (x)
        # R = E_pitch @ E_yaw @ E_roll  # (x) --> Known: first roll then pitch
        R = E_roll @ E_pitch @ E_yaw # --> Might be correct
        # R = E_yaw @ E_roll @ E_pitch # (x)
        return R

    def get_Euler_from_rotation_matrix(self, R_in, verbose=True, is_degree=False):
        '''
        R --> roll, pitch, yaw

        is_degree - True: the angle unit is degree, False: the angle unit is rad
        '''
        _eps = 10.0**-7
        # if np.linalg.norm([R_in[2,1], R_in[2,2]]) <= _eps: # Note: abs(c2) = np.linalg.norm([R_in[2,1], R_in[2,2]])
        if np.abs(np.pi/2.0 - np.arcsin(np.abs(R_in[2,0]))) <= _eps: # Faster
            # Singularity, "gimbal locked"
            theta_2 = np.sign(-R_in[2,1]) * (np.pi/2.0) # theta_2
            # Assume theta_3 = 0.0 (theta_3 = 0.0)
            theta_3 = 0.0 # theta_3
            theta_1 = np.arctan2(-R_in[1,0], R_in[0,0]) # theta_1
        else:
            # Normal case
            theta_1 = np.arctan2( R_in[0,1], R_in[1,1] ) # theta_1
            theta_3 = np.arctan2( R_in[2,0], R_in[2,2] ) # theta_3
            #
            c1 = np.cos(theta_1)
            c3 = np.cos(theta_3)
            if np.abs(c1) > np.abs(c3):
                if verbose:
                    self.lib_print("Set c2 as (r22/c1)")
                c2 = R_in[1,1]/c1
            else:
                if verbose:
                    self.lib_print("Set c2 as (r33/c3)")
                c2 = R_in[2,2]/c3
            theta_2 = np.arctan2(-R_in[2,1], c2) # theta_2
        #
        roll = theta_1
        pitch = theta_2
        yaw = theta_3
        # Convert to degree if needed
        if is_degree:
            roll = np.rad2deg(roll)
            yaw = np.rad2deg(yaw)
            pitch = np.rad2deg(pitch)
        # Mirror correction term
        #------------------------------#
        yaw = -yaw
        #------------------------------#
        return (roll, yaw, pitch)
    #-----------------------------------------------------------#

    def transform_3D_point(self, point_3D, R_in, t_in):
        '''
        Note: we don't use homogeneous expression here since the operation here is simple.
        Input:
        - point_3D: [[x,y,z]].T, shape=(3,1)
        - R_in: SO(3), shape=(3,1)
        - t_in: [[x,y,z]].T, shape=(3,1)
        Output:
        - point_3D_trans: [[x,y,z]].T, shape=(3,1)
        '''
        return (R_in @ point_3D + t_in)

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
        # dir_x = self.unit_vec(uv_x1 - uv_o)
        # dir_y = self.unit_vec(uv_y1 - uv_o)
        # dir_z = self.unit_vec(uv_z1 - uv_o)
        dir_x = (uv_x1 - uv_o)
        dir_y = (uv_y1 - uv_o)
        dir_z = (uv_z1 - uv_o)
        return (uv_o, dir_x, dir_y, dir_z)


    def perspective_projection_golden_landmarks(self, np_R, np_t, is_quantized=False, is_pretrans_points=False):
        '''
        Project the golden landmarks onto the image space using the given camera intrinsic.
        '''
        # [x,y,1].T, shape: (3,1)
        np_point_image_dict = dict()
        np_point_image_no_q_err_dict = dict()
        np_point_quantization_error_dict = dict()

        # Perspective Projection + quantization
        #--------------------------------------------------#
        # Not the self.np_point_3d_pretransfer_dict
        for _k in self.np_point_3d_dict:
            if is_pretrans_points:
                _point = self.np_point_3d_pretransfer_dict[_k]
            else:
                _point = self.np_point_3d_dict[_k]
            _np_point_image, _projection_no_q = self.perspective_projection(_point, self.np_K_camera_est, np_R, np_t, is_quantized=is_quantized)
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



def main():
    # Parameters and data
    point_3d_dict = dict()
    # Camera intrinsic matrix
    f_camera = 188.55 # 175.0
    fx_camera = f_camera
    fy_camera = f_camera
    xo_camera = 320/2.0
    yo_camera = 240/2.0
    np_K_camera_est = np.array([[fx_camera, 0.0, xo_camera], [0.0, fy_camera, yo_camera], [0.0, 0.0, 1.0]]) # Estimated
    print("np_K_camera_est = \n%s" % str(np_K_camera_est))
    #
    pnp_solver = PNP_SOLVER_A2_M3(np_K_camera_est, point_3d_dict, verbose=True)

    # test
    # roll = 10.0
    # yaw = 35.0
    # pitch = -25.0
    roll = 45.0
    yaw = 40.0
    pitch = -15.0
    print("(roll, yaw, pitch) = %s" % str((roll, yaw, pitch)))
    R_1 = pnp_solver.get_rotation_matrix_from_Euler( roll, yaw, pitch, is_degree=True)
    roll_1, yaw_1, pitch_1 = pnp_solver.get_Euler_from_rotation_matrix(R_1, is_degree=True)
    print("R_1 = \n %s" % str(R_1))
    print("(roll_1, yaw_1, pitch_1) = %s" % str((roll_1, yaw_1, pitch_1)))

if __name__ == '__main__':
    main()
