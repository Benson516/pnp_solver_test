import numpy as np
# import scipy.linalg as sp_lin
import copy
import time


class PNP_SOLVER_A2_M3(object):
    '''
    '''
    def __init__(self, np_K_camera_est, point_3d_dict_list, pattern_scale_list=None, verbose=False):
        '''
        '''
        self.verbose = True
        self.np_K_camera_est = copy.deepcopy(np_K_camera_est)
        self.pattern_scale_list = pattern_scale_list
        self.point_3d_dict_list = copy.deepcopy(point_3d_dict_list)

        if self.pattern_scale_list is None:
            self.pattern_scale_list = [1.0 for _i in range(len(self.point_3d_dict_list))]
        elif len(self.pattern_scale_list) < len(self.point_3d_dict_list):
            # Padding the rest to zero
            self.pattern_scale_list += [1.0 for _i in range(len(self.point_3d_dict_list) - len(self.pattern_scale_list))]


        # test, pre-transfer
        #---------------------------#
        self.is_using_pre_transform = False
        # self.is_using_pre_transform = True
        self.pre_trans_R_a_h = np.eye(3)
        self.pre_trans_t_a_h = np.array([[0.0, 0.0, -0.5]]).T
        # self.pre_trans_R_a_h = self.get_rotation_matrix_from_Euler(0.0, 0.0, 45.0, is_degree=True)
        # self.pre_trans_t_a_h = np.array([[0.0, 0.0, 0.0]]).T

        # For storing the estimated result of R_ca and t_ca
        self.np_R_c_a_est = np.eye(3)
        self.np_t_c_a_est = np.zeros((3,1))
        #---------------------------#
        self.np_point_3d_dict_list = list()
        self.np_point_3d_pretransfer_dict_list = list()
        for _i in range(len(self.point_3d_dict_list)):
            _point_3d_dict = self.point_3d_dict_list[_i]
            _pattern_scale = self.pattern_scale_list[_i]
            _np_point_3d_dict, _np_point_3d_pretransfer_dict = self.get_np_point_3d_dict(_point_3d_dict, _pattern_scale, self.is_using_pre_transform, self.pre_trans_R_a_h, self.pre_trans_t_a_h)
            self.np_point_3d_dict_list.append(_np_point_3d_dict)
            self.np_point_3d_pretransfer_dict_list.append(_np_point_3d_pretransfer_dict)

        # Setup the global variables
        self.set_golden_pattern_id(0)
        # self.np_point_3d_dict = self.get_current_golden_pattern()
        # self.np_point_3d_pretransfer_dict = self.get_current_pretransfered_golden_pattern()

        # Setup the desire verbose status
        self.verbose = verbose

    def set_golden_pattern_id(self, id):
        self.current_golden_pattern_id = id

    def get_current_golden_pattern(self):
        return self.np_point_3d_dict_list[ self.current_golden_pattern_id ]

    def get_current_pretransfered_golden_pattern(self):
        return self.np_point_3d_pretransfer_dict_list[ self.current_golden_pattern_id ]

    def get_np_point_3d_dict(self, point_3d_dict, pattern_scale, is_using_pre_transform=False, pre_trans_R_a_h=None, pre_trans_t_a_h=None):
        '''
        '''
        # LM in face local frame
        #---------------------------#
        # Convert to numpy vector, shape: (3,1)
        # Applying the scale as well
        _np_point_3d_dict = dict()
        for _k in point_3d_dict:
            _np_point_3d_dict[_k] = np.array(point_3d_dict[_k]).reshape((3,1))
            _np_point_3d_dict[_k] *= pattern_scale # Multiply the scale
            # # Tilting or other correction
            # _R_correct = self.get_rotation_matrix_from_Euler(0.0, 0.0, 10.0, is_degree=True) # 15 deg up
            # _t_correct = np.array([[0.0, 0.0, 0.0]]).T
            # _np_point_3d_dict[_k] = self.transform_3D_point(_np_point_3d_dict[_k], _R_correct, _t_correct)
        # self.lib_print(_np_point_3d_dict)
        #---------------------------#

        # Pre-transfer
        #---------------------------#
        _np_point_3d_pretransfer_dict = copy.deepcopy(_np_point_3d_dict)
        if is_using_pre_transform:
            for _k in _np_point_3d_pretransfer_dict:
                _np_point_3d_pretransfer_dict[_k] = self.transform_3D_point(_np_point_3d_pretransfer_dict[_k], pre_trans_R_a_h, pre_trans_t_a_h)
        #---------------------------#


        # Logging
        #---------------------------#
        self.lib_print("-"*35)
        self.lib_print("3D points in local coordinate:")
        self.lib_print("pattern_scale = %f" % pattern_scale)
        for _k in _np_point_3d_dict:
            # self.lib_print("%s:\n%s" % (_k, str(_np_point_3d_dict[_k])))
            # self.lib_print("%s:\n%s" % (_k, str(_np_point_3d_pretransfer_dict[_k])))
            np.set_printoptions(suppress=True, precision=2)
            print("%s:%sp_3D=%s.T | p_3D_pretrans=%s.T" %
                (   _k,
                    " "*(12-len(_k)),
                    str(_np_point_3d_dict[_k].T),
                    str(_np_point_3d_pretransfer_dict[_k].T)
                )
            )
            np.set_printoptions(suppress=False, precision=8)
        self.lib_print("-"*35)
        #---------------------------#

        return (_np_point_3d_dict, _np_point_3d_pretransfer_dict)


    def lib_print(self, str=''):
        if self.verbose:
            print(str)

    # Solution
    #-----------------------------------------------------------#
    def solve_pnp(self, np_point_image_dict):
        '''
        '''
        res_norm_list = list()
        min_res_norm = None
        idx_best = None
        result_best = None

        for _idx in range(len(self.np_point_3d_pretransfer_dict_list)):
            _point_3d_dict = self.np_point_3d_pretransfer_dict_list[_idx]
            # # _result = (np_R_est, np_t_est, t3_est, roll_est, yaw_est, pitch_est, res_norm)
            # _result = self.solve_pnp_single_pattern(np_point_image_dict, _point_3d_dict)
            # _result = self.solve_pnp_seperate_single_pattern(np_point_image_dict, _point_3d_dict)
            # _result = self.solve_pnp_formulation_2_single_pattern(np_point_image_dict, _point_3d_dict) # f2
            # _result = self.solve_pnp_constrained_optimization_single_pattern(np_point_image_dict, _point_3d_dict)
            _result = self.solve_pnp_EKF_single_pattern(np_point_image_dict, _point_3d_dict)
            # _res_norm_n_est = _result[-1] * _result[2] # (res_norm * t3_est) Normalize the residual with distance estimation
            _res_norm = _result[-1]
            # Note: _res_norm is more stable than the _res_norm_n_est. When using _res_norm_n_est, the estimated depth will prone to smaller (since the _res_norm_n_est is smaller when estimated depth is smaller)
            self.lib_print("---\nPattern [%d]: _res_norm = %f\n---\n" % (_idx, _res_norm))
            res_norm_list.append(_res_norm)
            if (min_res_norm is None) or (_res_norm < min_res_norm):
                min_res_norm = _res_norm
                idx_best = _idx
                result_best = _result

        self.lib_print("-"*70)
        for _idx in range(len(res_norm_list)):
            self.lib_print("Pattern [%d]: _res_norm = %f" % (_idx, res_norm_list[_idx]))
        self.lib_print("-"*70)
        self.lib_print("--> Best fitted pattern is [%d] with res_norm_n_est = %f" % (idx_best, min_res_norm))
        self.lib_print("-"*70 + "\n")

        # Update the global id
        self.set_golden_pattern_id(idx_best)

        # Note: Euler angles are in degree
        return result_best
        # return [*result_best, idx_best, min_res_norm]

    def solve_pnp_single_pattern(self, np_point_image_dict, np_point_3d_pretransfer_dict):
        '''
        For each image frame,
        return (np_R_est, np_t_est, t3_est, roll_est, yaw_est, pitch_est )
        '''
        # Form the problem for solving
        B_all = self.get_B_all(np_point_image_dict, self.np_K_camera_est)
        self.lib_print("B_all = \n%s" % str(B_all))
        self.lib_print("B_all.shape = %s" % str(B_all.shape))

        self.lib_print("np_point_image_dict.keys() = %s" % str( np_point_image_dict.keys() ))
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
        # # Estimate the nose's height (scale)
        # _s3_est = 1.0
        #
        while k_it < num_it:
            k_it += 1
            self.lib_print("!!!!!!!!!!!!!!!!!!!!!!>>>>> k_it = %d" % k_it)

            # # Real update of phi_3_est
            # phi_3_est = copy.deepcopy(phi_3_est_new)

            # Generate Delta_i(k-1) and A(k-1)
            # Delta_all, A_all = self.get_Delta_A_all(self.np_point_3d_dict, np_point_image_dict, phi_3_est)
            Delta_all, A_all = self.get_Delta_A_all( np_point_3d_pretransfer_dict, np_point_image_dict, phi_3_est)
            #-------------------------#
            # self.lib_print("A_all = \n%s" % str(A_all))
            self.lib_print("A_all.shape = %s" % str(A_all.shape))
            rank_A_all = np.linalg.matrix_rank(A_all)
            self.lib_print("rank_A_all = %d" % rank_A_all)
            A_u, A_s, A_vh = np.linalg.svd(A_all)
            np.set_printoptions(suppress=True, precision=4)
            self.lib_print("A_s = %s" % str(A_s))
            # Note: the first row of the vh reveal the most significant element in phi_est (more precise in it estimation), and so on
            self.lib_print("A_vh = %s" % str(A_vh))
            np.set_printoptions(suppress=False, precision=8)
            #-------------------------#

            # Solve for phi
            #-------------------------#
            # # phi_est = np.linalg.inv(A_all.T @ A_all) @ A_all.T @ B_all
            #
            # # W_all_diag = np.ones((B_all.shape[0],))
            # #
            # _res_old_unit = self.unit_vec(res_old)
            # W_all_diag = 1.0/(0.0000001 +  np.squeeze(_res_old_unit)**4)
            # #
            # # W_all_diag = 0.5 * W_all_diag_old + 0.5 / (0.001 +  np.squeeze(_res_old_unit)**2)
            # #
            # # _res_old_abs = np.abs(res_old)
            # # _res_MAE = np.average( np.abs(res_old) )
            # # _res_dev = np.squeeze(np.abs(_res_old_abs - _res_MAE)) / _res_MAE
            # # self.lib_print("_res_dev = \n%s" % str(_res_dev))
            # # W_all_diag = 1.0/(0.001 +  _res_dev)
            # #
            # # W_all_diag = 100.0*np.exp(-1.0*_res_dev**2)
            # #
            # W_all_diag_old = copy.deepcopy(W_all_diag)
            # self.lib_print("W_all_diag = \n%s" % str(W_all_diag))
            # W_all = np.diag(W_all_diag)
            # # self.lib_print("W_all = \n%s" % str(W_all))
            # phi_est = np.linalg.inv(A_all.T @ W_all @ A_all) @ A_all.T @ W_all @ B_all
            #

            # Psudo inverse
            phi_est = np.linalg.pinv(A_all) @ B_all

            # # Seperated solve
            # #----------------------------------#
            # _A_x = A_all[0::2, [0,1,2,6]]
            # _A_y = A_all[1::2, [3,4,5,7]]
            # _B_x = B_all[0::2, :]
            # _B_y = B_all[1::2, :]
            # _phi_est_x = np.linalg.pinv(_A_x) @ _B_x
            # _phi_est_y = np.linalg.pinv(_A_y) @ _B_y
            # phi_est = np.vstack([_phi_est_x[0:3,:], _phi_est_y[0:3,:], _phi_est_x[3:4,:], _phi_est_y[3:4,:] ])
            # #----------------------------------#

            self.lib_print("phi_est = \n%s" % str(phi_est))
            # # residule
            # # _res = (A_all @ phi_est) - B_all
            # _res = B_all - (A_all @ phi_est)
            # res_old = copy.deepcopy(_res)
            # # self.lib_print("(A_all @ phi_est) = %s.T" % str((A_all @ phi_est).T))
            # self.lib_print("_res = %s.T" % str(_res.T))
            # # _res_delta = _res - np_quantization_error_world_space_vec
            # # self.lib_print("_res_delta = \n%s" % str(_res_delta))
            # res_norm = np.linalg.norm(_res)
            # self.lib_print("norm(_res) = %f" % res_norm)
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



            # '''
            # Fix the nose:
            # phi_1_x^2 + phi_1_y^2 + (s3*phi_1_z)^2 = phi_2_x^2 + phi_2_y^2 + (s3*phi_2_z)^2
            # =>
            # s3^2 = ( (phi_1_x^2 + phi_1_y^2) - (phi_2_x^2 + phi_2_y^2) ) / (phi_2_z^2 - phi_1_z^2)
            # '''
            # #-------------------------#
            # _s3_est_2 = (np.linalg.norm(phi_1_est[0:2,0]) - np.linalg.norm(phi_2_est[0:2,0])) / (phi_2_est[2,0] - phi_1_est[2,0])
            # print("_s3_est_2 = %f" % _s3_est_2)
            # #-------------------------#

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

            # Real residule (residule after reconstruction)
            #---------------------------------------------#
            phi_est_new = np.vstack([ np_R_est[0:1,:].T, np_R_est[1:2,:].T, np_t_est[0:2,:] ] ) / t3_est
            self.lib_print("phi_est_new = %s.T" % str(phi_est_new.T))
            _res = B_all - (A_all @ phi_est_new)
            res_old = copy.deepcopy(_res)
            # self.lib_print("(A_all @ phi_est) = %s.T" % str((A_all @ phi_est).T))
            self.lib_print("_res = %s.T" % str(_res.T))
            # _res_delta = _res - np_quantization_error_world_space_vec
            # self.lib_print("_res_delta = \n%s" % str(_res_delta))
            res_norm = np.linalg.norm(_res)
            self.lib_print("norm(_res) = %f" % res_norm)
            #---------------------------------------------#


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

    def solve_pnp_seperate_single_pattern(self, np_point_image_dict, np_point_3d_pretransfer_dict):
        '''
        For each image frame,
        return (np_R_est, np_t_est, t3_est, roll_est, yaw_est, pitch_est )
        '''
        # Form the problem for solving
        B_x, B_y = self.get_B_xy(np_point_image_dict, self.np_K_camera_est)
        self.lib_print("B_x = \n%s" % str(B_x))
        self.lib_print("B_y = \n%s" % str(B_y))
        self.lib_print("B_x.shape = %s" % str(B_x.shape))
        self.lib_print("B_y.shape = %s" % str(B_y.shape))
        #
        self.lib_print("np_point_image_dict.keys() = %s" % str( np_point_image_dict.keys() ))

        # Solve by iteration
        #--------------------------------------#
        # Initial guess, not neccessaryly unit vector!!
        # phi_3_est = np.array([-1.0, -1.0, -1.0]).reshape((3,1))
        phi_3_est = np.array([0.0, 0.0, 1.0]).reshape((3,1))
        # phi_3_est = np.array([0.0, 0.0, 0.0]).reshape((3,1))
        phi_3_est_new = copy.deepcopy(phi_3_est)
        step_alpha = 1.0 # 0.5
        num_it = 3
        #
        # update_phi_3_method = 1
        update_phi_3_method = 0
        # Iteration
        k_it = 0
        self.lib_print("---")
        #
        res_old_x = np.ones(B_x.shape)
        res_old_y = np.ones(B_y.shape)
        res_norm_x = 10*3
        res_norm_y = 10*3
        #
        # weight
        w_sqrt_x_vec = np.ones(B_x.shape)
        w_sqrt_y_vec = np.ones(B_x.shape)
        #
        while k_it < num_it:
            k_it += 1
            self.lib_print("!!!!!!!!!!!!!!!!!!!!!!>>>>> k_it = %d" % k_it)

            # # Real update of phi_3_est
            # phi_3_est = copy.deepcopy(phi_3_est_new)

            # Generate Delta_i(k-1) and A(k-1)
            # Delta_all, A_x, A_y = self.get_Delta_all_A_xy( self.np_point_3d_dict, np_point_image_dict, phi_3_est)
            Delta_all, A_x, A_y = self.get_Delta_all_A_xy( np_point_3d_pretransfer_dict, np_point_image_dict, phi_3_est)
            #-------------------------#
            # self.lib_print("A_all = \n%s" % str(A_all))
            self.lib_print("A_x.shape = %s" % str(A_x.shape))
            self.lib_print("A_y.shape = %s" % str(A_y.shape))
            rank_A_x = np.linalg.matrix_rank(A_x)
            rank_A_y = np.linalg.matrix_rank(A_y)
            self.lib_print("rank_A_x = %d" % rank_A_x)
            self.lib_print("rank_A_y = %d" % rank_A_y)
            A_x_u, A_x_s, A_x_vh = np.linalg.svd(A_x)
            A_y_u, A_y_s, A_y_vh = np.linalg.svd(A_y)
            np.set_printoptions(suppress=True, precision=4)
            self.lib_print("A_x_s = %s" % str(A_x_s))
            self.lib_print("A_x_vh = \n%s" % str(A_x_vh)) # Note: the first row of the vh reveal the most significant element in phi_est (more precise in it estimation), and so on
            self.lib_print("A_y_s = %s" % str(A_y_s))
            self.lib_print("A_y_vh = \n%s" % str(A_y_vh)) # Note: the first row of the vh reveal the most significant element in phi_est (more precise in it estimation), and so on
            np.set_printoptions(suppress=False, precision=8)
            #-------------------------#


            # Separately solve for phi
            #----------------------------------#
            res_grow_det_list = None
            # phi_x_est = self.solve_phi_half(A_x, B_x, name='x')
            # phi_y_est = self.solve_phi_half(A_y, B_y, name='y')
            phi_x_est, D_x, Bd_x, Delta_vec = self.solve_phi_half_numerator(A_x, B_x, name='x')
            phi_y_est, D_y, Bd_y, Delta_vec = self.solve_phi_half_numerator(A_y, B_y, name='y')
            # res_grow_det_list = (D_x, B_x, D_y, B_y, phi_3_est, -res_old_x*Delta_vec, -res_old_y*Delta_vec)

            # Dpinv_x = np.linalg.pinv(D_x)
            # self.lib_print("Dpinv_x = \n%s" % str(Dpinv_x))
            # DDpinv_x = D_x @ Dpinv_x
            # self.lib_print("DDpinv_x = \n%s" % str(DDpinv_x))
            # I_DDpinv_x = np.eye( Bd_x.shape[0] ) - DDpinv_x
            # self.lib_print("I_DDpinv_x = \n%s" % str(I_DDpinv_x))
            # rank_I_DDpinv_x = np.linalg.matrix_rank(I_DDpinv_x)
            # self.lib_print("rank_I_DDpinv_x = %d" % rank_I_DDpinv_x)
            # I_DDpinv_x_u, I_DDpinv_x_s, I_DDpinv_x_vh = np.linalg.svd(I_DDpinv_x)
            # # self.lib_print("I_DDpinv_x_u = \n%s" % str(I_DDpinv_x_u))
            # self.lib_print("I_DDpinv_x_s = \n%s" % str(I_DDpinv_x_s))
            # # self.lib_print("I_DDpinv_x_vh = \n%s" % str(I_DDpinv_x_vh))
            # I_DDpinv_x_s_is_dropped = I_DDpinv_x_s < 10**(-7)
            # I_DDpinv_x_s_truncate = copy.deepcopy(I_DDpinv_x_s)
            # I_DDpinv_x_s_truncate[I_DDpinv_x_s_is_dropped] = 0.0
            # I_DDpinv_x_truncate = I_DDpinv_x_u @ np.diag(I_DDpinv_x_s_truncate) @ I_DDpinv_x_vh
            # self.lib_print("I_DDpinv_x_truncate = \n%s" % str(I_DDpinv_x_truncate))
            #
            # # I_DDpinv_Bdiag_P_x = I_DDpinv_x @ np.diag(B_x[:,0]) @ D_x[:,0:3]
            # I_DDpinv_Bdiag_P_x = I_DDpinv_x_truncate @ np.diag(B_x[:,0]) @ D_x[:,0:3]
            # self.lib_print("I_DDpinv_Bdiag_P_x = \n%s" % str(I_DDpinv_Bdiag_P_x))
            # I_DDpinv_Bdiag_P_pinv_x = np.linalg.pinv(I_DDpinv_Bdiag_P_x)
            # self.lib_print("I_DDpinv_Bdiag_P_pinv_x = \n%s" % str(I_DDpinv_Bdiag_P_pinv_x))
            #
            # # e0_x = I_DDpinv_x @ B_x
            # e0_x = I_DDpinv_x_truncate @ B_x
            # self.lib_print("e0_x = \n%s" % str(e0_x))
            # phi_3_est_x = I_DDpinv_Bdiag_P_pinv_x @ (-e0_x)
            # self.lib_print("phi_3_est_x = \n%s" % str(phi_3_est_x))

            # piB_op_x = np.diag(B_x[:,0]) @ D_x[:,0:3]
            # piB_op_y = np.diag(B_y[:,0]) @ D_y[:,0:3]
            # self.lib_print("piB_op_x = \n%s" % str(piB_op_x))
            # self.lib_print("piB_op_y = \n%s" % str(piB_op_y))
            # rank_piB_op_x = np.linalg.matrix_rank(piB_op_x)
            # rank_piB_op_y = np.linalg.matrix_rank(piB_op_y)
            # self.lib_print("rank_piB_op_x = %d" % rank_piB_op_x)
            # self.lib_print("rank_piB_op_y = %d" % rank_piB_op_y)
            # # SVD
            # piB_op_x_u, piB_op_x_s, _piB_op_x_vh = np.linalg.svd(piB_op_x)
            # print("piB_op_x_s = %s" % str(piB_op_x_s))
            # #
            # piB_op_xy = np.vstack((piB_op_x, piB_op_y))
            # rank_piB_op_xy = np.linalg.matrix_rank(piB_op_xy)
            # self.lib_print("rank_piB_op_xy = %d" % rank_piB_op_xy)
            # B_xy = np.vstack( (B_x, B_y) )
            # #
            # piB_op_pinv_x = np.linalg.pinv(piB_op_x, rcond=10**(-7))
            # piB_op_pinv_y = np.linalg.pinv(piB_op_y, rcond=10**(-7))
            # piB_op_pinv_xy = np.linalg.pinv(piB_op_xy, rcond=10**(-7))
            # self.lib_print("piB_op_pinv_x = \n%s" % str(piB_op_pinv_x))
            # self.lib_print("piB_op_pinv_y = \n%s" % str(piB_op_pinv_y))
            # self.lib_print("piB_op_pinv_xy = \n%s" % str(piB_op_pinv_xy))
            # phi_3_est_x = piB_op_pinv_x @ (-B_x)
            # phi_3_est_y = piB_op_pinv_y @ (-B_y)
            # phi_3_est_xy = piB_op_pinv_xy @ (-B_xy)
            # # self.lib_print("B_x = \n%s" % str(B_x))
            # self.lib_print("phi_3_est_x = \n%s" % str(phi_3_est_x))
            # self.lib_print("phi_3_est_y = \n%s" % str(phi_3_est_y))
            # self.lib_print("phi_3_est_xy = \n%s" % str(phi_3_est_xy))


            # # Update square-rooted weight vectors
            # w_sqrt_x_vec = self.get_weight_from_residual(res_old_x, name="x")
            # w_sqrt_y_vec = self.get_weight_from_residual(res_old_y, name="y")
            # phi_x_est = self.solve_phi_half_weighted(A_x, B_x, w_sqrt_x_vec, name='x')
            # phi_y_est = self.solve_phi_half_weighted(A_y, B_y, w_sqrt_y_vec, name='y')

            phi_est = self.get_phi_est_from_halves(phi_x_est, phi_y_est)
            #----------------------------------#

            self.lib_print("phi_est = \n%s" % str(phi_est))
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


            #-------------------------#
            if update_phi_3_method == 1:
                # First update phi_3_est
                phi_3_est_new, norm_phi_3_est = self.update_phi_3_est_m1(phi_1_est, norm_phi_1_est, phi_2_est, norm_phi_2_est, phi_3_est, step_alpha)
                # Then, test (not necessary)
                #---------------------------------#
                if self.verbose:
                    np_R_est, np_t_est, t3_est = self.reconstruct_R_t_m1(phi_est, phi_3_est_new)
                else:
                    pass # Save the computing power
                #---------------------------------#
                # end Test
            else: # update_phi_3_method == 0
                # First reconstructing R
                np_R_est, np_t_est, t3_est = self.reconstruct_R_t_block_reconstruction(phi_est, phi_3_est, res_grow_det_list=res_grow_det_list)
                # Then, update phi_3_est
                phi_3_est_new, norm_phi_3_est = self.update_phi_3_est_m2(np_R_est, t3_est, phi_3_est, step_alpha)

            # Real residule (residule after reconstruction)
            # As well as checking which sign of alpha is correct
            #---------------------------------------------#
            _nz = np.array([[1.0, 1.0, -1.0, 1.0]]).T
            phi_est_new = np.vstack([ np_R_est[0:1,:].T, np_R_est[1:2,:].T, np_t_est[0:2,:] ] ) / t3_est
            phi_x_est, phi_y_est = self.get_phi_half_from_whole(phi_est_new)
            self.lib_print("phi_x_est = %s.T" % str(phi_x_est.T))
            self.lib_print("phi_y_est = %s.T" % str(phi_y_est.T))
            res_result_p = self.cal_res_all(A_x, B_x, phi_x_est, A_y, B_y, phi_y_est)
            res_result_n = self.cal_res_all(A_x, B_x, (phi_x_est*_nz), A_y, B_y, (phi_y_est*_nz))
            self.lib_print("res_norm_p = %f" % res_result_p[0])
            self.lib_print("res_norm_n = %f" % res_result_n[0])
            #
            res_norm, res_x, res_norm_x, res_y, res_norm_y = res_result_p
            # if res_result_n[0] <= res_result_p[0]:
            #     print(">>> The sign of alpha is wrong, fix it!!")
            #     res_norm, res_x, res_norm_x, res_y, res_norm_y = res_result_n
            #     # Fix all the related things...
            #     np_R_est[0:2,2] *= -1.0 # alpha
            #     np_R_est[2,0:2] *= -1.0 # beta.T
            #     phi_3_est_new[0:2,0] *= -1.0 # beta
            # else:
            #     print(">>> The sign of alpha is correct~")
            #     res_norm, res_x, res_norm_x, res_y, res_norm_y = res_result_p
            #
            self.lib_print("res_norm = %f" % res_norm)
            # Update
            res_old_x = copy.deepcopy(res_x)
            res_old_y = copy.deepcopy(res_y)
            #---------------------------------------------#

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
        else:
            pass # Note: if we are using the update_phi_3_method==0, we don't need to reconstruct the rotation matrix again
            # np_R_est, np_t_est, t3_est = self.reconstruct_R_t_block_reconstruction(phi_est, phi_3_est)
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
        #--------------------------------------------------------#

        # # Get the whole residual
        # #-----------------------------#
        # res_norm = np.sqrt(res_norm_x**2 + res_norm_y**2)
        # #-----------------------------#

        # Note: Euler angles are in degree
        return (np_R_est, np_t_est, t3_est, roll_est, yaw_est, pitch_est, res_norm)

    def solve_pnp_formulation_2_single_pattern(self, np_point_image_dict, np_point_3d_pretransfer_dict):
        '''
        For each image frame,
        return (np_R_est, np_t_est, t3_est, roll_est, yaw_est, pitch_est )
        '''
        # Form the problem for solving
        # Constant, only changed when golden pattern changed
        #--------------------------------#
        f2_P = self.f2_get_P(np_point_3d_pretransfer_dict)
        f2_D = self.f2_get_D_from_P(f2_P)
        f2_D_pinv = self.f2_get_D_pinv(f2_D)
        #--------------------------------#
        self.lib_print("f2_P = \n%s" % str(f2_P))
        self.lib_print("f2_D = \n%s" % str(f2_D))

        # Inspection
        #-------------------------#
        self.lib_print("f2_D.shape = %s" % str(f2_D.shape))
        rank_f2_D = np.linalg.matrix_rank(f2_D)
        self.lib_print("rank_f2_D = %d" % rank_f2_D)
        f2_D_u, f2_D_s, f2_D_vh = np.linalg.svd(f2_D)
        np.set_printoptions(suppress=True, precision=4)
        self.lib_print("f2_D_s = %s" % str(f2_D_s))
        self.lib_print("f2_D_vh = \n%s" % str(f2_D_vh)) # Note: the first row of the vh reveal the most significant element in phi_est (more precise in it estimation), and so on
        np.set_printoptions(suppress=False, precision=8)
        #-------------------------#

        self.lib_print("f2_D_pinv = \n%s" % str(f2_D_pinv))

        # Measure the duration of the process
        #-------------------------------------#
        _stamp_s = time.time()
        #-------------------------------------#

        # Change with sample, constant in iteration
        #--------------------------------#
        np_K_camera_inv = np.linalg.inv( self.np_K_camera_est )
        B_x, B_y = self.f2_get_B_xy(np_point_image_dict, np_K_camera_inv)
        #
        v_phi_o_x = self.f2_get_v_phi_o_half(f2_D_pinv, B_x)
        v_phi_o_y = self.f2_get_v_phi_o_half(f2_D_pinv, B_y)
        #
        M_x = self.f2_get_M_half(f2_D_pinv, B_x, f2_P)
        M_y = self.f2_get_M_half(f2_D_pinv, B_y, f2_P)
        #--------------------------------#

        #
        self.lib_print("B_x = \n%s" % str(B_x))
        self.lib_print("B_y = \n%s" % str(B_y))
        self.lib_print("B_x.shape = %s" % str(B_x.shape))
        self.lib_print("B_y.shape = %s" % str(B_y.shape))
        #
        self.lib_print("v_phi_o_x = \n%s" % str(v_phi_o_x))
        self.lib_print("v_phi_o_y = \n%s" % str(v_phi_o_y))
        self.lib_print("M_x = \n%s" % str(M_x))
        self.lib_print("M_y = \n%s" % str(M_y))


        #
        self.lib_print("np_point_image_dict.keys() = %s" % str( np_point_image_dict.keys() ))

        # Solve by iteration
        #--------------------------------------#
        # Initial guess, not neccessaryly unit vector!!
        # phi_3_est = np.array([-1.0, -1.0, -1.0]).reshape((3,1))
        phi_3_est = np.array([0.0, 0.0, 1.0]).reshape((3,1))
        # phi_3_est = np.array([0.0, 0.0, 0.0]).reshape((3,1))
        phi_3_est_new = copy.deepcopy(phi_3_est)
        step_alpha = 1.0 # 0.5
        num_it = 3
        #
        # update_phi_3_method = 1
        update_phi_3_method = 0
        # Iteration
        k_it = 0
        self.lib_print("---")
        #
        res_old_x = np.ones(B_x.shape)
        res_old_y = np.ones(B_y.shape)
        res_norm_x = 10*3
        res_norm_y = 10*3
        #
        # weight
        w_sqrt_x_vec = np.ones(B_x.shape)
        w_sqrt_y_vec = np.ones(B_x.shape)
        #
        while k_it < num_it:
            k_it += 1
            self.lib_print("!!!!!!!!!!!!!!!!!!!!!!>>>>> k_it = %d" % k_it)

            # # Real update of phi_3_est
            # phi_3_est = copy.deepcopy(phi_3_est_new)

            # Separately solve for phi
            #----------------------------------#
            res_grow_det_list = None
            phi_x_est = self.f2_solve_phi_half_from_phi_3(v_phi_o_x, M_x, phi_3_est, name='x')
            phi_y_est = self.f2_solve_phi_half_from_phi_3(v_phi_o_y, M_y, phi_3_est, name='y')
            # res_grow_det_list = (D_x, B_x, D_y, B_y, phi_3_est, -res_old_x*Delta_vec, -res_old_y*Delta_vec)
            #
            # Note: phi_est = [phi_1_est; phi_2_est; delta_est]
            phi_est = self.get_phi_est_from_halves(phi_x_est, phi_y_est)
            self.lib_print("phi_est = \n%s" % str(phi_est))
            #----------------------------------#


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
            self.lib_print("phi_3_est = \n%s" % str(phi_3_est))
            #-------------------------#


            # # Experiment
            # #----------------------------------#
            # np.set_printoptions(suppress=True, precision=4)
            # n_point = f2_D.shape[0]
            # _Q = np.eye(n_point) - (f2_D @ f2_D_pinv)
            # self.lib_print("_Q = %s" % _Q)
            # DBx = np.diag(B_x[:,0])
            # DBy = np.diag(B_y[:,0])
            #
            # DBxP = DBx @ f2_P
            # DByP = DBy @ f2_P
            #
            # self.lib_print("DBxP = \n%s" % DBxP)
            # PTDBxQ = DBxP.T @ _Q
            # self.lib_print("PTDBxQ = \n%s" % PTDBxQ)
            # np.set_printoptions(suppress=False, precision=8)
            #
            # PTDBxQBx = PTDBxQ @ B_x
            # PTDBxQDBxP = PTDBxQ @ DBxP
            # self.lib_print("PTDBxQBx = \n%s" % PTDBxQBx)
            # self.lib_print("PTDBxQDBxP = \n%s" % PTDBxQDBxP)
            #
            # # THe derivative
            # d_phi_3_x = DBxP.T @ _Q @ ( B_x + DBxP @ phi_3_est)
            # d_phi_3_y = DByP.T @ _Q @ ( B_y + DByP @ phi_3_est)
            # d_phi_3_total = d_phi_3_x + d_phi_3_y
            # self.lib_print("d_phi_3_x = \n%s" % d_phi_3_x)
            # self.lib_print("d_phi_3_y = \n%s" % d_phi_3_y)
            # self.lib_print("d_phi_3_total = \n%s" % d_phi_3_total)
            #
            # phi_3_est_GD = phi_3_est - 10.0**5 * d_phi_3_total;
            # self.lib_print("phi_3_est_GD = \n%s" % phi_3_est_GD)
            # #----------------------------------#


            #-------------------------#
            if update_phi_3_method == 1:
                # First update phi_3_est
                phi_3_est_new, norm_phi_3_est = self.update_phi_3_est_m1(phi_1_est, norm_phi_1_est, phi_2_est, norm_phi_2_est, phi_3_est, step_alpha)
                # Then, test (not necessary)
                #---------------------------------#
                if self.verbose:
                    np_R_est, np_t_est, t3_est = self.reconstruct_R_t_m1(phi_est, phi_3_est_new)
                else:
                    pass # Save the computing power
                #---------------------------------#
                # end Test
            else: # update_phi_3_method == 0
                # First reconstructing R
                np_R_est, np_t_est, t3_est = self.reconstruct_R_t_block_reconstruction(phi_est, phi_3_est, res_grow_det_list=res_grow_det_list)
                # Then, update phi_3_est
                phi_3_est_new, norm_phi_3_est = self.update_phi_3_est_m2(np_R_est, t3_est, phi_3_est, step_alpha)

            # Real residule (residule after reconstruction)
            # As well as checking which sign of alpha is correct
            #---------------------------------------------#
            _nz = np.array([[1.0, 1.0, -1.0, 1.0]]).T
            phi_est_new = np.vstack([ np_R_est[0:1,:].T, np_R_est[1:2,:].T, np_t_est[0:2,:] ] ) / t3_est
            phi_x_est, phi_y_est = self.get_phi_half_from_whole(phi_est_new)
            self.lib_print("phi_x_est = %s.T" % str(phi_x_est.T))
            self.lib_print("phi_y_est = %s.T" % str(phi_y_est.T))
            #
            # is_f1_res = True
            is_f1_res = False
            res_result_p = self.f2_cal_res_all(f2_D, f2_P, B_x, B_y, phi_x_est, phi_y_est, phi_3_est, is_f1_res=is_f1_res)
            res_result_n = self.f2_cal_res_all(f2_D, f2_P, B_x, B_y, (phi_x_est*_nz), (phi_y_est*_nz), phi_3_est, is_f1_res=is_f1_res)
            self.lib_print("res_norm_p = %f" % res_result_p[0])
            self.lib_print("res_norm_n = %f" % res_result_n[0])
            #
            res_norm, res_x, res_norm_x, res_y, res_norm_y = res_result_p
            # if res_result_n[0] <= res_result_p[0]:
            #     print(">>> The sign of alpha is wrong, fix it!!")
            #     res_norm, res_x, res_norm_x, res_y, res_norm_y = res_result_n
            #     # Fix all the related things...
            #     np_R_est[0:2,2] *= -1.0 # alpha
            #     np_R_est[2,0:2] *= -1.0 # beta.T
            #     phi_3_est_new[0:2,0] *= -1.0 # beta
            # else:
            #     print(">>> The sign of alpha is correct~")
            #     res_norm, res_x, res_norm_x, res_y, res_norm_y = res_result_p
            #
            self.lib_print("res_norm = %f" % res_norm)
            # Update
            res_old_x = copy.deepcopy(res_x)
            res_old_y = copy.deepcopy(res_y)
            #---------------------------------------------#

            # Real update of phi_3_est
            phi_3_est = copy.deepcopy(phi_3_est_new)
            # phi_3_est = copy.deepcopy(phi_3_est_GD)
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
        else:
            pass # Note: if we are using the update_phi_3_method==0, we don't need to reconstruct the rotation matrix again
            # np_R_est, np_t_est, t3_est = self.reconstruct_R_t_block_reconstruction(phi_est, phi_3_est)
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
        #--------------------------------------------------------#

        # # Get the whole residual
        # #-----------------------------#
        # res_norm = np.sqrt(res_norm_x**2 + res_norm_y**2)
        # #-----------------------------#


        # Measure the duration of the process
        #-------------------------------------#
        _duration = time.time() - _stamp_s
        self.lib_print("\n*** _duration = %f ms ***\n" % (_duration*1000.0))
        #-------------------------------------#

        # Note: Euler angles are in degree
        return (np_R_est, np_t_est, t3_est, roll_est, yaw_est, pitch_est, res_norm)

    def solve_pnp_constrained_optimization_single_pattern(self, np_point_image_dict, np_point_3d_pretransfer_dict):
        '''
        For each image frame,
        return (np_R_est, np_t_est, t3_est, roll_est, yaw_est, pitch_est )
        '''
        # Form the problem for solving
        # Constant, only changed when golden pattern changed
        #--------------------------------#
        co_P = self.f2_get_P(np_point_3d_pretransfer_dict)
        #--------------------------------#
        self.lib_print("co_P = \n%s" % str(co_P))
        #
        self.lib_print("np_point_image_dict.keys() = %s" % str( np_point_image_dict.keys() ))



        # Measure the duration of the process
        #-------------------------------------#
        _stamp_s = time.time()
        #-------------------------------------#


        # Change with sample, constant in iteration
        #--------------------------------#
        np_K_camera_inv = np.linalg.inv( self.np_K_camera_est )
        B_x, B_y = self.f2_get_B_xy(np_point_image_dict, np_K_camera_inv)
        co_matrices = self.co_prepare_matrix_components(B_x, B_y, co_P)
        co_W, co_Ux, co_Uy, co_VxpVy, co_PTone, co_PTBx, co_PTBy, co_PTBxBxByBy, co_BxTone, co_ByTone = co_matrices
        #--------------------------------#

        # B
        self.lib_print("B_x = \n%s" % str(B_x))
        self.lib_print("B_y = \n%s" % str(B_y))
        self.lib_print("B_x.shape = %s" % str(B_x.shape))
        self.lib_print("B_y.shape = %s" % str(B_y.shape))
        # 3x3 matrices
        self.lib_print("co_W = \n%s" % str(co_W))
        self.lib_print("co_Ux = \n%s" % str(co_Ux))
        self.lib_print("co_Uy = \n%s" % str(co_Uy))
        self.lib_print("co_VxpVy = \n%s" % str(co_VxpVy))
        # 3x1 vectors
        self.lib_print("co_PTone = \n%s" % str(co_PTone))
        self.lib_print("co_PTBx = \n%s" % str(co_PTBx))
        self.lib_print("co_PTBy = \n%s" % str(co_PTBy))
        self.lib_print("co_PTBxBxByBy = \n%s" % str(co_PTBxBxByBy))
        # 1x1 scalars
        self.lib_print("co_BxTone = %f" % co_BxTone)
        self.lib_print("co_ByTone = %f" % co_ByTone)

        # Prepare big matrices
        #--------------------------------#
        co_A_list = list()
        co_bT_list = list()
        co_c_list = list()
        #
        zeros_3x3 = np.zeros((3,3))
        eye_3x3 = np.eye(3)
        zeros_3x1 = np.zeros((3,1))
        n_point = co_P.shape[0]

        # f1
        # --A
        _co_Ai = np.zeros((11,11))
        _co_Ai[0:3,:] = np.hstack([co_Ux, co_Uy, (-co_VxpVy), co_PTBx, co_PTBy])
        _co_Ai[6:9,:] = np.hstack([co_W, zeros_3x3, (-co_Ux), co_PTone, zeros_3x1])
        # --b
        _co_bi = np.zeros((1,11))
        _co_bi[:,0:3] = (-co_PTBxBxByBy.T)
        _co_bi[:,6:9] = (-co_PTBx.T)
        # --c
        _co_ci = 0.0
        #
        co_A_list.append(_co_Ai)
        co_bT_list.append(_co_bi)
        co_c_list.append(_co_ci)


        # f2
        # --A
        _co_Ai = np.zeros((11,11))
        _co_Ai[3:6,:] = np.hstack([co_Ux, co_Uy, (-co_VxpVy), co_PTBx, co_PTBy])
        _co_Ai[6:9,:] = np.hstack([zeros_3x3, co_W, (-co_Uy), zeros_3x1, co_PTone])
        # --b
        _co_bi = np.zeros((1,11))
        _co_bi[:,3:6] = (-co_PTBxBxByBy.T)
        _co_bi[:,6:9] = (-co_PTBy.T)
        # --c
        _co_ci = 0.0
        #
        co_A_list.append(_co_Ai)
        co_bT_list.append(_co_bi)
        co_c_list.append(_co_ci)


        # f3
        # --A
        _co_Ai = np.zeros((11,11))
        _co_Ai[0:3,:] = np.hstack([zeros_3x3, (-co_W), co_Uy, zeros_3x1, (-co_PTone)])
        _co_Ai[3:6,:] = np.hstack([co_W, zeros_3x3, (-co_Ux), co_PTone, zeros_3x1])
        # --b
        _co_bi = np.zeros((1,11))
        _co_bi[:,0:3] = co_PTBy.T
        _co_bi[:,3:6] = (-co_PTBx.T)
        # --c
        _co_ci = 0.0
        #
        co_A_list.append(_co_Ai)
        co_bT_list.append(_co_bi)
        co_c_list.append(_co_ci)

        # f4
        # --A
        _co_Ai = np.zeros((11,11))
        _co_Ai[0:3,:] = np.hstack([co_W, zeros_3x3, (-co_Ux), co_PTone, zeros_3x1])
        _co_Ai[3:6,:] = np.hstack([zeros_3x3, co_W, (-co_Uy), zeros_3x1, co_PTone])
        _co_Ai[6:9,:] = np.hstack([(-co_Ux), (-co_Uy), co_VxpVy, (-co_PTBx), (-co_PTBy)])
        # --b
        _co_bi = np.zeros((1,11))
        _co_bi[:,0:3] = (-co_PTBx.T)
        _co_bi[:,3:6] = (-co_PTBy.T)
        _co_bi[:,6:9] = co_PTBxBxByBy.T
        # --c
        _co_ci = 0.0
        #
        co_A_list.append(_co_Ai)
        co_bT_list.append(_co_bi)
        co_c_list.append(_co_ci)


        # f5
        # --A
        _co_Ai = np.zeros((11,11))
        # --b
        _co_bi = np.zeros((1,11))
        _co_bi[:,0:3] = co_PTone.T
        _co_bi[:,6:9] = (-co_PTBx.T)
        _co_bi[0,9] = n_point
        # --c
        _co_ci = (-co_BxTone)
        #
        co_A_list.append(_co_Ai)
        co_bT_list.append(_co_bi)
        co_c_list.append(_co_ci)


        # f6
        # --A
        _co_Ai = np.zeros((11,11))
        # --b
        _co_bi = np.zeros((1,11))
        _co_bi[:,3:6] = co_PTone.T
        _co_bi[:,6:9] = (-co_PTBy.T)
        _co_bi[0,10] = n_point
        # --c
        _co_ci = (-co_ByTone)
        #
        co_A_list.append(_co_Ai)
        co_bT_list.append(_co_bi)
        co_c_list.append(_co_ci)


        # f7
        # --A
        _co_Ai = np.zeros((11,11))
        _co_Ai[0:3,6:9] = eye_3x3
        # --b
        _co_bi = np.zeros((1,11))
        # --c
        _co_ci = 0.0
        #
        co_A_list.append(_co_Ai)
        co_bT_list.append(_co_bi)
        co_c_list.append(_co_ci)

        # f8
        # --A
        _co_Ai = np.zeros((11,11))
        _co_Ai[3:6,6:9] = eye_3x3
        # --b
        _co_bi = np.zeros((1,11))
        # --c
        _co_ci = 0.0
        #
        co_A_list.append(_co_Ai)
        co_bT_list.append(_co_bi)
        co_c_list.append(_co_ci)


        # f9
        # --A
        _co_Ai = np.zeros((11,11))
        _co_Ai[0:3,3:6] = eye_3x3
        # --b
        _co_bi = np.zeros((1,11))
        # --c
        _co_ci = 0.0
        #
        co_A_list.append(_co_Ai)
        co_bT_list.append(_co_bi)
        co_c_list.append(_co_ci)


        # f10
        # --A
        _co_Ai = np.zeros((11,11))
        _co_Ai[0:3,0:3] = eye_3x3
        _co_Ai[6:9,6:9] = (-eye_3x3)
        # --b
        _co_bi = np.zeros((1,11))
        # --c
        _co_ci = 0.0
        #
        co_A_list.append(_co_Ai)
        co_bT_list.append(_co_bi)
        co_c_list.append(_co_ci)


        # f11
        # --A
        _co_Ai = np.zeros((11,11))
        _co_Ai[3:6,3:6] = eye_3x3
        _co_Ai[6:9,6:9] = (-eye_3x3)
        # --b
        _co_bi = np.zeros((1,11))
        # --c
        _co_ci = 0.0
        #
        co_A_list.append(_co_Ai)
        co_bT_list.append(_co_bi)
        co_c_list.append(_co_ci)
        #--------------------------------#

        #
        self.lib_print("-"*70)
        for _idx in range(len(co_A_list)):
            self.lib_print("A[%d] = \n%s" % (_idx, co_A_list[_idx]))
            self.lib_print("bT[%d] = %s" % (_idx, co_bT_list[_idx]))
            self.lib_print("c[%d] = %f" % (_idx, co_c_list[_idx]))
        self.lib_print("-"*70)
        #


        # Solve by iteration (Newton-Raphson method)
        #--------------------------------------#
        # Initial guess
        co_x = np.zeros((11,1)) # [phi_1; phi_2; phi_3; delta_1; delta_2]
        # Set the initial scaled rotation matrix (Gamma) to be an identidy matrix
        co_x[(3*0)+0, 0] = 1.0
        co_x[(3*1)+1, 0] = 1.0
        co_x[(3*2)+2, 0] = 1.0
        #

        step_alpha = 1.0 # 0.5
        num_it = 14 # 3
        #
        # Iteration
        k_it = 0
        self.lib_print("---")
        while k_it < num_it:
            k_it += 1
            self.lib_print("!!!!!!!!!!!!!!!!!!!!!!>>>>> k_it = %d" % k_it)

            # Calculate delta_x
            #-----------------------------#
            # co_fx = None # 11 x 1
            # co_Jf  = None # 11 x 11
            co_fx, co_Jf = self.co_get_function_value_and_Jacobian(co_x, co_A_list, co_bT_list, co_c_list)
            co_Jf_u, co_Jf_s, co_Jf_vh = np.linalg.svd(co_Jf)
            #
            co_Jf_pinv = np.linalg.pinv(co_Jf) # 11 x 11

            co_delta_x = -1.0 * (co_Jf_pinv @ co_fx) # 11 x 1
            #
            self.lib_print("co_fx = \n%s" % str(co_fx))
            self.lib_print("co_Jf = \n%s" % str(co_Jf))
            self.lib_print("co_Jf_s = \n%s" % str(co_Jf_s))
            self.lib_print("co_Jf_pinv = \n%s" % str(co_Jf_pinv))
            self.lib_print("co_delta_x = \n%s" % str(co_delta_x))
            #-----------------------------#

            # Update x
            #-----------------------------#
            self.lib_print("(old) co_x = \n%s" % str(co_x))
            co_x += co_delta_x
            self.lib_print("(new) co_x = \n%s" % str(co_x))
            #-----------------------------#

            # # Iterative method
            # #-----------------------------#
            # co_sysA_list = list()
            # co_sysB_list = list()
            # for _idx in range(len(co_A_list)):
            #     co_sysA_list.append( ( co_x.T @ co_A_list[_idx] + co_bT_list[_idx] ) )
            #     co_sysB_list.append( -co_c_list[_idx] )
            #     # co_sysA_list.append( ( co_x.T @ co_A_list[_idx] ) )
            #     # co_sysB_list.append( (-co_c_list[_idx]) - (co_bT_list[_idx] @ co_x) )
            # co_sysA = np.vstack(co_sysA_list)
            # co_sysB = np.vstack(co_sysB_list)
            # co_sysA_u, co_sysA_s, co_sysA_vh = np.linalg.svd(co_sysA)
            # co_sysA_pinv = np.linalg.pinv(co_sysA)
            # #
            # self.lib_print("co_sysA = \n%s" % str(co_sysA))
            # self.lib_print("co_sysA_s = \n%s" % str(co_sysA_s))
            # self.lib_print("co_sysA_pinv = \n%s" % str(co_sysA_pinv))
            # self.lib_print("co_sysB = \n%s" % str(co_sysB))
            # # Update
            # self.lib_print("(old) co_x = \n%s" % str(co_x))
            # co_x = co_sysA_pinv @ co_sysB
            # self.lib_print("(new) co_x = \n%s" % str(co_x))
            # #-----------------------------#


            self.lib_print("---")

        #--------------------------------------#

        self.lib_print()
        # Reconstruct (R, t)
        #--------------------------------------------------------#
        np_R_est, np_t_est, t3_est = self.co_reconstruct_R_t_m1(co_x)

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
        #--------------------------------------------------------#

        # # Get the whole residual
        # #-----------------------------#
        # res_norm = np.sqrt(res_norm_x**2 + res_norm_y**2)
        res_norm = 0.0
        # #-----------------------------#


        # Measure the duration of the process
        #-------------------------------------#
        _duration = time.time() - _stamp_s
        self.lib_print("\n*** _duration = %f ms ***\n" % (_duration*1000.0))
        #-------------------------------------#

        # Note: Euler angles are in degree
        return (np_R_est, np_t_est, t3_est, roll_est, yaw_est, pitch_est, res_norm)

    def solve_pnp_EKF_single_pattern(self, np_point_image_dict, np_point_3d_pretransfer_dict):
        '''
        For each image frame,
        return (np_R_est, np_t_est, t3_est, roll_est, yaw_est, pitch_est )
        '''
        # Form the problem for solving
        # Constant, only changed when golden pattern changed
        #--------------------------------#
        co_P = self.f2_get_P(np_point_3d_pretransfer_dict)
        n_point = co_P.shape[0]
        ekf_G = np.eye(11)
        ekf_R = np.eye(11) * (10**-3)
        #--------------------------------#
        self.lib_print("co_P = \n%s" % str(co_P))
        #
        self.lib_print("np_point_image_dict.keys() = %s" % str( np_point_image_dict.keys() ))



        # Measure the duration of the process
        #-------------------------------------#
        _stamp_s = time.time()
        #-------------------------------------#


        # Change with sample, constant in iteration
        #--------------------------------#
        np_K_camera_inv = np.linalg.inv( self.np_K_camera_est )
        B_x, B_y = self.f2_get_B_xy(np_point_image_dict, np_K_camera_inv)
        #
        ekf_z = np.zeros( ((2*n_point+5), 1) )
        ekf_z[:n_point,:] = B_x
        ekf_z[n_point:(2*n_point),:] = B_y
        #--------------------------------#

        # B
        self.lib_print("B_x = \n%s" % str(B_x))
        self.lib_print("B_y = \n%s" % str(B_y))
        self.lib_print("B_x.shape = %s" % str(B_x.shape))
        self.lib_print("B_y.shape = %s" % str(B_y.shape))
        #
        self.lib_print("ekf_z = \n%s" % str(ekf_z))


        # Solve by iteration (EKF)
        #--------------------------------------#
        # Initial guess
        ekf_x = np.zeros((11,1)) # [phi_1; phi_2; phi_3; delta_1; delta_2]
        # Set the initial scaled rotation matrix (Gamma) to be an identidy matrix
        ekf_x[(3*0)+0, 0] = 1.0
        ekf_x[(3*1)+1, 0] = 1.0
        ekf_x[(3*2)+2, 0] = 1.0
        #
        ekf_Sigma = np.eye(11) * 10**5


        num_it = 14 # 3
        #
        # Iteration
        k_it = 0
        self.lib_print("---")
        while k_it < num_it:
            k_it += 1
            self.lib_print("!!!!!!!!!!!!!!!!!!!!!!>>>>> k_it = %d" % k_it)

            # Calculate K
            #-----------------------------#
            ekf_Q = np.eye((2*n_point+5))
            ekf_Q[-5:, -5:] *= 225.68 # f_camera # 2.3*10**2
            ekf_Q /= 225.68
            #
            ekf_x_bar = ekf_x
            ekf_Sigma_bar = ekf_G @ ekf_Sigma @ ekf_G.T + ekf_R
            #
            ekf_hx, ekf_Hx = self.EKF_get_hx_H(ekf_x, B_x, B_y, co_P)
            #
            ekf_S = ( ekf_Hx @ ekf_Sigma_bar @ (ekf_Hx.T) + ekf_Q )
            ekf_S_u, ekf_S_s, ekf_S_vh = np.linalg.svd(ekf_S)
            ekf_S_pinv = np.linalg.pinv(ekf_S)
            ekf_K = ekf_Sigma_bar @ (ekf_Hx.T) @ ekf_S_pinv
            #
            self.lib_print("ekf_Q = \n%s" % str(ekf_Q))
            # self.lib_print("ekf_Hx = \n%s" % str(ekf_Hx))
            # self.lib_print("ekf_S = \n%s" % str(ekf_S))
            self.lib_print("ekf_S_s = \n%s" % str(ekf_S_s))
            # self.lib_print("ekf_S_pinv = \n%s" % str(ekf_S_pinv))
            self.lib_print("ekf_K = \n%s" % str(ekf_K))
            self.lib_print("ekf_z = \n%s" % str(ekf_z))
            self.lib_print("ekf_hx = \n%s" % str(ekf_hx))
            self.lib_print("(ekf_z-ekf_hx) = \n%s" % str(ekf_z-ekf_hx))
            #-----------------------------#

            # Update x
            #-----------------------------#
            self.lib_print("(old) ekf_x = \n%s" % str(ekf_x))
            ekf_x += ekf_K @ (ekf_z - ekf_hx)
            ekf_Sigma -= (ekf_K @ ekf_Hx) @ ekf_Sigma_bar
            self.lib_print("(new) ekf_x = \n%s" % str(ekf_x))
            #-----------------------------#


            self.lib_print("---")

        #--------------------------------------#

        self.lib_print()
        # Reconstruct (R, t)
        #--------------------------------------------------------#
        np_R_est, np_t_est, t3_est = self.co_reconstruct_R_t_m1(ekf_x)

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
        #--------------------------------------------------------#

        # # Get the whole residual
        # #-----------------------------#
        # res_norm = np.sqrt(res_norm_x**2 + res_norm_y**2)
        res_norm = 0.0
        # #-----------------------------#


        # Measure the duration of the process
        #-------------------------------------#
        _duration = time.time() - _stamp_s
        self.lib_print("\n*** _duration = %f ms ***\n" % (_duration*1000.0))
        #-------------------------------------#

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
        # # _phi_3_norm_2 = np.linalg.norm(phi_3_est)**2
        # # phi_3_est_1 = phi_3_est / _phi_3_norm_2 * 1000.0
        # # A_i_list.append( np.hstack([phi_3_est_1.T, np.zeros((1,3)), np.zeros((1,2))]))
        # # A_i_list.append( np.hstack([np.zeros((1,3)), phi_3_est_1.T, np.zeros((1,2))]))
        # # _zs_T = np.array([[0.0, 0.0, 1.0]]) * 10**-3
        # # A_i_list.append( np.hstack([ _zs_T, np.zeros((1,3)), np.zeros((1,2))]) )
        # # A_i_list.append( np.hstack([ np.zeros((1,3)), _zs_T, np.zeros((1,2))]) )
        # _scale = 5.0 * 10**-2
        # _dist = 1.0 / np.linalg.norm(phi_3_est)
        # _xs_T = np.array([[1.0, 0.0, 0.0]]) * _scale
        # _ys_T = np.array([[0.0, 1.0, 0.0]]) * _scale
        # A_i_list.append( np.hstack([ _xs_T * _dist, np.zeros((1,3)), np.zeros((1,2))]) )
        # A_i_list.append( np.hstack([ np.zeros((1,3)), _ys_T * _dist, np.zeros((1,2))]) )
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
        # # # test, add the perpendicular condition
        # # #---------------------------------#
        # # B_i_list.append( np.zeros((2,1)))
        # # B_i_list.append( np.zeros((2,1)))
        # _scale = 5.0 * 10**-2
        # B_i_list.append( np.ones((2,1)) * _scale )
        # # #---------------------------------#
        B_all = np.vstack(B_i_list)
        return B_all

    # Separate x and y
    #-------------------------------------------#
    def get_A_i_half(self, theta_i, Delta_i):
        '''
        theta_i=[[x,y,z]].T, shape=(3,1)
        Delta_i=(theta_i.T @ phi_3 + 1.0), scalar
        '''
        _theta_i_T = theta_i.reshape((1,3)) # Incase it's 1-D array, use reshape instead of theta_i.T
        #
        _theta_i_T_div_Delta_i = _theta_i_T / Delta_i
        A_i_half = np.hstack( [_theta_i_T_div_Delta_i, 1.0/Delta_i ] )
        return A_i_half

    def get_Delta_all_A_xy(self, np_point_3d_dict_in, np_point_image_dict_in, phi_3_est):
        '''
        '''
        # To be more realistic, use the image point to search
        A_i_half_list = list()
        Delta_i_list = list()
        for _id, _k in enumerate(np_point_image_dict_in):
            Delta_i = self.get_Delta_i(np_point_3d_dict_in[_k], phi_3_est, id=_id)
            Delta_i_list.append( Delta_i )
            A_i_half_list.append( self.get_A_i_half(np_point_3d_dict_in[_k], Delta_i) )
        Delta_all = np.vstack(Delta_i_list)
        A_xy = np.vstack(A_i_half_list)
        # Assign the A_x and A_y
        A_x = A_xy
        A_y = A_xy
        return (Delta_all, A_x, A_y)

    def get_B_xy(self, np_point_image_dict_in, K_in):
        '''
        '''
        _K_inv = np.linalg.inv(K_in)
        # To be more realistic, use the image point to search
        B_i_list = list()
        for _k in np_point_image_dict_in:
            nu_i = _K_inv @ np_point_image_dict_in[_k] # shape = (3,1)
            B_i_list.append( nu_i[0:2,:] )
        B_all = np.vstack(B_i_list)
        # Separate the B
        B_x = B_all[0::2]
        B_y = B_all[1::2]
        return (B_x, B_y)

    def get_phi_est_from_halves(self, phi_x, phi_y):
        '''
        '''
        phi_est = phi_est = np.vstack([phi_x[0:3,:], phi_y[0:3,:], phi_x[3:4,:], phi_y[3:4,:] ])
        return phi_est

    def get_phi_half_from_whole(self, phi_est):
        '''
        '''
        phi_x = np.vstack( [phi_est[0:3,:], phi_est[6:7,:]])
        phi_y = np.vstack( [phi_est[3:6,:], phi_est[7:8,:]])
        return (phi_x, phi_y)

    def solve_phi_half(self, A_half, B_half, name='half'):
        '''
        '''
        phi_half = np.linalg.pinv(A_half) @ B_half
        self.lib_print("phi_est [%s] = %s.T" % (name, str(phi_half.T)))
        return phi_half

    def solve_phi_half_numerator(self, A_half, B_half, name='half'):
        '''
        '''
        Delta_vec = 1.0 / A_half[:,3:4]
        D_half = A_half * Delta_vec
        Bd_half = Delta_vec * B_half
        # phi_half = np.linalg.pinv(A_half) @ B_half
        phi_half = np.linalg.pinv(D_half) @ Bd_half
        self.lib_print("phi_est [%s] = %s.T" % (name, str(phi_half.T)))
        return (phi_half, D_half, Bd_half, Delta_vec)


    def get_weight_from_residual(self, res, name="half"):
        '''
        '''
        res_unit = self.unit_vec(res)
        w_sqrt_half_vec = 1.0 / (0.001 + res_unit**2)
        np.set_printoptions(suppress=True, precision=4)
        self.lib_print("w_sqrt_vec [%s] = %s.T" % (name, str(w_sqrt_half_vec.T)) )
        np.set_printoptions(suppress=False, precision=8)
        return w_sqrt_half_vec

    # def get_weight_from_residual(self, res, name="half"):
    #     '''
    #     '''
    #     res_unit = self.unit_vec(res)
    #     res_unit_abs = np.abs(res_unit)
    #     res_unit_abs_mean = np.average(res_unit_abs)
    #     res_unit_dev = np.abs(res_unit_abs - res_unit_abs_mean)
    #     w_sqrt_half_vec = 1.0 / (0.001 + res_unit_dev**2)
    #     np.set_printoptions(suppress=True, precision=4)
    #     self.lib_print("w_sqrt_vec [%s] = %s.T" % (name, str(w_sqrt_half_vec.T)) )
    #     np.set_printoptions(suppress=False, precision=8)
    #     return w_sqrt_half_vec

    # def get_weight_from_residual(self, res, name="half"):
    #     '''
    #     '''
    #     res_unit = self.unit_vec(res)
    #     w_sqrt_half_vec = res_unit**2
    #     np.set_printoptions(suppress=True, precision=4)
    #     self.lib_print("w_sqrt_vec [%s] = %s.T" % (name, str(w_sqrt_half_vec.T)) )
    #     np.set_printoptions(suppress=False, precision=8)
    #     return w_sqrt_half_vec


    def solve_phi_half_weighted(self, A_half, B_half, w_sqrt_half_vec=None, name='half'):
        '''
        w_sqrt_half_vec: the vector of square roots of the weights
        '''
        # phi_half = np.linalg.pinv(A_half) @ B_half
        # phi_half = np.linalg.pinv(A_half.T @ A_half) @ A_half.T @ B_half
        if w_sqrt_half_vec is None:
            w_sqrt_half_vec = np.ones(B_half.shape)
        w_sqrt_half_vec = w_sqrt_half_vec.reshape(B_half.shape) # Note: should be in the same shape as B_half
        Aw = w_sqrt_half_vec * A_half # Note: row-wise, broadcast through column
        Bw = w_sqrt_half_vec * B_half # Note: per-element multiplication
        phi_half = np.linalg.pinv(Aw.T @ Aw) @ Aw.T @ Bw
        self.lib_print("phi_est [%s] = %s.T" % (name, str(phi_half.T)))
        return phi_half

    def cal_res_half(self, A_half, B_half, phi_half, name='half'):
        '''
        '''
        res = B_half - A_half @ phi_half
        res_norm = np.linalg.norm(res)
        np.set_printoptions(suppress=True, precision=4)
        self.lib_print("[%s] res_norm = %f\n\tres = %s.T" % (name, res_norm, str(res.T)))
        np.set_printoptions(suppress=False, precision=8)
        return (res, res_norm)

    def cal_res_all(self, A_x, B_x, phi_x_est, A_y, B_y, phi_y_est):
        '''
        '''
        res_x, res_norm_x = self.cal_res_half(A_x, B_x, phi_x_est, name='x')
        res_y, res_norm_y = self.cal_res_half(A_y, B_y, phi_y_est, name='y')
        res_norm_all = np.sqrt(res_norm_x**2 + res_norm_y**2)
        return (res_norm_all, res_x, res_norm_x, res_y, res_norm_y)
    #-------------------------------------------#

    # Formulation 2
    #-------------------------------------------#
    def f2_get_P(self, np_point_3d_dict_in):
        '''
        Constant, only changed when golden pattern changed
        P = [theta[0], ..., theta[n-1]].T, shape=(n,3)
        '''
        P_list = [np_point_3d_dict_in[_k].T for _k in np_point_3d_dict_in]
        P = np.vstack(P_list)
        return P

    def f2_get_D_from_P(self, P):
        '''
        Constant, only changed when golden pattern changed
        D = [P|1], shape=(n,4)
        '''
        _n = P.shape[0]
        D = np.hstack([P, np.ones((_n,1))])
        return D

    def f2_get_D_pinv(self, D):
        '''
        Constant, only changed when golden pattern changed
        D_pinv = (D.T @ D)^-1 @ D if D is full rank, shape=(4,n)
        '''
        D_pinv = np.linalg.pinv(D)
        return D_pinv

    def f2_get_B_xy(self, np_point_image_dict_in, K_inv_in):
        '''
        Change with sample, constant in iteration
        B_half = [nu_half[0], ..., nu_half[n-1]].T, shape=(n,1)
        '''
        # To be more realistic, use the image point to search
        _n = len(np_point_image_dict_in)
        B_x_list = list()
        B_y_list = list()
        for _k in np_point_image_dict_in:
            nu_i = K_inv_in @ np_point_image_dict_in[_k] # shape = (3,1)
            B_x_list.append(nu_i[0])
            B_y_list.append(nu_i[1])
        B_x = np.vstack(B_x_list).reshape((_n,1))
        B_y = np.vstack(B_y_list).reshape((_n,1))
        return (B_x, B_y)

    def f2_get_v_phi_o_half(self, D_pinv, B_half):
        '''
        Change with sample, constant in iteration
        v_phi_o_half = D_pinv @ B_half, shape=(4,1)
        '''
        v_phi_o_half = D_pinv @ B_half
        return v_phi_o_half

    def f2_get_M_half(self, D_pinv, B_half, P):
        '''
        Change with sample, constant in iteration
        M_half = D_pinv @ (diag_B_half @ P), shape=(4,3)
        '''
        M_half = D_pinv @ (B_half * P) # Note: broadcast in column to get the element-wise product.
        return M_half

    def f2_get_Q(self, D, D_pinv):
        '''
        '''
        _n = D.shape[0]
        Q = np.eye(_n) - (D @ D_pinv)
        return Q

    def f2_solve_phi_half_from_phi_3(self, v_phi_o_half, M_half, phi_3_est, name='half'):
        '''
        '''
        v_phi_half = v_phi_o_half + M_half @ phi_3_est
        self.lib_print("v_phi_half [%s] = %s.T" % (name, str(v_phi_half.T)))
        return v_phi_half

    def f2_get_Delta_bar(self, P, phi_3_est):
        '''
        shape=(n,1)
        '''
        _n = P.shape[0]
        Delta_bar = 1.0 + P @ phi_3_est
        return Delta_bar

    def f2_cal_res_half(self, B_half, D, v_phi_half, Delta_bar, name='half', is_f1_res=False):
        '''
        is_f1_res: Determine if using the residual of the formulation 1 (f1)
                    - True: Use f1's residual
                    - False: Use f2's residual
        '''
        if is_f1_res:
            res = B_half - (D @ v_phi_half) / Delta_bar
        else:
            res = (B_half * Delta_bar) - (D @ v_phi_half)
        res_norm = np.linalg.norm(res)
        np.set_printoptions(suppress=True, precision=4)
        self.lib_print("[%s] res_norm = %f\n\tres = %s.T" % (name, res_norm, str(res.T)))
        np.set_printoptions(suppress=False, precision=8)
        return (res, res_norm)

    def f2_cal_res_all(self, D, P, B_x, B_y, v_phi_x, v_phi_y, phi_3_est, is_f1_res=False):
        '''
        '''
        Delta_bar = self.f2_get_Delta_bar(P, phi_3_est)
        res_x, res_norm_x = self.f2_cal_res_half(B_x, D, v_phi_x, Delta_bar, name='x', is_f1_res=is_f1_res)
        res_y, res_norm_y = self.f2_cal_res_half(B_y, D, v_phi_y, Delta_bar, name='y', is_f1_res=is_f1_res)
        res_norm_all = np.sqrt(res_norm_x**2 + res_norm_y**2)
        return (res_norm_all, res_x, res_norm_x, res_y, res_norm_y)
    #-------------------------------------------#

    # Constrained optimization: Lagrange multiplier + Newton-Raphson method
    #-------------------------------------------#
    def co_prepare_matrix_components(self, B_x, B_y, P):
        '''
        '''
        n_point = P.shape[0] # The number of row of P
        BxP = B_x * P # Broadcast through column to get element-wise product.
        ByP = B_y * P # Broadcast through column to get element-wise product.
        one_n = np.ones((n_point,1)) # Column vector

        # 3x3 square matrices
        W = P.T @ P
        Ux = P.T @ BxP
        Uy = P.T @ ByP
        # Vx = BxP.T @ BxP
        # Vy = ByP.T @ ByP
        VxpVy = BxP.T @ BxP + ByP.T @ ByP # We only need (Vx + Vy), instead of separated Vx and Vy

        # 3x1 column vectors
        PTone = P.T @ one_n
        PTBx = P.T @ B_x
        PTBy = P.T @ B_y
        PTBxBxByBy = P.T @ (B_x*B_x + B_y*B_y) # Element-wise product

        # 1x1 scalar
        BxTone = B_x.T @ one_n
        ByTone = B_y.T @ one_n

        return (W, Ux, Uy, VxpVy, PTone, PTBx, PTBy, PTBxBxByBy, BxTone, ByTone)

    def co_get_function_value_and_Jacobian(self, co_x, co_A_list, co_bT_list, co_c_list):
        '''
        '''
        # The container for fx and Jf
        fx_list = list()
        Jf_list = list()

        #
        for _idx in range(len(co_A_list)):
            fx_list.append( ( co_x.T @ co_A_list[_idx] @ co_x + co_bT_list[_idx] @ co_x + co_c_list[_idx] ) )
            Jf_list.append( ( co_x.T @ (co_A_list[_idx] + co_A_list[_idx].T) + co_bT_list[_idx] ) )
        #

        # Generate fx and Jf
        fx = np.array(fx_list).reshape((11,1)) # 11x1
        Jf = np.vstack(Jf_list) # 11x11
        return (fx, Jf)

    # def co_get_function_value_and_Jacobian(self, co_x, co_A_list, co_bT_list, co_c_list):
    #     '''
    #     '''
    #     phi_1 = co_x[0:3,:]
    #     phi_2 = co_x[3:6,:]
    #     phi_3 = co_x[6:9,:]
    #     gamma_2_123 = phi_1.T @ phi_1
    #     # gamma_2_123 = phi_2.T @ phi_2
    #     # gamma_2_123 = phi_3.T @ phi_3
    #     # gamma_2_123 = (phi_1.T @ phi_1 + phi_2.T @ phi_2 +  phi_3.T @ phi_3)
    #     # The container for fx and Jf
    #     fx_list = list()
    #     Jf_list = list()
    #
    #     #
    #     for _idx in range(len(co_A_list)):
    #         _fx = ( co_x.T @ co_A_list[_idx] @ co_x + co_bT_list[_idx] @ co_x + co_c_list[_idx] )
    #         _Jf = ( co_x.T @ (co_A_list[_idx] + co_A_list[_idx].T) + co_bT_list[_idx] )
    #         if _idx < 4:
    #             _Jf /= gamma_2_123
    #             _fx_gamma_123 = _fx / (gamma_2_123**2)
    #             _Jf -= np.hstack( [ (_fx_gamma_123 * (phi_1.T)), np.zeros((1,8)) ])
    #             # _Jf -= np.hstack( [ np.zeros((1,3)), (_fx_gamma_123 * (phi_2.T)), np.zeros((1,5)) ])
    #             # _Jf -= np.hstack( [ np.zeros((1,6)), (_fx_gamma_123 * (phi_3.T)), np.zeros((1,2)) ])
    #             # _Jf -= np.hstack( [ (_fx_gamma_123 * (phi_1.T)), (_fx_gamma_123 * (phi_2.T)), (_fx_gamma_123 * (phi_3.T)), np.zeros((1,2)) ])
    #             _fx /= gamma_2_123
    #         fx_list.append( _fx )
    #         Jf_list.append( _Jf )
    #     #
    #
    #     # Generate fx and Jf
    #     fx = np.array(fx_list).reshape((11,1)) # 11x1
    #     Jf = np.vstack(Jf_list) # 11x11
    #     return (fx, Jf)

    def co_reconstruct_R_t_m1(self, co_x):
        '''
        - Use co_x
        '''
        # Test
        #---------------------------------#
        Gamma_list = [co_x[0:3,:].T, co_x[3:6,:].T, co_x[6:9,:].T]
        np_Gamma_est = np.vstack(Gamma_list)
        self.lib_print("np_Gamma_est = \n%s" % str(np_Gamma_est))
        G_u, G_s, G_vh = np.linalg.svd(np_Gamma_est)
        # self.lib_print("G_u = \n%s" % str(G_u))
        self.lib_print("G_s = \n%s" % str(G_s))
        # self.lib_print("G_vh = \n%s" % str(G_vh))
        G_D = np.linalg.det(G_u @ G_vh)
        self.lib_print("G_D = %f" % G_D)


        # Reconstruct R
        np_R_est = G_u @ np.diag([1.0, 1.0, G_D]) @ G_vh
        self.lib_print("np_R_est = \n%s" % str(np_R_est))
        # Convert to Euler angle
        Euler_angle_est = self.get_Euler_from_rotation_matrix(np_R_est, is_degree=True)
        self.lib_print("(roll, yaw, pitch) \t\t= %s" % str( Euler_angle_est ) ) # Note: Euler angles are in degree.
        # roll_est, yaw_est, pitch_est = Euler_angle_est

        # Reconstruct t vector
        # Get the "value" of G
        value_G = np.linalg.norm(np_Gamma_est, ord=2)
        self.lib_print("value_G = %f" % value_G)
        #
        t3_est = 1.0 / value_G
        # t3_est = 1.0 / np.average(G_s)
        self.lib_print("t3_est = %f" % t3_est)
        np_t_est = np.vstack((co_x[9:11,:], 1.0)) * t3_est
        self.lib_print("np_t_est = \n%s" % str(np_t_est))
        #---------------------------------#
        # end Test
        return (np_R_est, np_t_est, t3_est)
    #-------------------------------------------#

    # Constrained optimization: Lagrange multiplier + Newton-Raphson method
    #-------------------------------------------#
    def EKF_get_hx_H(self, ekf_x, B_x, B_y, P):
        '''
        '''
        n_point = P.shape[0]
        #
        phi_1 = ekf_x[0:3,:]
        phi_2 = ekf_x[3:6,:]
        phi_3 = ekf_x[6:9,:]
        delta_1 = ekf_x[9,0]
        delta_2 = ekf_x[10,0]
        #
        zeros_nx3 = np.zeros((n_point,3))
        zeros_nx1 = np.zeros((n_point,1))
        ones_nx1 = np.ones((n_point,1))
        H1 = np.hstack([P, zeros_nx3, (-B_x*P), ones_nx1, zeros_nx1])
        H2 = np.hstack([zeros_nx3, P, (-B_y*P), zeros_nx1, ones_nx1])
        # hx
        hx = np.zeros((2*n_point+5, 1))
        hx[:n_point,:] = H1 @ ekf_x # P @ phi_1 - (B_x*P) + delta_1
        hx[n_point:(2*n_point),:] = H2 @ ekf_x # P @ phi_2 - (B_y*P) + delta_2
        hx[(2*n_point),0] = phi_1.T @ phi_3
        hx[(2*n_point+1),0] = phi_2.T @ phi_3
        hx[(2*n_point+2),0] = phi_1.T @ phi_2
        hx[(2*n_point+3),0] = phi_1.T @ phi_1 - phi_3.T @ phi_3
        hx[(2*n_point+4),0] = phi_2.T @ phi_2 - phi_3.T @ phi_3
        # Hx
        Hx = np.zeros( (2*n_point+5, 11))
        Hx[:n_point, :] = H1
        Hx[n_point:(2*n_point), :] = H2
        # Hg, 1
        Hx[(2*n_point),0:3] = phi_3.T
        Hx[(2*n_point),6:9] = phi_1.T
        # Hg, 2
        Hx[(2*n_point+1),3:6] = phi_3.T
        Hx[(2*n_point+1),6:9] = phi_2.T
        # Hg, 3
        Hx[(2*n_point+2),0:3] = phi_2.T
        Hx[(2*n_point+2),3:6] = phi_1.T
        # Hg, 4
        Hx[(2*n_point+3),0:3] = phi_1.T
        Hx[(2*n_point+3),6:9] = -phi_3.T
        # Hg, 5
        Hx[(2*n_point+4),3:6] = phi_2.T
        Hx[(2*n_point+4),6:9] = -phi_3.T
        #
        return (hx, Hx)
    #-------------------------------------------#


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

    def reconstruct_R_t_block_reconstruction(self, phi_est, phi_3_est, res_grow_det_list=None):
    # def reconstruct_R_t_block_reconstruction(self, phi_est):
        '''
        Reconstruct the G = gamma*R using only the 2x2 block of the element in G
        while solving the scale: gamma at the same time.

        G = | K         beta |
            | alpha.T   c    |
          = gamma * R

        res_grow_det_list = (D_x, B_x, D_y, B_y, phi_3_est, res_old_x, res_old_y)
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
        # _e_2 = _k2**2 / (_gamma_2 - _k3)
        # _e_se = np.sqrt(_e_2) # !! What is teh sign?
        # _d_se = -(_k2/_e_se)
        # # _d_2 = _d_se**2
        #
        # New solution for avoiding the sigularity
        _e_2 = _gamma_2 - _k1
        _e_se = np.sqrt(_e_2) # !! What is the sign?
        _d_2 = _gamma_2 - _k3
        _d_se = -np.sign(_k2)*np.sqrt(_d_2) # !! What is the sign?
        self.lib_print("(_e_2, _e_se) = %s" % str((_e_2, _e_se)) )
        self.lib_print("(_d_2, _d_se) = %s" % str((_d_2, _d_se)) )
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
        _similarity = _alpha_se[:,0].dot(_alpha_lsq)
        # _similarity_angle = np.arccos(self.unit_vec(_alpha_se[:,0]).dot(self.unit_vec(_alpha_lsq))) * 180.0 / np.pi # The angle
        self.lib_print("_similarity = %f" % _similarity)
        # _se = np.sign(_similarity)
        if _similarity < 0.0:
            _se = -1.0
        else:
            _se = 1.0
        # _beta_lsq_gamma = np.cross(np_Gamma_est[0, :], np_Gamma_est[1, :])[0:2]
        # self.lib_print("_beta_lsq_gamma.T = %s.T" % str(_beta_lsq_gamma))
        # _se = np.sign(_beta_se[:,0].dot(_beta_lsq_gamma))

        # test
        #------------------------#
        if res_grow_det_list is not None:
            D_x, B_x, D_y, B_y, phi_3_est, res_old_x, res_old_y = res_grow_det_list
            _phi_3_est_new_p = np.vstack([_beta_se, _c]).reshape((3,1))
            _pi_x_p = D_x[:,0:3] @ _phi_3_est_new_p
            _pi_y_p = D_y[:,0:3] @ _phi_3_est_new_p
            _piB_x_p = _pi_x_p * B_x # Element wise
            _piB_y_p = _pi_y_p * B_y # Element wise
            _e_grow_p = -1 * (res_old_x.T @ _piB_x_p + res_old_y.T @ _piB_y_p)  # Note: the residual definition is different here

            _phi_3_est_new_n = np.vstack([-1.0*_beta_se, _c]).reshape((3,1))
            _pi_x_n = D_x[:,0:3] @ _phi_3_est_new_n
            _pi_y_n = D_y[:,0:3] @ _phi_3_est_new_n
            _piB_x_n = _pi_x_n * B_x # Element wise
            _piB_y_n = _pi_y_n * B_y # Element wise
            _e_grow_n = -1 * (res_old_x.T @ _piB_x_n + res_old_y.T @ _piB_y_n)  # Note: the residual definition is different here

            self.lib_print("_e_grow_p = %f" % _e_grow_p)
            self.lib_print("_e_grow_n = %f" % _e_grow_n)

            if _e_grow_n < _e_grow_p:
                _se = -1.0
            else:
                _se = 1.0

        #------------------------#

        _alpha = _se * _alpha_se
        _beta = _se * _beta_se
        self.lib_print(">>>>>>>>>>>>>>> _se = %f" % _se )
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
        if np.abs(np.pi/2.0 - np.arcsin(np.abs(R_in[2,1]))) <= _eps: # Faster
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
        # projection_no_q = _ray/_ray[2,0]
        projection_no_q = _ray/abs(_ray[2,0]) # Incase that the point is behind the camera
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


    def perspective_projection_golden_landmarks(self, np_R, np_t, is_quantized=False, is_pretrans_points=False, is_returning_homogeneous_vec=True):
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
        _np_point_3d_dict = self.get_current_golden_pattern()
        _np_point_3d_pretransfer_dict = self.get_current_pretransfered_golden_pattern()
        for _k in _np_point_3d_dict:
            if is_pretrans_points:
                _point = _np_point_3d_pretransfer_dict[_k]
            else:
                _point = _np_point_3d_dict[_k]
            _np_point_image, _projection_no_q = self.perspective_projection(_point, self.np_K_camera_est, np_R, np_t, is_quantized=is_quantized, is_returning_homogeneous_vec=is_returning_homogeneous_vec)
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
    pnp_solver = PNP_SOLVER_A2_M3(np_K_camera_est, [point_3d_dict], verbose=True)

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
