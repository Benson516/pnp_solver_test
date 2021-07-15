import numpy as np
import copy
import time
import csv
# import json
# import heapq
#
import cv2
#
# import PNP_SOLVER_LIB as PNPS
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
# end Integration tools



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
        print("\n*** Wrote the statistic results [%s] to the csv file:\n\t[%s]\n" % ( _hv_adv_str, csv_path))
    return row_dict_list
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
