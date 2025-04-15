#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIC4DVAR-LC
Copyright (C) 2025 INRAE

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
import numpy as np
from sic4dvar_functions import sic4dvar_calculations as calc
from pathlib import Path
from sic4dvar_functions.sic4dvar_calculations import verify_name_length
from sic4dvar_functions.sic4dvar_helper_functions import *
import math

def algo3(apr_array, Qwbm, params, SLOPEM1, valid_output, node_z, node_z_input, node_z_ini, node_x, last_node_for_integral, bathymetry_array, param_dict, reach_id='', reach_t=[], bb=9999.0, reliability='', Qsdev=0.0):
    node_w_simp = apr_array['node_w_simp']
    node_a = apr_array['node_a']
    node_p = apr_array['node_p']
    node_r = apr_array['node_r']
    node_xr = bathymetry_array['node_xr']
    node_yr = bathymetry_array['node_yr']
    if params.V32:
        nsobs = 0
        for ij in range(node_z.shape[1]):
            if params.useEXT:
                INdata = node_z[:, ij]
                index_temp = np.where(INdata.mask == False)
                index_temp2 = np.argwhere(np.isnan(INdata[index_temp]))
                if len(index_temp) > 0:
                    nsobs += len(index_temp[0])
                if len(index_temp2) > 0:
                    nsobs -= len(index_temp2[0])
            else:
                INdata = node_z_ini[:, ij]
                index_temp = np.where(INdata.mask == False)
                index_temp2 = np.argwhere(np.isnan(INdata[index_temp]))
                if len(index_temp) > 0:
                    nsobs += len(index_temp[0])
                if len(index_temp2) > 0:
                    nsobs -= len(index_temp2[0])
    Wmean, Qp, QM1, QM2 = ([], [], [], [])
    Wmin = 10000
    Wmean = []
    for i in range(len(node_w_simp)):
        Wmean.append(min(node_w_simp[i]))
        if np.min(node_w_simp[i]) < Wmin:
            Wmin = np.min(node_w_simp[i])
    Wmean = np.average(Wmean)
    Qp = params.algo_bounds[0][0]
    QM1 = Qwbm / Qp
    QM2 = Qwbm * Qp
    if params.qsdev_activate:
        QM1s = Qsdev / Qp
        QM2s = Qsdev * Qp
    Zb1 = params.algo_bounds[1][0]
    Zb2 = params.algo_bounds[1][1]
    dZb = (Zb1 - Zb2) / 20
    if -bb > Zb1:
        Zb1 = -bb
        dZb = (Zb1 - Zb2) / 20
    if abs(Zb1) > Wmean:
        Zb1 = -Wmean + dZb
        dZb = (Zb1 - Zb2) / 20
    logging.debug('Zb1, Zb2, dZb, bb:', Zb1, Zb2, dZb, bb)
    if params.pankaj_test:
        Zb1 = 0.0
        Zb2 = 0.0
    Km1 = params.algo_bounds[2][0]
    Km2 = params.algo_bounds[2][1]
    dKm = params.algo_bounds[2][2]
    I2m = 1 + int((Zb1 - Zb2) / dZb)
    I3m = 1 + int((Km2 - Km1) / dKm)
    trialps0_max = 0.0
    p_YMS_a = 0.0
    iexit = 0
    temp_i3 = []
    trialps0_max = 0.0
    shapeP = []
    shapeP += [[(Qwbm - QM1) / (QM2 - QM1), params.shape03]]
    if params.qsdev_activate:
        shapePs = []
        shapePs += [[(Qsdev - QM1s) / (QM2s - QM1s), params.shape03]]
    delayZB = (params.val1 - Zb2) / (Zb1 - Zb2)
    shapeP += [[delayZB, params.shape13]]
    delayK = (params.val2 - Km1) / (Km2 - Km1)
    shapeP += [[delayK, params.shape23]]
    kDim = params.kDim
    q_pdf_table = calc.betad(int(param_dict['q_parametrization']), kDim, shapeP[0][0], shapeP[0][1])
    if params.qsdev_activate:
        qs_pdf_table = calc.betad(int(param_dict['q_parametrization']), kDim, shapePs[0][0], shapePs[0][1])
    zb_pdf_table = calc.betad(0, kDim, shapeP[1][0], shapeP[1][1])
    km_pdf_table = calc.betad(0, kDim, shapeP[2][0], shapeP[2][1])
    step = 1 / (kDim - 1)
    value = 0
    q_pdf_table = np.array(q_pdf_table[1])
    if params.qsdev_activate:
        qs_pdf_table = np.array(qs_pdf_table[1])
    zb_pdf_table = np.array(zb_pdf_table[1])
    km_pdf_table = np.array(km_pdf_table[1])
    q_pdf_table = q_pdf_table + value
    if params.qsdev_activate:
        qs_pdf_table = qs_pdf_table + value
    q_positional_arg = np.arange(0, len(q_pdf_table))
    if params.qsdev_activate:
        qs_positional_arg = np.arange(0, len(qs_pdf_table))
    zb_pdf_table[0] = zb_pdf_table[1]
    zb_pdf_table[-1] = zb_pdf_table[len(km_pdf_table) - 2]
    km_pdf_table[0] = km_pdf_table[1]
    km_pdf_table[-1] = km_pdf_table[len(km_pdf_table) - 2]
    if param_dict['gnuplot_saving']:
        reach_id = str(reach_id)
        reach_id = verify_name_length(reach_id)
        reach_id = verify_name_length(reach_id)
        output_dir = param_dict['output_dir'].joinpath('gnuplot_data', reach_id)
        if not output_dir.is_dir():
            output_dir.mkdir(parents=True, exist_ok=True)
        gnuplot_save_tables(q_pdf_table, zb_pdf_table, km_pdf_table, output_dir.joinpath('tables'), 1)
    temp = []
    SS1_array = []
    if not valid_output:
        logging.debug('Slope is invalid. Stopping.')
        return (np.ones(node_z[0].shape[0]) * np.nan, 0)
    LSM = params.slope_smoothing_num_passes
    for k in range(LSM):
        SLOPEM2 = [0.75 * SLOPEM1[0] + 0.25 * SLOPEM1[1]]
        for t in range(1, node_z[0].shape[0] - 1):
            SLOPEM2 += [0.25 * SLOPEM1[t - 1] + 0.5 * SLOPEM1[t] + 0.25 * SLOPEM1[t + 1]]
        SLOPEM2 += [0.25 * SLOPEM1[-2] + 0.75 * SLOPEM1[-1]]
        SLOPEM1 = np.array(SLOPEM2)
    QM = np.zeros(node_z[0].shape[0])
    Nbeta = 0.0
    QM0 = 0.0
    QMT_array = np.zeros((I2m, node_z[0].shape[0]), dtype=np.float64)
    QMEAN_array = np.zeros(I2m, dtype=np.float64)
    temp_Q = np.zeros((len(node_z), node_z[0].shape[0]))
    temp_Q = []
    temp_Q_t = []
    if params.V32:
        Q_sample = np.zeros((I2m * I3m, node_z[0].shape[0]))
        Weight = np.zeros(I2m * I3m)
        iis = 0
        cost_all = []
    test_tmp = []
    for i2 in range(1, I2m + 1):
        Zb = np.zeros(len(node_z))
        temp_ss1_array = []
        for t in range(node_z[0].shape[0]):
            SS1 = 0.0
            temp_Q_t = []
            for n in range(last_node_for_integral):
                Zmin, Wmin = (np.nanmin(node_yr[n]), np.nanmin(node_xr[n]))
                if Wmin == 0:
                    Wmin = 1.0
                if t == 0:
                    Zb[n] = node_yr[n][0] + Zb2 + (i2 - 1) * dZb * (Wmean / Wmin)
                Z1 = node_yr[n][0]
                W1 = node_xr[n][0]
                if Zmin != Z1:
                    pass
                if Wmin != W1:
                    pass
                if Wmin == 0 or calc.check_na(Wmin):
                    pass
                if W1 == 0:
                    pass
                A1 = node_a[n][t]
                P1 = node_p[n][t]
                A0 = W1 * (Z1 - Zb[n])
                if calc.check_na(A0):
                    pass
                P0 = 2 * (Z1 - Zb[n])
                AA = A0 + A1
                PP = P0 + P1
                if AA < 0:
                    logging.info('Area (AA) must be > 0 !')
                if PP < 0:
                    logging.info('Perimeter (PP) must be > 0 !')
                R = AA / PP
                QMi2i3 = AA * math.pow(R, 2.0 / 3.0)
                temp_Q_t.append(QMi2i3)
                if n == 0:
                    QM0 = QMi2i3
                else:
                    if QM0 <= 0 or QMi2i3 <= 0:
                        print('n, t:', n, t)
                    if QM0 > 0 and QMi2i3 > 0:
                        SS1 += abs(node_x[n] - node_x[n - 1]) * (math.pow(QM0, -2) + math.pow(QMi2i3, -2)) / 2.0
                    QM0 = QMi2i3
            temp_Q.append(temp_Q_t)
            Sl = SLOPEM1[t]
            if Sl / abs(node_x[-1] - node_x[0]) < 1e-06:
                Sl = abs(node_x[-1] - node_x[0]) * 1e-06
            temp_ss1_array.append(SS1)
            QMi2i3 = math.sqrt(Sl / SS1)
            QMEAN_array[i2 - 1] += QMi2i3
            QMT_array[i2 - 1, t] = QMi2i3
            SS1_array.append(1 / SS1)
        if param_dict['gnuplot_saving']:
            if not output_dir.is_dir():
                output_dir.mkdir(parents=True, exist_ok=True)
            times2 = np.around(reach_t / 3600 / 24)
            times2 = times2 - min(times2)
            gnuplot_save_slope(temp_ss1_array, times2, output_dir.joinpath('integral'))
        SS1_array = []
        QMEAN_array /= node_z[0].shape[0]
        if params.qsdev_activate:
            Qsdev = 0.0
            for ist in range(0, node_z[0].shape[0]):
                Qsdev = Qsdev + (QMT_array[i2 - 1, ist] - QMEAN_array[i2 - 1]) ** 2
            Qsdev = np.sqrt(Qsdev / node_z[0].shape[0])
        QMT_const_array = np.copy(QMT_array)
        trialps1_max = 0.0
        for i3 in range(1, I3m + 1):
            theta = (Zb2 + (i2 - 1) * dZb - Zb2) / (Zb1 - Zb2)
            if theta < 0.0:
                theta = 0.0
            if theta > 1.0:
                theta = 1.0
            if not params.pankaj_test:
                Zb_pdf = zb_pdf_table[int(theta * (kDim - 1))]
            if params.pankaj_test:
                Zb_pdf = 1.0
            theta = (Km1 + (i3 - 1) * dKm - Km1) / (Km2 - Km1)
            if theta < 0.0:
                theta = 0.0
            if theta > 1.0:
                theta = 1.0
            Km_pdf = km_pdf_table[int(theta * (kDim - 1))]
            Kmi3 = Km1 + (i3 - 1) * dKm
            SS1 = QMEAN_array[i2 - 1] * Kmi3
            if params.qsdev_option == 0:
                SS1_qs = Qsdev * Kmi3
            elif params.qsdev_option == 1:
                SS1_qs = Qsdev / QMEAN_array[i2 - 1]
            theta = (SS1 - QM1) / (QM2 - QM1)
            if params.qsdev_activate:
                theta_qs = (SS1_qs - QM1s) / (QM2s - QM1s)
            if theta <= 0 or calc.check_na(theta):
                theta = 1e-06
            if theta >= 1:
                theta = 1
            if params.qsdev_activate:
                if theta_qs <= 0 or calc.check_na(theta_qs):
                    theta_qs = 1e-06
                if theta_qs >= 1:
                    theta_qs = 1
            Q_pdf = interp_pdf_tables(kDim - 1, theta * kDim, q_positional_arg, q_pdf_table)
            if params.qsdev_activate:
                Qs_pdf = interp_pdf_tables(kDim - 1, theta_qs * kDim, qs_positional_arg, qs_pdf_table)
            if params.V32:
                Qtrial = theta
                if theta > 1.0:
                    Qtrial = 1.0
                if theta < 0.0:
                    Qtrial = 0.0
                QMT_array[i2 - 1, :] = QMT_const_array[i2 - 1, :] / QMEAN_array[i2 - 1] * (Qtrial * QM2 + (1.0 - Qtrial) * QM1)
            else:
                QMT_array[i2 - 1, :] = QMT_const_array[i2 - 1, :] * SS1 / QMEAN_array[i2 - 1]
            if not params.qsdev_activate:
                Nbeta += Zb_pdf * Km_pdf * Q_pdf
            if params.qsdev_activate:
                Nbeta += Zb_pdf * Km_pdf * Q_pdf * Qs_pdf
            trial_ps0 = Zb_pdf * Km_pdf * Q_pdf
            ss_a = trial_ps0
            p_YMS_a = p_YMS_a + ss_a
            if not params.qsdev_activate:
                QM = QM + QMT_array[i2 - 1, :] * Zb_pdf * Km_pdf * Q_pdf
            if params.qsdev_activate:
                QM = QM + QMT_array[i2 - 1, :] * Zb_pdf * Km_pdf * Q_pdf * Qs_pdf
            if params.a31_early_stop:
                if trial_ps0 > trialps1_max:
                    trialps1_max = trial_ps0
                if trial_ps0 > trialps0_max:
                    trialps0_max = trial_ps0
                if trial_ps0 / (trialps0_max + 1e-16) < 0.001 and trial_ps0 < trialps1_max and (i3 > 0):
                    break
            if params.V32:
                NBS = len(node_z[:, :]) - 1
                icost = 0.0
                cost = 0.0
                Q_sample[iis] = QMT_array[i2 - 1, :]
                Weight[iis] = Zb_pdf * Km_pdf * Q_pdf
                TDIM = len(Q_sample[iis, :])
                for ij in range(0, TDIM):
                    if Zb_pdf > 0.0 and Km_pdf > 0.0 and (Q_pdf > 0.0) and (Weight[iis] > 0.0):
                        NBT = len([Q_sample[iis, ij]]) - 1
                        if not node_z_ini[NBS, ij]:
                            icost += 0.0
                        else:
                            if params.useEXT:
                                Z_SS, A_SS, P_SS, R_SS, cost = calc.compute_steady_state(NBS, NBT, node_xr, node_yr, node_z_ini[:, ij], node_x, Zb, Kmi3, [Q_sample[iis, ij]])
                            else:
                                Z_SS, A_SS, P_SS, R_SS, cost = calc.compute_steady_state(NBS, NBT, node_xr, node_yr, node_z_ini[:, ij], node_x, Zb, Kmi3, [Q_sample[iis, ij]])
                            icost += cost
                if Zb_pdf > 0.0 and Km_pdf > 0.0 and (Q_pdf > 0.0) and (icost > 0.0):
                    cost_all += icost.tolist()
                else:
                    cost_all += [np.nan]
                iis += 1
        temp_i3.append(i3)
        if params.a31_early_stop:
            if trialps1_max / (trialps0_max + 1e-16) < 0.001:
                iexit = 1
            if i2 > 1 and iexit == 1:
                break
    Q_min = QMT_array[0, :] * Km1
    Q_max = QMT_array[-1, :] * Km2
    if param_dict['gnuplot_saving']:
        reach_id = str(reach_id)
        reach_id = verify_name_length(reach_id)
        reach_id = verify_name_length(reach_id)
        node_x = node_x
        test_t_array = reach_t
        nodes2 = (node_x - node_x[0]) / 1000
        times2 = test_t_array / 3600 / 24
        if not output_dir.is_dir():
            output_dir.mkdir(parents=True, exist_ok=True)
        gnuplot_save_q(QM, times2, output_dir.joinpath('QM'))
    if Nbeta == 0:
        logging.info('Nbeta = 0.0, discharge will be incorrect ! Valid set of solutions is empty, observations and qwbm are not consistent !')
        if param_dict['q_min_modif']:
            QM = Q_min
            reliability = 'unreliable'
        else:
            valid_output = 0
            reliability = 'invalid'
            QM = np.ones(node_z[0].shape[0]) * np.nan
    if Nbeta != 0:
        QM /= Nbeta
    if params.V32:
        print('nsobs=', nsobs)
        if nsobs != 0 and (not calc.check_na(nsobs)):
            lcmax = int(math.log(nsobs) / math.log(2.0)) + 1
        else:
            lcmax = np.nan
        distp0 = []
        lc0 = []
        lc0 += [lcmax]
        id = 0
        distp, LH_pdf, QPost = calc.likelihood_2(TDIM, nsobs, lc0[id], Q_sample, QM, cost_all, Weight)
        distp0 += [distp]
        while distp < params.thres and distp >= distp0[id] and (not calc.check_na(distp)):
            id += 1
            lc0 += [lc0[id - 1] + 1]
            distp, LH_pdf, QPost = calc.likelihood_2(TDIM, nsobs, lc0[id], Q_sample, QM, cost_all, Weight)
            distp0 += [distp]
        while distp >= params.thres and distp <= distp0[id] and (not calc.check_na(distp)):
            id += 1
            lc0 += [lc0[id - 1] - 1]
            distp, LH_pdf, QPost = calc.likelihood_2(TDIM, nsobs, lc0[id], Q_sample, QM, cost_all, Weight)
            distp0 += [distp]
        ilc = min(range(len(distp0)), key=lambda k: abs(distp0[k] - 1.0))
        lc = lc0[ilc]
        distp, LH_pdf, QPost = calc.likelihood_2(TDIM, nsobs, lc, Q_sample, QM, cost_all, Weight)
        QM = QPost
    for t in range(node_z[0].shape[0]):
        if QM[t] < params.local_QM1:
            QM[t] = params.local_QM1
    mask_qm = np.ones(len(QM), dtype=bool)
    for i in range(len(QM)):
        if not calc.check_na(QM[i]):
            mask_qm[i] = False
    QM_masked = np.ma.array(QM, mask=mask_qm)
    if param_dict['gnuplot_saving']:
        reach_id = str(reach_id)
        reach_id = verify_name_length(reach_id)
        reach_id = verify_name_length(reach_id)
        node_x = node_x
        test_t_array = reach_t
        nodes2 = (node_x - node_x[0]) / 1000
        times2 = test_t_array / 3600 / 24
        if not output_dir.is_dir():
            output_dir.mkdir(parents=True, exist_ok=True)
        gnuplot_save_q(QM, times2, output_dir.joinpath('qalgo31'))
    if param_dict['gnuplot_saving']:
        reach_id = str(reach_id)
        reach_id = verify_name_length(reach_id)
        reach_id = verify_name_length(reach_id)
        output_dir = output_dir.joinpath('gnuplot_data', reach_id)
        node_x = node_x
        test_t_array = reach_t
        nodes2 = (node_x - node_x[0]) / 1000
        times2 = test_t_array / 3600 / 24
        if not output_dir.is_dir():
            output_dir.mkdir(parents=True, exist_ok=True)
    if False:
        plt.plot(QM_masked)
        plt.show()
        plt.clf()
        print(bug)
    return (QM_masked, valid_output, reliability)
