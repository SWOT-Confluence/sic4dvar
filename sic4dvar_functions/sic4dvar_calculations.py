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
import sic4dvar_params as params
if params.cython_version:
    import sic4dvar_cyfuncs as cy
import numpy as np
import pandas as pd
import scipy.signal as scisig
import scipy.stats as scistats
import math
import cmath
import logging
import inspect
from pathlib import Path
import matplotlib.pyplot as plt
from sic4dvar_functions.j.z753 import M
from lib.lib_verif import verify_name_length, check_na
try:
    from Confluence.input.input.extract.CalculateHWS import CalculateHWS
    from Confluence.input.input.extract.DomainHWS import DomainHWS
    from Confluence.input.input.extract.HWS_IO import HWS_IO
    Confluence_HWS_method = True
except ImportError:
    Confluence_HWS_method = False

def ManningLW(a0, abar, w, s, n):
    return n ** (-1) * (a0 + abar) ** (5 / 3) * w ** (-2 / 3) * abs(s) ** (1 / 2)

def DarcyW(a0, abar, w, s, cf):
    return 1 / math.pow(cf, 1 / 2) * math.pow(a0 + abar, 3 / 2) * math.pow(w, -1 / 2) * math.pow(9.81 * abs(s), 1 / 2)

def ManningVK(a0, abar, w, s, alpha, beta, z=[]):
    w = np.maximum(10, w)
    h = (a0 + abar) / w
    n = alpha * h ** beta
    return ManningLW(a0, abar, w, s, n)

def objective_internal_data_Q_any(_x, *_params):
    parameters_dict = {'q': _params[1][0], 'abar': _params[1][1], 'w': _params[1][2], 's': _params[1][3]}
    if len(_params[1]) > 4:
        parameters_dict['z'] = _params[1][4]
    bounds_dict = {}
    for j in range(0, len(_params[2])):
        bounds_dict[_params[2][j]] = _x[j]
    params_func = inspect.signature(globals()[_params[0]]).parameters
    param_names = list(params_func.keys())
    function_params = np.zeros(len(param_names))
    sum = 0.0
    for t in range(len(parameters_dict['q'])):
        print('T=', t)
        for j, parameter in enumerate(param_names):
            index_param = np.where(np.array(param_names) == parameter)[0]
            if np.array(index_param).size > 0:
                if parameter in parameters_dict:
                    function_params[j] = parameters_dict[parameter][t]
                elif parameter in bounds_dict:
                    function_params[j] = bounds_dict[parameter]
        func_params_dict = dict(zip(param_names, function_params))
        suitable = check_suitability_any(bounds_dict, parameters_dict, _params[0], t)
        if suitable:
            result = globals()[_params[0]](**func_params_dict)
        else:
            pass
        if suitable:
            sum += (parameters_dict['q'][t] - result) ** 2
        else:
            pass
    return sum

def check_suitability(_n, _A_bar, _A_prime, _W, _S):
    if _n < 0:
        print('_n < 0')
        return False
    if _A_bar + _A_prime < 0:
        print("_A_bar + A' < 0")
        return False
    if _W < 0:
        print('_W < 0')
        return False
    if _S <= 0:
        return False
    return True

def check_suitability_any(bounds_dict, parameters_dict, equation, t):
    params_func = inspect.signature(globals()[equation]).parameters
    param_names = list(params_func.keys())
    function_params = np.zeros(len(param_names))
    for j, parameter in enumerate(param_names):
        index_param = np.where(np.array(param_names) == parameter)[0]
        if np.array(index_param).size > 0:
            if parameter in parameters_dict:
                function_params[j] = parameters_dict[parameter][t]
            elif parameter in bounds_dict:
                function_params[j] = bounds_dict[parameter]
    func_params_dict = dict(zip(param_names, function_params))
    if 'n' in func_params_dict:
        if func_params_dict['n'] < 0:
            print('n < 0')
            return False
    if 'cf' in func_params_dict:
        if func_params_dict['cf'] < 0:
            print('cf < 0')
            return False
    if 'abar' in func_params_dict:
        if func_params_dict['abar'] < params.valid_min_dA or check_na(func_params_dict['abar']):
            print('Abar < valid_min_dA or Abar is NaN')
            return False
    if 'a0' in func_params_dict:
        if check_na(func_params_dict['a0']):
            print('A0 is NaN')
            return False
    if 'a0' in func_params_dict and 'abar' in func_params_dict:
        if func_params_dict['a0'] + func_params_dict['abar'] < 0:
            print('Abar + A < 0')
            return False
    else:
        print('A0 or Abar unavailable, should never happen.')
        print(bug)
    if 'w' in func_params_dict:
        if func_params_dict['w'] < 0 or check_na(func_params_dict['w']):
            print('W <= 0 or W is NaN')
            return False
    if 's' in func_params_dict:
        if func_params_dict['s'] <= 0 or check_na(func_params_dict['s']):
            pass
            print('S <= 0 or S is NaN')
            return False
    return True

def filter_pom(weights, Y1, idist):
    nbw = weights.size
    nby = Y1.size
    ymax = Y1.max()
    ymin = Y1.min()
    if ymax > ymin:
        ry = math.pow(ymax - ymin, -2)
    else:
        ry = 1
    rx = math.pow(nbw, -2)
    imax = np.argmax(Y1)
    imin = np.argmin(Y1)
    Y2 = []
    for i in range(nby):
        Y2 += [0]
        sumw = 0
        k = 0
        for j in range(i - math.floor((nbw - 1) / 2), i + math.floor((nbw - 1) / 2)):
            k += 1
            if j >= 1 and j < nby:
                if idist == 0:
                    poids = weights[k]
                elif idist == 1:
                    if j == 1:
                        poids = weights[k]
                    else:
                        poids = math.sqrt(ry * (Y1[i] - Y1[j]) ** 2 + rx * (i - j) ** 2)
                        poids = weights[k] / poids
                elif idist == 2:
                    if j == 1:
                        poids = 1
                    else:
                        poids = math.sqrt(ry * (Y1[i] - Y1[j]) ** 2 + rx * (i - j) ** 2)
                        poids = 1 / poids
                Y2[i] = Y2[i] + poids * Y1[j]
                sumw = sumw + poids
        Y2[i] = Y2[i] / sumw
    Y2 = np.array(Y2)
    return Y2

def f_approx_sections_v6(X, Y, NbIter=4, SeuilDistance=0.1, FSort=2):
    debug = 0
    dmax = []
    list_to_keep = []
    for i in range(len(X)):
        if X[i] != np.nan and Y[i] != np.nan:
            list_to_keep += [i]
    X_tmp, Y_tmp = (np.zeros_like(X), np.zeros_like(Y))
    X_tmp = X[list_to_keep]
    Y_tmp = Y[list_to_keep]
    X = np.copy(X_tmp)
    Y = np.copy(Y_tmp)
    if FSort == 0:
        Xt, Yt = (np.copy(X), np.copy(Y))
    elif FSort == 1:
        Yt = np.sort(Y)
        indX = np.argsort(Y)
        Xt = np.take_along_axis(X, indX, axis=0)
    elif FSort == 2:
        Yt = np.sort(Y)
        indY = np.argsort(Y)
        Xt = np.sort(X)
        indX = np.argsort(X)
    elif FSort == 3:
        Xt = np.sort(X)
        indX = np.argsort(X)
        Yt = np.take_along_axis(Y, indX, axis=0)
    elif FSort == 4:
        Yt = np.sort(Y)
        indY = np.argsort(Y)
        Xt = np.take_along_axis(X, indY, axis=0)
        pol = np.polyfit(Xt, Yt, 7)
        Yt = np.polyval(pol, pol)
    elif FSort == 5:
        Yt = np.sort(Y)
        indY = np.argsort(Y)
        Xt = np.take_along_axis(X, indY, axis=0)
        windowSize = 10
        kernel = np.ones(windowSize) / windowSize
        Yt = scisig.lfilter(kernel, 1, Yt)
    elif FSort == 6:
        Yt = np.sort(Y)
        indY = np.argsort(Y)
        Xt = np.take_along_axis(X, indY, axis=0)
        idist = 1
        weights = np.append(np.arange(0.1, 1.1, 0.1), np.arange(0.9, 0.0, -0.1))
        Xt = filter_pom(weights, Xt, idist)
        Yt = filter_pom(weights, Yt, idist)
    elif FSort == 7:
        idist = 1
        weights = np.append(np.arange(0.1, 1.1, 0.1), np.arange(0.9, 0.0, -0.1))
        Xt = filter_pom(weights, X, idist)
        Yt = filter_pom(weights, Y, idist)
        Yt = np.sort(Yt)
        indY = np.argsort(Yt)
        Xt = np.take_along_axis(Xt, indY, axis=0)
    elif FSort == 8:
        Xt = np.average(X, axis=0, weights=0.1)
        Yt = np.average(Y, axis=0, weights=0.1)
        Yt = np.sort(Yt)
        indY = np.argsort(Yt)
        Xt = np.take_along_axis(Xt, indY, axis=0)
    nb = Xt.size
    C = np.array([0])
    for i in range(1, nb):
        C = np.append(C, [C[i - 1] + math.sqrt(math.pow(Xt[i] - Xt[i - 1], 2) + math.pow(Yt[i] - Yt[i - 1], 2))])
    N = np.zeros(nb)
    N[0] = 1.0
    if nb > 1:
        N[nb - 1] = 2.0
    if SeuilDistance <= 0 and NbIter <= 0:
        NbIter = nb
        SeuilDistance = 0
    elif SeuilDistance > 0 and NbIter <= 0:
        NbIter = nb - 1
    D = np.zeros(nb)
    Err = np.zeros(nb)
    count = 0
    for itera in range(NbIter):
        nam = np.nan
        nav = np.nan
        for i in range(nb):
            count += 1
            if nam is np.nan:
                if N[i] > 0:
                    nam = i
                    D[i] = 0
                    Err[i] = -5
            elif nav is np.nan:
                if N[i] > 0:
                    nav = i
            if nam is not np.nan and nav is not np.nan:
                x1, y1 = (Xt[nam], Yt[nam])
                x2, y2 = (Xt[nav], Yt[nav])
                a = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if a > 0:
                    for j in range(nam, nav):
                        x, y = (Xt[j], Yt[j])
                        d1c = (x1 - x) ** 2 + (y1 - y) ** 2
                        d2c = (x2 - x) ** 2 + (y2 - y) ** 2
                        c = (d1c - d2c) / a
                        a1 = (a + c) / 2
                        a2 = (a - c) / 2
                        dc1 = d1c - a1 ** 2
                        dc2 = d2c - a2 ** 2
                        if abs(dc1 - dc2) < 1e-10:
                            if abs(dc1) < 1e-10:
                                D[j] = 0
                                Err[j] = 0
                            elif dc1 >= 0:
                                D[j] = math.sqrt(dc1)
                                Err[j] = 0
                            else:
                                D[j] = 0
                                Err[j] = -4
                        else:
                            D[j] = (dc1 + dc2) / 2
                            Err[j] = -1
                        if a1 < -9e-06:
                            D[j] = math.sqrt(d1c)
                            Err[j] = -2
                        elif a2 < -9e-06:
                            D[j] = math.sqrt(d2c)
                            Err[j] = -3
                else:
                    for j in range(nam, nav):
                        Err[j] = -5
                        D[j] = 0
                nam = nav
                nav = np.nan
        k = np.argmax(D)
        d = D[k]
        if N[k] == 0:
            N[k] = itera + 2
            dmax += [d]
        if d <= SeuilDistance:
            break
    indN = []
    for i in range(len(N)):
        if N[i] > 0:
            indN += [i]
    indN = np.array(indN)
    Xr = Xt[indN]
    Yr = Yt[indN]
    if FSort > 0:
        isY_inv = []
        for i in range(len(Xt)):
            isY_inv += [np.where(indY == i)[0][0]]
        C = C[isY_inv]
        N = N[isY_inv]
        D = D[isY_inv]
        Err = Err[isY_inv]
    return (Xr, Yr, C, N, D, dmax, Err, Xt, Yt)

def fnc_APR(_z, _Ws, _Zs):
    nz = len(_z)
    ns = len(_Zs) - 1
    _Ws = _Ws / 2
    A = np.zeros(nz)
    W = np.zeros(nz)
    P = np.zeros(nz)
    R = np.zeros(nz)
    w = np.zeros(nz)
    for i in range(nz):
        A[i] = 0
        w[i] = _Ws[0]
        if _z[i] >= _Zs[0]:
            P[i] = _Ws[0]
        else:
            P[i] = _Ws[0]
        j = 0
        if i == 60:
            pass
        while j < ns and _z[i] > _Zs[j + 1]:
            A[i] = A[i] + (_Ws[j] + _Ws[j + 1]) * (_Zs[j + 1] - _Zs[j])
            P[i] = P[i] + math.sqrt((_Ws[j + 1] - _Ws[j]) ** 2 + (_Zs[j + 1] - _Zs[j]) ** 2)
            j += 1
        if _z[i] > _Zs[j]:
            if j < ns:
                w[i] = _Ws[j] + (_Ws[j + 1] - _Ws[j]) * (_z[i] - _Zs[j]) / (_Zs[j + 1] - _Zs[j])
            else:
                w[i] = _Ws[ns]
            A[i] = A[i] + (_Ws[j] + w[i]) * (_z[i] - _Zs[j])
            P[i] = P[i] + math.sqrt((w[i] - _Ws[j]) ** 2 + (_z[i] - _Zs[j]) ** 2)
        if P[i] <= 0:
            pass
        P[i] = P[i] * 2
        w[i] = w[i] * 2
        if P[i] > 0:
            R[i] = A[i] / P[i]
        else:
            R[i] = 0
    return (A, P, R, w)

def betad(i_type, k, delay, sp):
    step = 1 / (k - 1)
    cbet = 0.1
    yarg_gm = np.zeros(k)
    bbet = 0.0
    if i_type == 0:
        abet = delay * (sp - 2.0) + 1.0
        bbet = (1.0 - delay) * (sp - 2.0) + 1.0
    elif i_type == 1:
        abet = delay * sp
        bbet = (1.0 - delay) * sp
    else:
        print('')
    yarg_gm = wgh3(k, abet, bbet, cbet)
    ym = 0.0
    for i in range(1, k):
        if yarg_gm[i - 1] > ym:
            ym = yarg_gm[i - 1]
    return (ym, yarg_gm, abet, bbet, cbet)

def betad_old(k, delay, sp):
    step = 1 / (k - 1)
    cbet = 0.1
    yarg_gm = np.zeros(k)
    ym = 0.0
    bbet = 0.0
    if delay < 0.5:
        bbet = sp
        dbet = 0.0
        xbet = 0.0
        while dbet <= 1.0 and xbet < delay:
            abet = 1.0 + dbet * bbet
            yarg_gm = wgh3(k, abet, bbet, cbet)
            ym = 0.0
            for i in range(1, k):
                if yarg_gm[i - 1] > ym:
                    xbet = (i - 1) * step
                    ym = yarg_gm[i - 1]
            dbet += 0.01
    else:
        abet = sp
        dbet = 0.0
        xbet = 1.0
        while dbet <= 1.0 and delay < xbet:
            bbet = 1.0 + dbet * abet
            yarg_gm = wgh3(k, abet, bbet, cbet)
            ym = 0.0
            for i in range(1, k):
                if yarg_gm[i - 1] > ym:
                    xbet = (i - 1) * step
                    ym = yarg_gm[i - 1]
            dbet += 0.01
    return (ym, yarg_gm, abet, bbet, cbet)

def wgh3(k, abet, bbet, cbet):
    step = 1 / (k - 1)
    yarg_gm = np.zeros(k)
    for i in range(1, k):
        teta = (i - 1) * step
        yarg_gm[i - 1] = ncbeta(teta, abet, bbet, cbet)
    return yarg_gm

def ncbeta(x, a, b, lam):
    const = np.divide(a, b)
    pz = lambda r: scistats.ncf.pdf(r * (const * (1 - r)) ** (-1), 2 * a, 2 * b, lam) * (const * (1 - r) ** 2) ** (-1)
    output = pz(x)
    if np.isnan(output):
        output = 0.0
    if np.isneginf(output):
        output = 0.0
    return output

def compute_steady_state(NBS, NBT, Xr_all, Yr_all, H_s, X_s, Zb, Ks, Qsimsic):
    Delta_H = []
    Fr2 = []
    cost = 0.0
    ind_t_flu_fin = NBT
    ind_t_flu_deb = -1
    ind_obj_fin = NBS
    ind_obj_deb = -1
    list_s_ss = np.arange(ind_obj_fin, ind_obj_deb, -1)
    list_t_ss = np.arange(ind_t_flu_fin, ind_t_flu_deb, -1)
    Q_ss = np.ones((len(list_s_ss), len(list_t_ss))) * np.nan
    Z_ss = np.ones((len(list_s_ss), len(list_t_ss))) * np.nan
    A_ss = np.ones((len(list_s_ss), len(list_t_ss))) * np.nan
    P_ss = np.ones((len(list_s_ss), len(list_t_ss))) * np.nan
    R_ss = np.ones((len(list_s_ss), len(list_t_ss))) * np.nan
    V_ss = np.ones((len(list_s_ss), len(list_t_ss))) * np.nan
    H_ss = np.ones((len(list_s_ss), len(list_t_ss))) * np.nan
    Zc_ss = np.ones((len(list_s_ss), len(list_t_ss))) * np.nan
    Ac_ss = np.ones((len(list_s_ss), len(list_t_ss))) * np.nan
    Pc_ss = np.ones((len(list_s_ss), len(list_t_ss))) * np.nan
    Hc_ss = np.ones((len(list_s_ss), len(list_t_ss))) * np.nan
    Lc_ss = np.ones((len(list_s_ss), len(list_t_ss))) * np.nan
    Iter_ss = np.ones((len(list_s_ss), len(list_t_ss))) * np.nan
    cG = 9.81
    c2G = 2 * cG
    c1s2G = 1 / c2G
    c43 = 4.0 / 3.0
    DH_Iter = params.DH_Iter
    DH_Esp = params.DH_Esp
    c2sG = 2 / cG
    for ind_s_ss in list_s_ss:
        Q_ss[ind_s_ss, list_t_ss] = Qsimsic[::-1]
        Xr = Xr_all[ind_s_ss]
        Yr = Yr_all[ind_s_ss]
        Yr[0] = Zb[ind_s_ss]
        if ind_s_ss == ind_obj_fin:
            Z_ss[ind_s_ss, list_t_ss] = H_s[ind_obj_fin]
            [A_ss[ind_s_ss, list_t_ss], P_ss_fic, R_ss[ind_s_ss, list_t_ss], w_fic] = fnc_APR(Z_ss[ind_s_ss, list_t_ss], Xr, Yr)
            V_ss[ind_s_ss, list_t_ss] = np.divide(Q_ss[ind_s_ss, list_t_ss], A_ss[ind_s_ss, list_t_ss])
            H_ss[ind_s_ss, list_t_ss] = Z_ss[ind_s_ss, list_t_ss] + c1s2G * np.power(V_ss[ind_s_ss, list_t_ss], 2)
            Iter_ss[ind_s_ss, list_t_ss] = 0
        else:
            dX = abs(X_s[ind_s_ss + 1] - X_s[ind_s_ss])
            for ind_t_ss in list_t_ss:
                Delta_H_dn = dX * V_ss[ind_s_ss + 1, ind_t_ss] ** 2 / (Ks ** 2 * R_ss[ind_s_ss + 1, ind_t_ss] ** c43)
                Delta_H_to = Delta_H_dn
                H_ss[ind_s_ss, ind_t_ss] = H_ss[ind_s_ss + 1, ind_t_ss] + Delta_H_to
                iter = 1
                Delta_H_min = -1
                Delta_H_max = -1
                [Hr, Ar, Pr, Fr2] = fnc_HAPF(Q_ss[ind_s_ss, ind_t_ss], Xr, Yr)
                flag_cal = 0
                while iter > 0:
                    i1 = np.where(H_ss[ind_s_ss, ind_t_ss] <= Hr)[0]
                    if i1.size == 0:
                        Z_ss[ind_s_ss, ind_t_ss] = Yr[-1]
                        [A_ss[ind_s_ss, ind_t_ss], P_ss[ind_s_ss, ind_t_ss], R_ss[ind_s_ss, ind_t_ss], w_fic] = fnc_APR([Yr[-1]], Xr, Yr)
                        V_ss[ind_s_ss, ind_t_ss] = Q_ss[ind_s_ss, ind_t_ss] / A_ss[ind_s_ss, ind_t_ss]
                        R_ss[ind_s_ss, ind_t_ss] = A_ss[ind_s_ss, ind_t_ss] / P_ss[ind_s_ss, ind_t_ss]
                        flag_cal = 1
                    else:
                        if np.isnan(Hr[i1[0]]):
                            i1 = i1[1]
                        else:
                            i1 = i1[0]
                        if i1 == 1 or Fr2[i1 - 1] > 1:
                            i2 = np.where(Fr2 <= 1)[0]
                            if len(i2) == 0:
                                return (Z_ss, A_ss, P_ss, R_ss, np.nan)
                            else:
                                if np.isnan(Fr2[i2[0]]):
                                    i2 = i2[1]
                                else:
                                    i2 = i2[0]
                                Z1 = Yr[i2 - 1]
                                A1 = Ar[i2 - 1]
                                P1 = Pr[i2 - 1]
                                L1 = Xr[i2 - 1]
                                Z2 = Yr[i2]
                                A2 = Ar[i2]
                                P2 = Pr[i2]
                                L2 = Xr[i2]
                                aa1 = -Q_ss[ind_s_ss, ind_t_ss] ** 2 / cG
                                aa2 = (L2 - L1) / (A2 - A1)
                                p = aa1 * aa2
                                q = aa1 * (L1 - A1 * aa2)
                                if q ** 2 / 4 + p ** 3 / 27 < 0.0:
                                    x0 = np.real(cmath.sqrt(q ** 2 / 4 + p ** 3 / 27))
                                else:
                                    x0 = np.sqrt(q ** 2 / 4 + p ** 3 / 27)
                                x1 = (-q / 2 + x0) ** (1 / 3) + (-q / 2 - x0) ** (1 / 3)
                                residu = x1 ** 3 + p * x1 + q
                                alpha = (x1 - A1) / (A2 - A1)
                                Zc_ss[ind_s_ss, ind_t_ss] = Z1 + alpha * (Z2 - Z1)
                                Ac_ss[ind_s_ss, ind_t_ss] = A1 + alpha * (A2 - A1)
                                Lc_ss[ind_s_ss, ind_t_ss] = L1 + alpha * (L2 - L1)
                                Pc_ss[ind_s_ss, ind_t_ss] = P1 + alpha * (P2 - P1)
                                Hc_ss[ind_s_ss, ind_t_ss] = Zc_ss[ind_s_ss, ind_t_ss] + c1s2G * (Q_ss[ind_s_ss, ind_t_ss] / Ac_ss[ind_s_ss, ind_t_ss]) ** 2
                                H1 = Hc_ss[ind_s_ss, ind_t_ss]
                                Z1 = Zc_ss[ind_s_ss, ind_t_ss]
                                A1 = Ac_ss[ind_s_ss, ind_t_ss]
                                P1 = Pc_ss[ind_s_ss, ind_t_ss]
                                residu = 1 / cG * Q_ss[ind_s_ss, ind_t_ss] ** 2 * Lc_ss[ind_s_ss, ind_t_ss] / Ac_ss[ind_s_ss, ind_t_ss] ** 3
                                residu = 1 - np.sqrt(residu)
                        else:
                            H1 = Hr[i1 - 1]
                            Z1 = Yr[i1 - 1]
                            A1 = Ar[i1 - 1]
                            P1 = Pr[i1 - 1]
                        if H_ss[ind_s_ss, ind_t_ss] > H1:
                            H2 = Hr[i1]
                            Z2 = Yr[i1]
                            A2 = Ar[i1]
                            P2 = Pr[i1]
                            aa1 = (H_ss[ind_s_ss, ind_t_ss] - Z1) / (Z2 - Z1)
                            aa2 = (Q_ss[ind_s_ss, ind_t_ss] / A1) ** 2 / (c2G * (Z2 - Z1))
                            a3 = A2 / A1 - 1
                            b1 = aa1 * a3 + 1
                            b2 = aa2 * a3
                            p = -b1 ** 2 / 3
                            q = p * b1 * 2 / 9 + b2
                            if q ** 2 / 4 + p ** 3 / 27 < 0.0:
                                x0 = np.real(cmath.sqrt(q ** 2 / 4 + p ** 3 / 27))
                            else:
                                x0 = np.sqrt(q ** 2 / 4 + p ** 3 / 27)
                            x1 = (-q / 2 + x0) ** (1 / 3) + (-q / 2 - x0) ** (1 / 3)
                            residu = x1 ** 3 + p * x1 + q
                            beta = x1 + b1 / 3
                            alpha = (beta - 1) / a3
                        else:
                            alpha = 0
                            Z2 = 0
                            A2 = 0
                            P2 = 0
                            flag_cal = 2
                        Z_ss[ind_s_ss, ind_t_ss] = Z1 + alpha * (Z2 - Z1)
                        A_ss[ind_s_ss, ind_t_ss] = A1 + alpha * (A2 - A1)
                        P_ss[ind_s_ss, ind_t_ss] = P1 + alpha * (P2 - P1)
                        V_ss[ind_s_ss, ind_t_ss] = Q_ss[ind_s_ss, ind_t_ss] / A_ss[ind_s_ss, ind_t_ss]
                        R_ss[ind_s_ss, ind_t_ss] = A_ss[ind_s_ss, ind_t_ss] / P_ss[ind_s_ss, ind_t_ss]
                    Delta_H_up = dX * V_ss[ind_s_ss, ind_t_ss] ** 2 / (Ks ** 2 * R_ss[ind_s_ss, ind_t_ss] ** c43)
                    Delta_H_to_new = 0.5 * (Delta_H_up + Delta_H_dn)
                    Iter_ss[ind_s_ss, ind_t_ss] = iter
                    DH = abs(Delta_H_to_new - Delta_H_to)
                    if DH < DH_Esp or iter >= DH_Iter or flag_cal > 0:
                        if iter > 1 and DH < DH_Esp:
                            pass
                        if iter >= DH_Iter:
                            flag_cal = 3
                        iter = 0
                    else:
                        iter = iter + 1
                    Delta_H_dir_new = np.sign(Delta_H_to_new - Delta_H_to)
                    if Delta_H_dir_new > 0:
                        if Delta_H_min > 0:
                            Delta_H_min = max(Delta_H_min, Delta_H_to)
                        else:
                            Delta_H_min = Delta_H_to
                    if Delta_H_dir_new < 0:
                        if Delta_H_max > 0:
                            Delta_H_max = min(Delta_H_max, Delta_H_to)
                        else:
                            Delta_H_max = Delta_H_to
                    if Delta_H_min > 0 and Delta_H_max > 0:
                        Delta_H_to_new = (Delta_H_min + Delta_H_max) / 2
                    Delta_H_to = Delta_H_to_new
                    H_ss[ind_s_ss, ind_t_ss] = H_ss[ind_s_ss, ind_t_ss] + Delta_H_to
        if not np.isnan(Z_ss[ind_s_ss, list_t_ss]) and (not np.isnan(H_s[ind_s_ss] and H_s[ind_s_ss] > params.valid_min_z and (Z_ss[ind_s_ss, list_t_ss] > params.valid_min_z))):
            cost += (Z_ss[ind_s_ss, list_t_ss] - H_s[ind_s_ss]) ** 2 / params.sigmaZ ** 2
    return (Z_ss, A_ss, P_ss, R_ss, cost)

def fnc_HAPF(Q, Ws, Zs):
    cG = 9.81
    c2G = 2 * cG
    c1s2G = 1 / c2G
    c2sG = 2 / cG
    ns = len(Zs)
    nq = np.array(Q).size
    Ws = np.array(Ws) / 2
    H = np.ones(ns) * np.nan
    A = np.ones(ns) * np.nan
    P = np.ones(ns) * np.nan
    Fr2 = np.ones(ns) * np.nan
    H[0] = np.nan
    A[0] = 0
    P[0] = Ws[1]
    Fr2[0] = np.nan
    V2 = []
    for i in range(1, ns):
        A[i] = A[i - 1] + (Ws[i - 1] + Ws[i]) * (Zs[i] - Zs[i - 1])
        P[i] = P[i - 1] + np.sqrt((Ws[i] - Ws[i - 1]) ** 2 + (Zs[i] - Zs[i - 1]) ** 2)
    P = P * 2
    Q_transpose = np.array(Q).reshape(nq, 1)
    V2 = np.power(np.divide(Q_transpose, A[1:ns]), 2)
    H[1:ns] = Zs[1:ns] + c1s2G * V2
    Fr2[1:ns] = np.divide(np.multiply(c2sG * V2, Ws[1:ns]), A[1:ns])
    return (H, A, P, Fr2)

def likelihood(TDIM, nsobs, lc0, QSample, QPrior, cost_all, Weight):
    LH_pdf = []
    QPost = np.zeros(TDIM)
    Nbeta2 = 0.0
    abet1 = 2.0 ** (float(lc0) - 1) / float(nsobs)
    beta1 = 1.0 / abet1
    for iis in range(len(cost_all)):
        if not pd.isna(cost_all[iis]):
            LH_pdf += [math.exp(-0.25 / beta1 * float(nsobs) * (cost_all[iis] / np.nanmin(cost_all) - 1.0) ** 2)]
        if pd.isna(cost_all[iis]):
            LH_pdf += [0.0]
        Nbeta2 += Weight[iis] * LH_pdf[iis]
        for ij in range(TDIM):
            QPost[ij] = QPost[ij] + QSample[iis, ij] * LH_pdf[iis] * Weight[iis]
    if Nbeta2 == 0.0 or pd.isna(Nbeta2):
        pass
    QPost /= Nbeta2
    Svar = []
    for ij in range(TDIM):
        temp = 0.0
        for iis in range(len(cost_all)):
            temp += (QSample[iis, ij] - QPost[ij]) ** 2 * LH_pdf[iis] * Weight[iis]
        Svar += [temp]
    Svar /= Nbeta2
    distp = 0.0
    for ij in range(TDIM):
        distp += (QPost[ij] - QPrior[ij]) ** 2 / Svar[ij]
    distp = distp / TDIM
    return (distp, LH_pdf, QPost)

def compute_bb(depth_mean, W0_mean, A0_mean, P0_mean):
    bb = 0.0
    for ii in range(0, 50):
        bb0 = bb
        ss1 = A0_mean + W0_mean * bb
        ss2 = P0_mean + 2.0 * bb
        bb = 1.5 * ss1 / ss2 - depth_mean
        print(ii, bb, (depth_mean + bb0) / (ss1 / ss2), abs(bb0 / bb - 1.0), A0_mean / P0_mean)
        if abs(bb0 / bb - 1.0) <= 0.01:
            break
    return bb

def create_confluence_dict(reach_time, reach_z, reach_w, reach_s):
    ObsData = {}
    ObsData['nR'] = 1
    ObsData['xkm'] = np.nan
    ObsData['L'] = np.nan
    ObsData['nt'] = len(reach_time)
    ObsData['t'] = reach_time
    ObsData['h'] = np.empty((ObsData['nR'], ObsData['nt']))
    ObsData['h0'] = np.empty((ObsData['nR'], 1))
    ObsData['S'] = np.empty((ObsData['nR'], ObsData['nt']))
    ObsData['w'] = np.empty((ObsData['nR'], ObsData['nt']))
    for i in range(0, ObsData['nR']):
        ObsData['h'][i, :] = reach_z
        ObsData['w'][i, :] = reach_w
        ObsData['S'][i, :] = reach_s
    ObsData['sigh'] = 0.1
    ObsData['sigw'] = 10.0
    ObsData['sigS'] = 1.7e-05
    ObsData['iDelete'] = np.where(np.isnan(ObsData['w'][0, :]) | np.isnan(ObsData['h'][0, :]))
    return ObsData

def bathymetry_computation(node_w, node_z, param_dict, params, input_data=[], filtered_data=[], slope=[]):
    node_xr = []
    node_yr = []
    for i in range(len(node_w)):
        if not params.pankaj_test:
            if param_dict['cs_method'] == 'POM':
                results = f_approx_sections_v6(node_w[i], node_z[i], params.approx_section_params[0], params.approx_section_params[1], params.approx_section_params[2])
            elif param_dict['cs_method'] == 'Igor':
                results = M(node_w[i], node_z[i], max_iter=params.LSMX, cor_z=None, inter_behavior=True, inter_behavior_min_thr=params.def_float_atol, inter_behavior_max_thr=params.DX_max_in, min_change_v_thr=0.0001, first_sweep='forward', cs_float_atol=params.def_float_atol, number_of_nodes=len(node_z), plot=False)
            elif param_dict['cs_method'] == 'Mike' and Confluence_HWS_method:
                if param_dict['use_reach_slope']:
                    ObsData = create_confluence_dict(filtered_data['reach_t'], node_z[i], node_w[i], filtered_data['reach_s'])
                else:
                    ObsData = create_confluence_dict(filtered_data['reach_t'], node_z[i], node_w[i], slope)
                D = DomainHWS(ObsData)
                hws_obj = CalculateHWS(D, ObsData)
                if hasattr(hws_obj, 'area_fit'):
                    results = [hws_obj.area_fit['w_break'], hws_obj.area_fit['h_break']]
                    results2 = [hws_obj.wobs, hws_obj.hobs]
                if i == 10:
                    print(i)
                    print(np.nanmin(results2[0]))
            node_xr += [results[0]]
            node_yr += [results[1]]
            if param_dict['cs_plot_debug']:
                results_pom = f_approx_sections_v6(node_w[i], node_z[i], params.approx_section_params[0], params.approx_section_params[1], params.approx_section_params[2])
                results_igor = M(node_w[i], node_z[i], max_iter=params.LSMX, cor_z=None, inter_behavior=True, inter_behavior_min_thr=params.def_float_atol, inter_behavior_max_thr=params.DX_max_in, min_change_v_thr=0.0001, first_sweep='forward', cs_float_atol=params.def_float_atol, number_of_nodes=len(node_z), plot=False)
                if Confluence_HWS_method:
                    ObsData = create_confluence_dict(filtered_data['reach_t'], node_z[i], node_w[i], filtered_data['reach_s'])
                    D = DomainHWS(ObsData)
                    hws_obj = CalculateHWS(D, ObsData)
                    results_mike = [hws_obj.cs_w[0], hws_obj.cs_z[0]]
                    plt.plot(results_mike[0], results_mike[1])
                plt.plot(results_pom[0], results_pom[1])
                plt.plot(results_igor[0], results_igor[1])
                plt.plot(node_w[i], node_z[i], marker='.', linestyle='None')
                plt.show()
                plt.clf()
    if param_dict['gnuplot_saving']:
        node_x = input_data['node_x']
        times2 = np.around(input_data['reach_t'] / 3600 / 24)
        times2 = times2 - np.nanmin(times2)
        nodes2 = (node_x - node_x[0]) / 1000
        reach_id = str(input_data['reach_id'])
        reach_id = verify_name_length(reach_id)
        output_path = param_dict['output_dir'].joinpath('gnuplot_data', str(reach_id))
        if not Path(output_path).is_dir():
            Path(output_path).mkdir(parents=True, exist_ok=True)
        output_path = param_dict['output_dir'].joinpath('gnuplot_data', str(reach_id), 'cs')
    return (node_xr, node_yr)

def call_func_APR(node_w, node_z, node_xr, node_yr, params, param_dict):
    node_a = node_w.copy()
    node_p = node_w.copy()
    node_r = node_w.copy()
    node_w_simp = node_w.copy()
    depth_mean = []
    A0_mean = []
    P0_mean = []
    W0_mean = []
    for i in range(len(node_w)):
        node_a[i], node_p[i], node_r[i], node_w_simp[i] = fnc_APR(node_z[i], node_xr[i], node_yr[i])
        depth_section = 0.0
        if param_dict['bb_computation']:
            depth_section = np.nanmax(node_yr[i]) - np.nanmin(node_yr[i])
            A0 = 0
            P0 = node_xr[i][0]
            for j in range(0, len(node_xr[i]) - 1):
                A0 = A0 + (node_xr[i][j] + node_xr[i][j + 1]) / 2 * (node_yr[i][j + 1] - node_yr[i][j])
                P0 = P0 + 2 * np.sqrt((node_xr[i][j + 1] / 2 - node_xr[i][j] / 2) ** 2 + (node_yr[i][j + 1] - node_yr[i][j]) ** 2)
                if node_yr[i][j] - np.nanmin(node_yr[i]) > (np.nanmax(node_yr[i]) - np.nanmin(node_yr[i])) / 2:
                    depth_section = (np.nanmax(node_yr[i]) - np.nanmin(node_yr[i])) / 2
                    break
            if False:
                results2 = f_approx_sections_v6(node_w[i], node_z[i], params.approx_section_params[0], params.approx_section_params[1], params.approx_section_params[2])
                plt.plot(node_xr[i], node_yr[i])
                plt.plot(results2[0], results2[1])
                plt.plot(node_w[i], node_z[i], marker='.', linestyle='None')
                print('A0, P0, R, depth, depth/R:', A0, P0, A0 / P0, depth_section, depth_section / (A0 / P0))
                plt.show()
                plt.clf()
            A0_mean.append(A0)
            P0_mean.append(P0)
            W0_mean.append(node_xr[i][0])
        index_temp = np.where(node_w_simp[i] == 0.0)
        node_w_simp[i][index_temp] = np.nanmin(node_xr[i][np.where(node_xr[i] > 0.0)])
        depth_mean.append(depth_section)
    if not param_dict['bb_computation']:
        bb = 9999.0
    else:
        depth_mean = np.nanmean(depth_mean)
        A0_mean = np.nanmean(A0_mean)
        P0_mean = np.nanmean(P0_mean)
        W0_mean = np.nanmean(W0_mean)
        print('depth_mean, A0_mean, P0_mean, W0_mean:', depth_mean, A0_mean, P0_mean, W0_mean)
        print('mean Radius:', A0_mean / P0_mean, depth_mean / (A0_mean / P0_mean))
        bb = compute_bb(depth_mean, W0_mean, A0_mean, P0_mean)
        print('Final computed bb:', bb)
        if bb < 1:
            bb = 1
        print('Final bb:', bb)
    return (node_a, node_p, node_r, node_w_simp, bb)

def compute_mean_discharge_from_SoS_quantiles(quantiles):
    nq = len(quantiles)
    bin = 1.0 / nq
    quant = np.zeros(nq + 1)
    dquant = np.zeros(nq + 1)
    for i in range(0, nq):
        quant[i] = quantiles[i]
    dquant[0] = (3.0 * quant[0] - quant[2]) / 2.0
    for i in range(0, nq - 1):
        dquant[i + 1] = (quant[i] + quant[i + 1]) / 2.0
    dquant[nq] = (3.0 * quant[nq - 1] - quant[nq - 2]) / 2.0
    for i in range(0, nq + 1):
        quant[i] = dquant[nq - i]
    for i in range(0, nq):
        dquant[i] = bin / (quant[i + 1] - quant[i]) * (quant[i + 1] - quant[i])
    ss = 0.0
    ss1 = 0.0
    for i in range(0, nq):
        ss = ss + (quant[i + 1] + quant[i]) / 2.0 * dquant[i]
        ss1 = ss1 + dquant[i]
    quant_mean = ss / ss1
    ss = 0.0
    ss1 = 0.0
    quant_var = 0.0
    for i in range(0, nq):
        ss = ss + ((quant[i + 1] + quant[i]) / 2.0 - quant_mean) ** 2 * dquant[i]
        ss1 = ss1 + dquant[i]
    quant_var = np.sqrt(ss / ss1)
    return (quant_mean, quant_var)
