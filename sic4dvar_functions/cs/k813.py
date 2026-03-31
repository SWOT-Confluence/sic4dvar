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

Created on December 18th 2023
by @Isadora Silva

Last modified on February 20th 2025 at 13:45
by @Isadora Silva

@authors: Igor Gejadze and Isadora Silva
"""
L = 'node'
K = 'width'
H = 'wse'
t = 'increase'
P = RuntimeError
g = list
f = len
e = range
O = ''
J = True
F = None
D = False
import copy as I
from typing import Tuple
import matplotlib as A3, numpy as A
from sic4dvar_classes.sic4dvar_0_defaults import sic_def as B
from sic4dvar_functions.helpers.helpers_arrays import masked_array_to_nan_array as Z, arrays_bounds as S, arrays_check_increase as A4
from sic4dvar_functions.helpers.helpers_generic import pairwise as A5
from sic4dvar_functions.helpers.helpers_plot import helper_plot_lines as A6
from sic4dvar_functions.j211 import D as r
from sic4dvar_functions.Q428 import u as s

def M(cs_i_w0_array, cs_i_z0_array, max_iter, cor_z=F, inter_behavior=J, inter_behavior_min_thr=B.def_lsm_w_min_dw, inter_behavior_max_thr=A.inf, min_change_v_thr=B.def_lsm_w_min_dw, cs_i_w_low_bound0_array=F, cs_i_w_up_bound0_array=F, cs_i_w_ref0_array=F, first_sweep=B.def_lsm_w_first_sweep, remove_bias_in_loop=B.def_lsm_w_rem_bias_in_loop, cs_float_atol=B.def_cs_float_atol, number_of_nodes=0, plot=J, plot_title=O, clean_run=D, debug_mode=D):
    R = number_of_nodes
    Q = cs_i_w_ref0_array
    N = debug_mode
    M = clean_run
    K = cor_z
    if N:
        M = D
    if M:
        N = D
    C = Z(cs_i_z0_array)
    B = Z(cs_i_w0_array)
    E, G = S(ref_array=B, value0_low_bound_array=cs_i_w_low_bound0_array, value0_up_bound_array=cs_i_w_up_bound0_array)
    if Q is F:
        L = A.full(B.shape, fill_value=A.nan, dtype=A.float32)
    else:
        L = Z(Q)
    I = A.isfinite(C) & A.isfinite(B)
    C = C[I]
    B = B[I]
    E = E[I]
    G = G[I]
    L = L[I]
    if C.size == 0:
        raise P
    if C.size == 1:
        raise P
    if C.size == 2:
        O = B.argsort()
        H = C.argsort()
        B = B[O]
        C = C[H]
        E = A.minimum(E[O], B)
        G = A.maximum(G[O], B)
        return (B, C, E, G)
    H = C.argsort()
    B = B[H]
    E = E[H]
    G = G[H]
    C = C[H]
    if K is F or A.isnan(K):
        R = C.size
        K = (C[-1] - C[0]) / R
    B, E, G, T = s(value0_array=B, base0_array=C, max_iter=max_iter, cor=K, min_change_v_thr=min_change_v_thr, behavior=t, inter_behavior=inter_behavior, inter_behavior_min_thr=inter_behavior_min_thr, inter_behavior_max_thr=inter_behavior_max_thr, check_behavior='force', value_low_bound0_array=E, value_up_bound0_array=G, value_ref0_array=L, first_sweep=first_sweep, remove_bias_in_loop=remove_bias_in_loop, always_smooth=J, inter_only=D, float_atol=cs_float_atol, plot=plot, plot_title=plot_title, clean_run=M, debug_mode=N)
    B = A.sort(B)
    E = A.fmin(E, B)
    G = A.fmax(G, B)
    return (B, C, E, G)

def N(w0_array, z0_array, cor_z=F, extrapolate_min=D, extrapolate_max=D, first_sweep=B.def_lsm_w_first_sweep, remove_bias_in_loop=B.def_lsm_w_rem_bias_in_loop, cs_float_atol=B.def_cs_float_atol, plot=J, plot_title=O, plot_colormap='YlOrBr', clean_run=D):
    A2 = 'linear'
    A1 = 'yellowgreen'
    x = extrapolate_max
    w = extrapolate_min
    v = z0_array
    u = w0_array
    c = clean_run
    b = plot
    a = cor_z
    Y = 1.0
    H = 0.0
    P, Q = ([], [])
    R, L, S, T = ([], [], [], [])
    U, V, M = ([], [], [])
    y = A3.colormaps.get_cmap(plot_colormap)
    AI = ''
    if not c:
        0
    if A.all(A.isnan(v)) or A.all(A.isnan(u)):
        raise TypeError
    E, G = [Z(A).flatten() for A in [v, u]]
    h, i = A.nanquantile(E, [H, Y])
    z = A.isfinite(E) & A.isfinite(G)
    E = E[z]
    G = G[z]
    W = A.argsort(E)
    G = G[W]
    E = E[W]
    X = int(abs(A.log10(cs_float_atol))) + 1
    G = A.round(G, X)
    E = A.round(E, X)
    if b:
        P.append(I.deepcopy(G / 2))
        Q.append(I.deepcopy(E))
        R.append(F)
        L.append(F)
        T.append(H)
        S.append('original')
        U.append('x')
        V.append(4.0)
        M.append('')
    j, k = (E[0], E[-1])
    A7, A8 = (G[0], G[-1])
    l = A.arange(j, k, min(max(0.1, (k - j) / 50), Y))
    C, B = ([j], [A7])
    for A9, AA in A5(e(f(l))):
        AB, AC = (l[A9], l[AA])
        m = (E >= AB) & (E < AC)
        if not A.any(m):
            continue
        AD = E[m]
        AE = G[m]
        AF = A.median(AD)
        AG = A.median(AE)
        C.append(AF)
        B.append(AG)
    C.append(k)
    B.append(A8)
    C = A.array(C)
    B = A.array(B)
    W = A.argsort(C)
    B = I.deepcopy(B[W])
    C = I.deepcopy(C[W])
    B = A.round(B, X)
    C = A.round(C, X)
    if a is F or A.isnan(a):
        a = A.nanmean(A.diff(C))
    if b:
        P.append(I.deepcopy(B / 2))
        Q.append(I.deepcopy(C))
        R.append(':')
        L.append(A1)
        T.append(2.0)
        S.append('')
        U.append('o')
        V.append(3.0)
        M.append(A1)
    AH = A.full_like(B, fill_value=A.nan)
    K = 0
    n, o = (H, 1e-07)
    while not A4(B, remove_nan=D):
        if K == 1:
            o = n / 10
        if not c:
            0
        B, _, _, n = s(value0_array=B, base0_array=C, max_iter=1, cor=a, behavior=t, inter_behavior=J, min_change_v_thr=o, inter_behavior_min_thr=o * 10.0, inter_behavior_max_thr=A.inf, check_behavior=O, value_ref0_array=AH, first_sweep=first_sweep, remove_bias_in_loop=remove_bias_in_loop, always_smooth=J, inter_only=D, plot=D, plot_title=f'Test {K}', clean_run=c, debug_mode=D)
        if not c:
            0
        B = A.round(B, X)
        B = A.sort(B)
        if b:
            P.append(I.deepcopy(B / 2))
            Q.append(I.deepcopy(C))
            R.append('--')
            L.append(F)
            T.append(2.0)
            S.append(f'points  {K}')
            U.append('x')
            V.append(4.0)
            M.append(F)
        K += 1
        if A.isclose(n, H, rtol=H, atol=1e-07):
            B = A.sort(B)
            break
        if K > B.size + 1:
            B = A.sort(B)
            break
    B[B < H] = H
    if h > C[0]:
        w = D
    if w:
        p = r(values_in_array=A.array([A.nan, B[0], B[1]]), base_in_array=A.array([h, C[0], C[1]]), limits=A2, check_nan=D)
        p[p < H] = H
        C = A.concatenate([A.array([h]), C])
        B = A.concatenate([A.array([p[0]]), B])
    if i < C[-1]:
        x = D
    if x:
        q = r(values_in_array=A.array([B[-2], B[-1], A.nan]), base_in_array=A.array([C[-2], C[-1], i]), limits=A2, check_nan=D)
        q[q < H] = H
        C = A.concatenate([C, A.array([i])])
        B = A.concatenate([B, A.array([q[-1]])])
    if b:
        if K > 1:
            N = g(A.arange(0.1, Y, 0.9 / (K - 1)))
        else:
            N = g(A.arange(0.1, Y, 0.45))
        if f(N) < K:
            N.append(Y)
        A0 = g(e(2, f(N) + 2, 1))
        for d in e(K):
            L[A0[d]] = y(N[d])
            M[A0[d]] = y(N[d])
        P.append(I.deepcopy(B / 2))
        Q.append(I.deepcopy(C))
        R.append('-')
        L.append('black')
        T.append(2.5)
        S.append('final')
        U.append(O)
        V.append(4.0)
        M.append(O)
        A6(xs=P, ys=Q, show=J, x_lim=(0.9 * A.nanmin(B / 2), 1.1 * A.nanmax(B / 2)), y_lim=(0.9 * A.nanmin(C), 1.1 * A.nanmax(C)), line_styles=R, line_widths=T, line_colors=L, line_labels=S, marker_styles=U, marker_sizes=V, marker_fill_colors=M, title=plot_title, x_axis_title='', y_axis_title='(m)', fig_width=15, fig_height=5, add_legend=J)
    return (B, C)
if __name__ == '__main__':
    from pathlib import Path
    from sic4dvar_low_cost.sic4dvar_functions.io.reader_swot_obs import get_vars_from_swot_nc as Q
    R = Path('C:\\Users\\isadora.rezende\\PhD\\Datasets')
    C, T = Q(reach_ids=(56463000571,), swot_file_pattern=R / 'SWOT' / 'Nominal' / 'netcdf' / '{}_SWOT.nc', node_vars=(H, K, 'time'))
    E = C[L][H][0]
    G = C[L][K][0]
    M(cs_i_z0_array=E[0, :], cs_i_w0_array=G[0, :], max_iter=10, cor_z=F, clean_run=D, debug_mode=D, plot=J)
    N(w0_array=G, z0_array=E, plot=J, clean_run=D)
