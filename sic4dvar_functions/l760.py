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
Z='node_q'
Y='time'
T='width'
P='wse'
h='increase'
g=TypeError
X='raise'
W='inc'
V=1.
G='node'
U=len
S='dec'
R=range
O=abs
K=RuntimeError
E=''
C=True
B=False
import copy as Q,logging as p,warnings as q
from typing import Literal,Tuple
import numpy as A
from sic4dvar_classes.sic4dvar_0_defaults import SIC4DVarLowCostDefaults as D
from sic4dvar_functions.helpers.helpers_arrays import masked_array_to_nan_array as N,arrays_check_decrease as r,arrays_check_increase as s,get_index_valid_data as t
from sic4dvar_functions.helpers.helpers_generic import pairwise as c
from sic4dvar_functions.helpers.helpers_plot import helper_plot_lines as u
from pathlib import Path
from sic4dvar_functions.io.reader_swot_obs import get_vars_from_swot_nc as a
def d(value_0,value_1,cor):return V-A.exp(-(value_1-value_0)/cor)
def e(value_init,value_smoothed,min_change_v_thr=D().def_float_atol,value_ref=A.nan):
	D=value_ref;C=value_smoothed;B=value_init;E=A.abs(C-B)
	if E<min_change_v_thr:return B,.0
	if A.isfinite(D):
		if A.abs(C-D)<=A.abs(B-D):return C,E
		else:return B,.0
	return C,E
def f(value_0,value_1,b=E,b_min_thr=D().def_float_atol,b_max_thr=A.inf):
	D=value_1;A=value_0
	if not b:return B
	if O(D-A)>b_max_thr:return B
	if O(D-A)<b_min_thr:return B
	if b==S:
		if D<A:return B
		return C
	if D>A:return B
	return C
def v(sub_value0_array,sub_base0_array,cor,min_change_v_thr=D().def_float_atol,inter_behaviour=E,inter_behaviour_min_thr=D().def_float_atol,inter_behaviour_max_thr=A.inf,inter_only=B,check_nan=C,debug_mode=B):
	k=debug_mode;j=inter_behaviour_max_thr;i=inter_behaviour_min_thr;h=inter_behaviour;X=min_change_v_thr;P=inter_only
	if S in h.lower():Q=S
	elif W in h.lower():Q=W
	else:
		Q=E
		if P:raise g('')
	Y=''if P else'';B=N(sub_value0_array);T=N(sub_base0_array);Z=[B,T]
	if A.any([A.ndim!=1 for A in Z]):raise K(f"")
	if A.any([A.shape!=B.shape for A in Z]):raise K(f"")
	if check_nan:
		if A.any([A.any(A.isnan(B))for B in Z[:-1]]):raise K(f"")
	J=int(O(A.log10(X)));I=0
	for(G,H)in c(R(U(B))):
		L=f"";D,F=B[G],B[H]
		if not P:
			M=d(T[G],T[H],cor);a=D*(V-M)+F*M;a,C=e(value_init=F,value_smoothed=a,min_change_v_thr=X);B[H]=a
			if C>I:I=C
			L+=f"";D,F=B[G],B[H]
		if f(value_0=D,value_1=F,b=Q,b_min_thr=i,b_max_thr=j):
			B[G],B[H]=F,D;C=O(F-D)
			if C>I:I=C
			L+=f""
		elif P:L+=f""
		if k:0
	for(H,G)in c(reversed(R(U(B)))):
		L=f"";D,F=B[G],B[H]
		if not P:
			M=d(T[G],T[H],cor);b=F*(V-M)+D*M;b,C=e(value_init=D,value_smoothed=b,min_change_v_thr=X);B[G]=b
			if C>I:I=C
			if C>I:I=C
			L+=f"";D,F=B[G],B[H]
		if f(value_0=D,value_1=F,b=Q,b_min_thr=i,b_max_thr=j):
			B[G],B[H]=F,D;C=O(F-D)
			if C>I:I=C
			L+=f""
		elif P:L+=f""
		if k:0
	return B,I
def b(value0_array,base0_array,max_iter,cor,always_run_first_iter=C,behaviour=E,inter_behaviour=B,inter_behaviour_min_thr=D().def_float_atol,inter_behaviour_max_thr=A.inf,check_behaviour=X,min_change_v_thr=D().def_float_atol,plot=C,plot_title=E,clean_run=B,debug_mode=B):
	o='print';n='warn';m='force';f=always_run_first_iter;e='none';d=None;Z=debug_mode;Y=min_change_v_thr;T=plot;M=inter_behaviour;L=behaviour;I=clean_run;H=check_behaviour
	if Z:I=B
	if I:Z=B
	F=N(value0_array);a=N(base0_array);i=[F,a]
	if A.any([A.ndim!=1 for A in i]):raise K(f"")
	if A.any([A.shape!=F.shape for A in i]):raise K(f"")
	if S in L.lower():J=r;b='decreasing';M='decrease'if M else E
	elif W in L.lower():J=s;b='increasing';M=h if M else E
	else:L=E;f=C;J=d;b=E;H=e;M=E
	if L:
		if not any([A==H.lower()for A in[m,X,n,o,e,E]]):raise g('')
	w=int(O(A.log10(Y)));G=t(F,a)[0]
	if not I:0
	V,P=[],[]
	if not f:
		if J(F,remove_nan=C):
			if T:0
			return F
	if U(G)<2:
		if not I:0
		return F
	D=F[G];x=A.mean(D)
	if not I:0
	if T:V.append(Q.deepcopy(F[G]));P.append('Initial')
	for j in R(max_iter):
		D,k=v(sub_value0_array=F[G],sub_base0_array=a[G],cor=cor,min_change_v_thr=Y,inter_behaviour=M,inter_only=B,inter_behaviour_min_thr=inter_behaviour_min_thr,inter_behaviour_max_thr=inter_behaviour_max_thr,check_nan=B,debug_mode=Z)
		if not I:p.debug(f"")
		if k<=Y:c=C
		elif J is not d:c=C if J(D,remove_nan=B)else B
		else:c=B
		if T:V.append(Q.deepcopy(D));P.append(f"")
		F[G]=D
		if c:break
	y=A.mean(D);D=D+(x-y)
	if J is not d:
		if L:
			l=f""
			if not H:0
			elif H.lower()==e:0
			elif not J(D,remove_nan=B):
				if H.lower()==n:q.warn(l,RuntimeWarning)
				elif H.lower()==o:0
				elif m in H.lower():
					D=A.sort(D)
					if S in L.lower():D=D[::-1]
				else:
					with A.printoptions(precision=4,suppress=C):0
					raise K(l)
	if not I:0
	if T:V.append(Q.deepcopy(D));P.append('');u(xs=[G]*U(P),ys=V,show=C,line_labels=P,title=plot_title,x_axis_title='',y_axis_title='',fig_width=15,fig_height=5,add_legend=C)
	F[G]=D;return F
def L(dim,value0_array,base0_array,max_iter,cor,always_run_first_iter=C,behaviour=E,inter_behaviour=B,inter_behaviour_min_thr=D().def_float_atol,inter_behaviour_max_thr=A.inf,check_behaviour=X,min_change_v_thr=D().def_float_atol,plot=B,plot_title=E,clean_run=B,debug_mode=B):
	J=value0_array;I=debug_mode;G=clean_run;F=dim
	if I:G=B
	if G:I=B
	D=N(J);H=N(base0_array);L=[D,H]
	if A.any([A.ndim>2 for A in L]):raise K(f"")
	if A.any([A.shape!=D.shape for A in L[2:]]):raise K(f"")
	if F>1:raise NotImplementedError('')
	D=Q.deepcopy(J);P=C if D.ndim==H.ndim else B
	if not G:0
	for E in R(D.shape[F]):
		if not G:0
		S=D[E,:]if F==0 else D[:,E]
		if P:M=H[E,:]if F==0 else H[:,E]
		else:M=H
		O=b(value0_array=S,base0_array=M,max_iter=max_iter,cor=cor,always_run_first_iter=always_run_first_iter,behaviour=behaviour,inter_behaviour=inter_behaviour,inter_behaviour_min_thr=inter_behaviour_min_thr,inter_behaviour_max_thr=inter_behaviour_max_thr,check_behaviour=check_behaviour,min_change_v_thr=min_change_v_thr,plot=plot,plot_title=plot_title+f" {E}",clean_run=G,debug_mode=I)
		if F==0:D[E,:]=O
		else:D[:,E]=O
	return D
