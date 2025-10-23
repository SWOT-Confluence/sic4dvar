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

V='node_q'
U='time'
Q='width'
P='wse'
n='increase'
k=float
m=TypeError
e='raise'
d='inc'
Z=1.
G='node'
Y=isinstance
W='dec'
T=len
S=abs
O=range
N=.0
M=RuntimeError
F=''
D=True
B=False
import copy as a,logging as x,warnings as y
from typing import Literal,Tuple
import numpy as A,sic4dvar_params as f
from sic4dvar_classes.sic4dvar_0_defaults import SIC4DVarLowCostDefaults as C
from sic4dvar_functions.helpers.helpers_arrays import masked_array_to_nan_array as R,arrays_check_decrease as z,arrays_check_increase as A0,get_index_valid_data as A1
from sic4dvar_functions.helpers.helpers_generic import pairwise as g
from sic4dvar_functions.helpers.helpers_plot import helper_plot_lines as A2
from pathlib import Path
from sic4dvar_functions.io.reader_swot_obs import get_vars_from_swot_nc as X
def h(value_0,value_1,cor):return Z-A.exp(-(value_1-value_0)/cor)
def i(value_init,value_smoothed,min_change_v_thr=C().def_float_atol,value_ref=A.nan):
	D=value_ref;C=value_smoothed;B=value_init;E=A.abs(C-B)
	if E<min_change_v_thr:return B,N
	if A.isfinite(D):
		if A.abs(C-D)<=A.abs(B-D):return C,E
		else:return B,N
	return C,E
def j(value_0,value_1,b=F,b_min_thr=C().def_float_atol,b_max_thr=A.inf):
	C=value_1;A=value_0
	if not b:return B
	if S(C-A)>b_max_thr:return B
	if S(C-A)<b_min_thr:return B
	if b==W:
		if C<A:return B
		return D
	if C>A:return B
	return D
def A3(sub_value0_array,sub_base0_array,cor,min_change_v_thr=C().def_float_atol,inter_behaviour=F,inter_behaviour_min_thr=C().def_float_atol,inter_behaviour_max_thr=A.inf,inter_only=B,check_nan=D,debug_mode=B):
	p=debug_mode;o=inter_behaviour_max_thr;n=inter_behaviour_min_thr;l=inter_behaviour;a=min_change_v_thr;U=inter_only;G=cor
	if W in l.lower():V=W
	elif d in l.lower():V=d
	else:
		V=F
		if U:raise m('')
	b='interchange'if U else'relaxation';B=R(sub_value0_array);X=R(sub_base0_array);c=[B,X]
	if A.any([A.ndim!=1 for A in c]):raise M(f"")
	if A.any([A.shape!=B.shape for A in c]):raise M(f"")
	if check_nan:
		if A.any([A.any(A.isnan(B))for B in c[:-1]]):raise M(f"")
	K=int(S(A.log10(a)));J=0
	for(D,E)in g(O(T(B))):
		P=f"";H,I=B[D],B[E];L=N
		if Y(G,A.ndarray):
			if G.shape[0]>1:L=(G[D]+G[E])/2
			else:L=G[D]
		elif Y(G,k):L=G
		if not U:
			Q=h(X[D],X[E],L);e=H*(Z-Q)+I*Q;e,C=i(value_init=I,value_smoothed=e,min_change_v_thr=a);B[E]=e
			if C>J:J=C
			P+=f"";H,I=B[D],B[E]
		if j(value_0=H,value_1=I,b=V,b_min_thr=n,b_max_thr=o):
			B[D],B[E]=I,H;C=S(I-H)
			if C>J:J=C
			P+=f""
		elif U:P+=f""
		if p:0
	for(E,D)in g(reversed(O(T(B)))):
		P=f"";H,I=B[D],B[E];L=N
		if Y(G,A.ndarray):
			if G.shape[0]>1:L=(G[D]+G[E])/2
			else:L=G[E]
		elif Y(G,k):L=G
		if not U:
			Q=h(X[D],X[E],L);f=I*(Z-Q)+H*Q;f,C=i(value_init=H,value_smoothed=f,min_change_v_thr=a);B[D]=f
			if C>J:J=C
			if C>J:J=C
			P+=f"";H,I=B[D],B[E]
		if j(value_0=H,value_1=I,b=V,b_min_thr=n,b_max_thr=o):
			B[D],B[E]=I,H;C=S(I-H)
			if C>J:J=C
			P+=f""
		elif U:P+=f""
		if p:0
	return B,J
def b(value0_array,base0_array,max_iter,cor,always_run_first_iter=D,behaviour=F,inter_behaviour=B,inter_behaviour_min_thr=C().def_float_atol,inter_behaviour_max_thr=A.inf,check_behaviour=e,min_change_v_thr=C().def_float_atol,plot=D,plot_title=F,clean_run=B,debug_mode=B,time_integration=B):
	w='print';v='warn';u='force';p=time_integration;o=always_run_first_iter;l='none';k=None;g=debug_mode;f=min_change_v_thr;b=plot;Q=inter_behaviour;P=behaviour;K=clean_run;J=check_behaviour
	if g:K=B
	if K:g=B
	G=R(value0_array);U=R(base0_array);q=[G,U]
	if A.any([A.ndim!=1 for A in q]):raise M(f"")
	if A.any([A.shape!=G.shape for A in q]):raise M(f"")
	if W in P.lower():L=z;h='decreasing';Q='decrease'if Q else F
	elif d in P.lower():L=A0;h='increasing';Q=n if Q else F
	else:P=F;o=D;L=k;h=F;J=l;Q=F
	if P:
		if not any([A==J.lower()for A in[u,e,v,w,l,F]]):raise m('')
	A4=int(S(A.log10(f)));I=A1(G,U)[0]
	if not K:0
	c,V=[],[]
	if not o:
		if L(G,remove_nan=D):
			if b:0
			return G
	if T(I)<2:
		if not K:0
		return G
	C=G[I]
	if p:
		X=N;from copy import deepcopy as i;H=i(U[I]);Y=N
		for E in O(1,T(H)):X+=(C[E]+C[E-1])/2*(H[E]-H[E-1]);Y+=H[E]-H[E-1]
		X=X/Y
	else:X=A.mean(C)
	if not K:0
	if b:c.append(a.deepcopy(G[I]));V.append('Initial')
	for r in O(max_iter):
		C,s=A3(sub_value0_array=G[I],sub_base0_array=U[I],cor=cor,min_change_v_thr=f,inter_behaviour=Q,inter_only=B,inter_behaviour_min_thr=inter_behaviour_min_thr,inter_behaviour_max_thr=inter_behaviour_max_thr,check_nan=B,debug_mode=g)
		if not K:x.debug(f"")
		if s<=f:j=D
		elif L is not k:j=D if L(C,remove_nan=B)else B
		else:j=B
		if b:c.append(a.deepcopy(C));V.append(f"")
		G[I]=C
		if j:break
	if p:
		from copy import deepcopy as i;H=i(U[I]);Z=N;Y=N
		for E in O(1,T(H)):Z+=(C[E]+C[E-1])/2*(H[E]-H[E-1]);Y+=H[E]-H[E-1]
		Z=Z/Y
	else:Z=A.mean(C)
	C=C+(X-Z)
	if L is not k:
		if P:
			t=f""
			if not J:0
			elif J.lower()==l:0
			elif not L(C,remove_nan=B):
				if J.lower()==v:y.warn(t,RuntimeWarning)
				elif J.lower()==w:0
				elif u in J.lower():
					C=A.sort(C)
					if W in P.lower():C=C[::-1]
				else:
					with A.printoptions(precision=4,suppress=D):0
					raise M(t)
	if not K:0
	if b:c.append(a.deepcopy(C));V.append('Final');A2(xs=[I]*T(V),ys=c,show=D,line_labels=V,title=plot_title,x_axis_title='',y_axis_title='',fig_width=15,fig_height=5,add_legend=D)
	G[I]=C;return G
def K(dim,value0_array,base0_array,max_iter,cor,always_run_first_iter=D,behaviour=F,inter_behaviour=B,inter_behaviour_min_thr=C().def_float_atol,inter_behaviour_max_thr=A.inf,check_behaviour=e,min_change_v_thr=C().def_float_atol,plot=B,plot_title=F,clean_run=B,debug_mode=B,time_integration=B):
	J=value0_array;I=debug_mode;G=clean_run;F=dim
	if I:G=B
	if G:I=B
	C=R(J);H=R(base0_array);K=[C,H]
	if A.any([A.ndim>2 for A in K]):raise M(f"")
	if A.any([A.shape!=C.shape for A in K[2:]]):raise M(f"")
	if F>1:raise NotImplementedError('')
	C=a.deepcopy(J);P=D if C.ndim==H.ndim else B
	if not G:0
	for E in O(C.shape[F]):
		if not G:0
		Q=C[E,:]if F==0 else C[:,E]
		if P:L=H[E,:]if F==0 else H[:,E]
		else:L=H
		N=b(value0_array=Q,base0_array=L,max_iter=max_iter,cor=cor,always_run_first_iter=always_run_first_iter,behaviour=behaviour,inter_behaviour=inter_behaviour,inter_behaviour_min_thr=inter_behaviour_min_thr,inter_behaviour_max_thr=inter_behaviour_max_thr,check_behaviour=check_behaviour,min_change_v_thr=min_change_v_thr,plot=plot,plot_title=plot_title+f" {E}",clean_run=G,debug_mode=I,time_integration=time_integration)
		if F==0:C[E,:]=N
		else:C[:,E]=N
	return C
if __name__=='__main__':
	E,l=X(reach_ids=(74292500201,),swot_file_pattern=Path('C:\\Users\\isadora.rezende\\Downloads')/'{}_SWOT.nc',node_vars=(P,Q,U,V));o=E[G][U][0];H=E[G][P][0];p=E[G][Q][0];q=E[G][V][0];c=A.arange(0,H.shape[0]*2e2,2e2)
	for I in O(2,H.shape[1]):J=H[:,I];J.shape=J.size,1;L=dict(dim=1,value0_array=J,base0_array=c,max_iter=10,cor=200,behaviour=n,check_behaviour=F,min_change_v_thr=.0001,plot=D);K(inter_behaviour=B,plot_title=f"{I}",debug_mode=B,**L);K(inter_behaviour=D,inter_behaviour_min_thr=.01,inter_behaviour_max_thr=A.inf,plot_title=f"{I}",debug_mode=D,**L);break