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

R='node'
L='width'
K='wse'
o=''
n=''
m=''
l=''
Q=range
e=''
d='ignore'
c=''
b=str
T=TypeError
P=''
H=.0
F=True
C=AssertionError
B=False
import copy as M,warnings as N,numpy as A
from sic4dvar_classes.sic4dvar_0_defaults import SIC4DVarLowCostDefaults as D
from sic4dvar_functions.y633 import b as A2
from sic4dvar_functions.helpers.helpers_arrays import masked_array_to_nan_array as O,nan_array_to_masked_array as G,find_nearest as S
from sic4dvar_functions.q558 import D as U
from pathlib import Path
from sic4dvar_functions.io.reader_swot_obs import get_vars_from_swot_nc as V
def Z(values0_array,space0_array,clean_run=B):
	E=space0_array;C=values0_array;H,R=0,1
	if not clean_run:0
	if C.ndim!=2:raise T(f"")
	if(C.shape[H],)!=E.shape:L=f"";raise T(L)
	N=C.shape[0];D=M.deepcopy(C)
	try:I=F;J=D.get_fill_value();D=O(D)
	except AttributeError:I=B;J=None
	G=M.deepcopy(D);P=O(M.deepcopy(E));K=A.count_nonzero(A.isfinite(G),axis=0);Q=A.nonzero((K>=1)&(K<G.shape[0]))[0];return D,G,P,Q,N,I,J
def a(values_out_array,space1_array,dx_max_in,dx_max_out,float_atol=D().def_float_atol):
	N=dx_max_out;H=space1_array;G=values_out_array;J=A.full_like(G,fill_value=A.nan,dtype=A.float32);I=A.full_like(G,fill_value=A.nan,dtype=A.float32);O=Q(G.shape[0])
	for D in Q(G.shape[1]):
		P=A.isnan(G[:,D])
		if A.all(P):continue
		if not A.any(P):J[:,D]=M.deepcopy(G[:,D]);continue
		E=A.isfinite(G[:,D]).nonzero()[0];R=len(E)
		if R==0:raise C('')
		elif R==1:B=E[0];I[:,D]=B;I[B,D]=A.nan;continue
		J[:,D]=U(values_in_array=G[:,D],base_in_array=H,limits='linear',check_nan=F,float_atol=float_atol);K=M.deepcopy(E)
		for B in O:
			if B in E:I[B,D]=B;continue
			if B>E[-1]:
				I[B,D]=E[-1]
				if A.abs(H[E[-1]]-H[B])>N:J[B,D]=A.nan
				continue
			if B<E[0]:
				I[B,D]=E[0]
				if A.abs(H[E[0]]-H[B])>N:J[B,D]=A.nan
				continue
			T,L=S([A for A in K if A!=B],B)
			if L<B:K=K[T:]
			I[B,D]=L
			if A.abs(H[L]-H[B])>dx_max_in:J[B,D]=A.nan
	del O;return I,J
def W(values0_array,space0_array,dx_max_in,dx_max_out,dw_min,float_atol=D().def_float_atol,interp_missing_nodes=B,clean_run=B,debug_mode=B):
	K=clean_run;E=debug_mode
	if E:K=B
	if K:E=B
	D,I,j,k,q,S,T=Z(values0_array=values0_array,space0_array=space0_array,clean_run=K)
	if A.all(A.isnan(I)):
		if S:D=G(D,fill_value=T)
		return D
	if A.all(A.isfinite(I)):
		if S:D=G(D,fill_value=T)
		return D
	if interp_missing_nodes:U=A.full(I.shape[0],fill_value=F)
	else:U=A.count_nonzero(A.isfinite(I),axis=1)>0
	l,L=a(values_out_array=D,space1_array=j,dx_max_in=dx_max_in,dx_max_out=dx_max_out,float_atol=float_atol)
	for J in k:
		r=c+b(J);M=A.isnan(I[:,J])
		if A.all(~M):raise C(P)
		if A.all(M):raise C(P)
		if E:
			V=A.nonzero(M)[0];m=A.nonzero(U)[0]
			for n in V:
				if n not in m:0
		V=A.nonzero(M&U)[0]
		for O in V:
			if E:0
			W=int(l[O,J]);X=I[W,J]
			if A.isnan(X):raise C('')
			h=A.nonzero(A.isfinite(L[O,:])&A.isfinite(L[W,:]))[0]
			if h.size==0:
				if E:0
				continue
			o,Q,Y=0,H,H
			for f in h:
				if f==J:continue
				i=L[O,f];g=L[W,f]
				if A.isnan(i):raise C('')
				if A.isnan(g):raise C('')
				p=g-i;R=(g-X)**2;R=1./max(R,dw_min);Q+=p*R;Y+=R;o+=1
			if Y>H:
				with N.catch_warnings():N.filterwarnings(d,message=e);Q=Q/Y;D[O,J]=X-Q
				if E:0
			elif E:0
	if not K:0
	if S:D=G(D,fill_value=T)
	return D
def X(values0_array,space0_array,weight0_array,dx_max_in,dx_max_out,dw_min,weight_exp_beta=.01,float_atol=.01,clean_run=B,debug_mode=B):
	X=weight0_array;W=values0_array;K=clean_run;E=debug_mode
	if E:K=B
	if K:E=B
	if W.shape!=X.shape:raise T(l)
	D,M,q,r,x,Q,R=Z(values0_array=W,space0_array=space0_array,clean_run=K)
	if A.all(A.isnan(M)):
		if Q:D=G(D,fill_value=R)
		return D
	if A.all(A.isfinite(M)):
		if Q:D=G(D,fill_value=R)
		return D
	y,S=a(values_out_array=D,space1_array=q,dx_max_in=dx_max_in,dx_max_out=dx_max_out,float_atol=float_atol);I=O(X);S[A.isnan(I)]=A.nan;s=A.nanmax(I)-A.nanmin(I)
	for J in r:
		z=c+b(J);L=A.isnan(M[:,J])
		if A.all(~L):raise C(P)
		if A.all(L):raise C(P)
		Y=A.isfinite(I[:,J])
		if E:
			t=A.nonzero(L)[0];u=A.nonzero(Y)[0]
			for F in t:
				if F not in u:0
		v=A.nonzero(L&Y)[0]
		for F in v:
			if E:0
			f=I[F,J]
			if A.isnan(f):raise C(m)
			g=A.isfinite(S[F,:]).nonzero()[0]
			if g.size==0:
				if E:0
				continue
			w,h,U=0,H,H
			for V in g:
				if V==J:continue
				i=S[F,V];j=I[F,V]
				if A.isnan(i):raise C(n)
				if A.isnan(j):raise C(o)
				k=A.abs(f-j);k/=s;p=A.exp(-weight_exp_beta*k);h+=i*p;U+=p;w+=1
			if U>H:
				with N.catch_warnings():N.filterwarnings(d,message=e);D[F,J]=h/U
				if E:0
			elif E:0
	if not K:0
	if Q:D=G(D,fill_value=R)
	return D
def h(values0_array,space0_array,weight0_array,values_avg0_array,weight_avg0_array,dx_max_in,dx_max_out,dw_min,weight_exp_beta=.01,float_atol=.01,clean_run=B,debug_mode=B):
	q=weight0_array;p=values0_array;U=clean_run;Q=float_atol;J=debug_mode
	if J:U=B
	if U:J=B
	if p.shape!=q.shape:raise T(l)
	E,f,A3,A4,AE,g,h=Z(values0_array=p,space0_array=space0_array,clean_run=U)
	if A.all(A.isnan(f)):
		if g:E=G(E,fill_value=h)
		return E
	if A.all(A.isfinite(f)):
		if g:E=G(E,fill_value=h)
		return E
	_,R=a(values_out_array=E,space1_array=A3,dx_max_in=dx_max_in,dx_max_out=dx_max_out,float_atol=Q);D=O(q);i=O(values_avg0_array);j=O(weight_avg0_array);A5=int(abs(A.log10(Q)));D,i,j,R=[A.round(B,A5)for B in[D,i,j,R]];K=A2(width=i,elevation=j,check_nan=F,sort=F,check_increasing=F,float_atol=Q);D[D<K.min_elevation-Q]=A.nan;D[D>K.max_elevation+Q]=A.nan;V=A.full_like(D,fill_value=A.nan);S=D.flatten();S=S[A.isfinite(S)];r=K.get_width_by_elevation(S);V[A.isfinite(D)]=M.deepcopy(r);del r,S;R[A.isnan(D)]=A.nan;A6=(K.max_elevation-K.min_elevation)/(K.max_width-K.min_width)
	for I in A4:
		AF=c+b(I);W=A.isnan(f[:,I])
		if A.all(~W):raise C(P)
		if A.all(W):raise C(P)
		s=A.isfinite(D[:,I]);t=A.isfinite(V[:,I])
		if J:
			A7=A.nonzero(W)[0];A8=A.nonzero(s)[0];A9=A.nonzero(t)[0]
			for u in A7:
				if u not in A8:0
				elif u not in A9:0
		AA=A.nonzero(W&s&t)[0]
		for L in AA:
			if J:0
			v=D[L,I];w=V[L,I]
			if A.isnan(v):raise C(m)
			if A.isnan(w):C('')
			x=A.isfinite(R[L,:]).nonzero()[0]
			if x.size==0:
				if J:0
				continue
			AB,y,k=0,H,H
			for X in x:
				if X==I:continue
				z=R[L,X];A0=D[L,X];A1=V[L,X]
				if A.isnan(z):raise C(n)
				if A.isnan(A0):raise C(o)
				if A.isnan(A1):raise C('')
				AC=A.abs(v-A0);AD=A.abs(w-A1);Y=A.exp(-weight_exp_beta*AC*AD*A6);Y=max(Y,dw_min);y+=z*Y;k+=Y;AB+=1
			if k>H:
				with N.catch_warnings():N.filterwarnings(d,message=e);E[L,I]=y/k
				if J:0
			elif J:0
	if not U:0
	if g:E=G(E,fill_value=h)
	return E
if __name__=='__main__':
	Y=Path('C:\\Users\\isadora.rezende\\PhD');E=V(reach_ids=(74259000071,),swot_file_pattern=Y/'Datasets'/'PEPSI'/'Ohio'/'{}_SWOT.nc',node_vars=(K,L));I=E[R][K][0];f=E[R][L][0];J=A.arange(0,I.shape[0]*2e2,2e2)
	with A.printoptions(precision=3,suppress=F):0
	g=W(values0_array=I,space0_array=J,dx_max_in=A.inf,dx_max_out=5e3,dw_min=.01,clean_run=B,debug_mode=B)
	with A.printoptions(precision=3,suppress=F):0
	with A.printoptions(precision=3,suppress=F):0
	i=X(values0_array=f,space0_array=J,weight0_array=g,dx_max_in=A.inf,dx_max_out=3e2,dw_min=.1,clean_run=B,debug_mode=B)
	with A.printoptions(precision=3,suppress=F):0