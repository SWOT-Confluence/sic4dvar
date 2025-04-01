i='debitance'
h='hydraulic_radius'
f='should be always increasing'
Z='perimeter'
e=str
d=range
Y=any
U='area'
T='z'
S='w_up_bound'
R='w_low_bound'
Q='z_up_bound'
P='z_low_bound'
G=list
O=AssertionError
M='bottom'
L='top'
N=TypeError
K=.0
I='elevation'
B=property
H='width'
F=False
D=True
C=None
import copy as E,pathlib
from typing import Iterable,Literal,Tuple
import numpy as A,pandas as q
from shapely import LineString as j
from sic4dvar_classes.sic4dvar_0_defaults import SIC4DVarLowCostDefaults as a
from sic4dvar_functions.helpers.helpers_arrays import arrays_rmv_nan_pair as b,arrays_check_increase as W,iterable_to_flattened_array as X,array_fix_next_same as c,arrays_bounds as J
from sic4dvar_functions.helpers.helpers_generic import pairwise as k
from sic4dvar_functions.helpers.helpers_plot import helper_plot_lines as l
from sic4dvar_functions.a import D as r
def m(width_array,elevation_array,comp_area=D,comp_per=D,check_nan=D,sort=D,check_increasing=D):
	G=comp_per;E=comp_area;C=elevation_array;B=width_array
	if check_nan:B,C,U=b(B,C)
	else:B=X(B);C=X(C)
	B=B.astype(A.float32);C=C.astype(A.float32)
	if B.size<2:0
	if sort:C=C[A.argsort(B)];B.sort()
	if check_increasing:
		if not W(C,remove_nan=F):0
		if not W(B,remove_nan=F):0
	H,I=[B[0]/2],[K]
	for((J,L),(M,N))in k(zip(B/2,C)):
		if G:
			D=j([[J,L],[M,N]]).length
			if A.isnan(D)or not A.isfinite(D)or D<K:0
			H.append(D)
		if E:Q=J;R=M;S=N-L;T=(Q+R)*S;I.append(T)
	if G:O=A.array(H)*2
	else:O=A.empty(0,dtype=A.float32)
	if E:P=A.array(I)
	else:P=A.empty(0,dtype=A.float32)
	return P,O
def g(width_array,elevation_array,comp_area=D,comp_per=D,check_nan=D,sort=D,check_increasing=D):
	D=comp_area;C=width_array;B=elevation_array;F,G=m(width_array=C,elevation_array=B,comp_area=D,comp_per=comp_per,check_nan=check_nan,sort=sort,check_increasing=check_increasing);E,H=A.nancumsum(F),A.nancumsum(G)
	if D:
		I=C[-1];J=B[-1]-B[0]
		if A.round(E[-1],2)>A.round(J*I,2)+1:0
	return E,H
class V(RuntimeError):
	def __init__(A,*B):
		if B:A.message=B[0]
		else:A.message=C
	def __str__(A):
		if A.message:return f"HydraulicGeomParamsError, {A.message}"
		else:return f"HydraulicGeomParamsError"
class n:
	def __init__(B,width,elevation,check_nan=D,sort=D,check_increasing=D,float_atol=a().def_cs_float_atol):D=float_atol;B.__n_sig_dig=int(abs(A.log10(D)));B.__float_atol=D;B._width=E.deepcopy(width);B._elevation=E.deepcopy(elevation);B._wet_a_array=A.empty(0,dtype=A.float32);B._wet_per_array=A.empty(0,dtype=A.float32);B._hydro_rad_array=A.empty(0,dtype=A.float32);B._debitance_array=A.empty(0,dtype=A.float32);B._wet_a=C;B._wet_per=C;B._hydro_rad=C;B._debitance=C;B._min_width=C;B._max_width=C;B._min_elevation=C;B._max_elevation=C;B._depth=C;B.__init_helper(check_nan=check_nan,sort=sort,check_increasing=check_increasing)
	@classmethod
	def from_dataframe(B,df,check_nan=D,sort=D,check_increasing=D,elevation_col=T,width_col='w',float_atol=a().def_cs_float_atol):return B(width=df[width_col].to_numpy().astype(A.float32),elevation=df[elevation_col].to_numpy().astype(A.float32),check_nan=check_nan,sort=sort,check_increasing=check_increasing,float_atol=float_atol)
	def __v_close(B,v0,v1):return A.isclose(v0,v1,rtol=K,atol=B.__float_atol)
	def __init_helper(B,check_nan,sort,check_increasing):
		if check_nan:B._width,B._elevation,D=b(B._width,B._elevation)
		else:B._width=X(B._width);B._elevation=X(B._elevation)
		if B._width.size!=B._elevation.size:0
		B._width=B._width.astype(A.float32);B._elevation=B._elevation.astype(A.float32);B._width=A.round(B._width,B.__n_sig_dig);B._elevation=A.round(B._elevation,B.__n_sig_dig)
		if B._width.size<2:0
		if sort:B._elevation=B._elevation[A.argsort(B._width)];B._width=A.sort(B._width)
		if check_increasing:
			if not W(B._elevation,remove_nan=F):0
			if not W(B._width,remove_nan=F):0
		B._width[B.__v_close(B._width,K)]=K
		if A.any(B._width<K):raise V(f"")
		B._width=c(B._width,float_atol=B.__float_atol);B._elevation=c(B._elevation,float_atol=B.__float_atol);B._wet_a_array=A.empty(0,dtype=A.float32);B._wet_per_array=A.empty(0,dtype=A.float32);B._hydro_rad_array=A.empty(0,dtype=A.float32);B._debitance_array=A.empty(0,dtype=A.float32);B._wet_a=C;B._wet_per=C;B._hydro_rad=C;B._debitance=C;B._min_width=B._width[0];B._max_width=B._width[-1];B._min_elevation=B._elevation[0];B._max_elevation=B._elevation[-1];B._depth=B.elevation-B.min_elevation
	def __get_var1_by_w_z(G,var0_value,var0_ref,var1_ref):
		l='perim';k='radius';j='comp_per';i='comp_area';h='elev';e='debit';V=var0_ref;R='var0_ref';P=var0_value;L=var1_ref;J='var1_ref'
		if V not in[I,H]:raise N('')
		if L in[U,Z]:L=f"wetted_{L}"
		if V==L:return P
		try:P[0]
		except N:W=D;M=A.array([P],dtype=A.float32)
		except IndexError:W=D;M=A.array([P],dtype=A.float32)
		else:W=F;M=A.array(E.deepcopy(P),dtype=A.float32)
		if V==H:X=A.concatenate([G.width,M],axis=0);a=A.concatenate([G.elevation,A.full_like(M,fill_value=A.nan)],axis=0);S=E.deepcopy(X);b=I
		else:X=A.concatenate([G.width,A.full_like(M,fill_value=A.nan)],axis=0);a=A.concatenate([G.elevation,M],axis=0);S=E.deepcopy(a);b=H
		M=(A.round(M,G.__n_sig_dig)*10**G.__n_sig_dig).astype(A.int32);S=(A.round(S,G.__n_sig_dig)*10**G.__n_sig_dig).astype(A.int32)
		try:B=q.DataFrame({R:S,H:X,I:a})
		except ValueError as m:raise m
		B.drop_duplicates(R,keep='first',inplace=D);B[R]=B[R].astype(A.int32);B.set_index(R,inplace=D);B.sort_index(inplace=D);B[b]=r(values_in_array=B[b].to_numpy(),base_in_array=B.index,limits='linear',check_nan=F,float_atol=G.__float_atol);C=B[H].to_numpy()>=G.min_width
		if Y([A in L for A in[h,H]]):
			if h in L:B[J]=B[I]
			else:B[J]=B[H]
			if A.any(~C):B.loc[~C,J]=A.nan
		else:
			B[J]=A.full(B.shape[0],A.nan);c={i:F,j:F}
			if Y([A in L for A in[e,k,U]]):c[i]=D
			if Y([A in L for A in[e,k,l]]):c[j]=D
			d,T=g(width_array=B[H].to_numpy()[C],elevation_array=B[I].to_numpy()[C],check_nan=F,sort=F,check_increasing=F,**c);n=max(G.__float_atol,G.min_width);T=A.maximum(T,n)
			if U in L:
				B.loc[C,J]=d
				if A.any(~C):B.loc[~C,J]=K
			elif l in L:
				B.loc[C,J]=T
				if A.any(~C):B.loc[~C,J]=K
			else:
				B.loc[C,J]=d/T
				if A.any(~C):B.loc[~C,J]=K
				if e in L:o=B.loc[C,J];p=d*o**(2./3.);B.loc[C,J]=p
		f=B.loc[M].index
		if f.size!=M.size:raise O('')
		Q=B.loc[f,J].values
		if Q.size!=M.size:raise O('')
		Q=A.round(Q,G.__n_sig_dig)
		if W:return Q[0]
		return Q
	def __ext_shrink_by_var(B,var_ref,value,loc_modif=C):
		U='??';T='adapt';S=value;R=var_ref;Q='shrink';P='extend';K=loc_modif
		if R==I:E=S
		else:
			E=B.__get_var1_by_w_z(S,R,I)
			if A.isnan(E):0
		if E>=B.max_elevation or B.__v_close(E,B.max_elevation):K=L
		elif E<=B.min_elevation or B.__v_close(E,B.min_elevation):K=M
		else:
			if L in K.lower():K=L
			elif'bot'in K.lower():K=M
			if K is C:0
		if R==H:G=S
		else:
			G=B.__get_var1_by_w_z(S,R,H)
			if B.__v_close(G,B.min_width):G=B.min_width
		if A.isnan(G):J=P;G=B.min_width
		elif B.min_width<G<B.max_width:J=Q
		elif B.__v_close(G,B.min_width):
			if E<B.min_elevation:J=P
			elif B.min_elevation<E<B.max_elevation:J=Q
			elif E>B.max_elevation:0
			else:J=T
		elif B.__v_close(G,B.max_width):
			if E>B.max_elevation:J=P
			elif B.min_elevation<E<B.max_elevation:J=Q
			elif E<B.min_elevation:raise O(U)
			else:J=T
		elif G>B.max_width:J=P
		else:raise O(U)
		if K==L and J==P:B._width=A.append(B._width,G);B._elevation=A.append(B._elevation,E)
		elif K==M and J==P:B._elevation=A.append(E,B._elevation);B._width=A.append(G,B._width)
		elif K==L and J==Q:
			N=(B._width<=G)&(B._elevation<E)
			if A.any(N):
				B._elevation=B._elevation[N];B._width=B._width[N]
				if not B.__v_close(B._elevation[-1],E)or not B.__v_close(B._width[-1],G):
					B._elevation=A.sort(A.append(B._elevation,E))
					if G>B._width[0]:B._width=A.append(B._width,G)
					else:B._width=A.append(B._width,B._width[-1])
			else:raise O(U)
		elif K==M and J==Q:
			N=(B._width>=G)&(B._elevation>E)
			if A.any(N):
				B._elevation=B._elevation[N];B._width=B._width[N]
				if not B.__v_close(B._elevation[0],E)or not B.__v_close(B._width[0],G):
					B._elevation=A.sort(A.append(E,B._elevation))
					if G<B._width[0]:B._width=A.append(G,B._width)
					else:B._width=A.append(B._width[0],B._width)
			else:raise O(U)
		elif K==L and J==T:B._width[-1]=G;B._elevation[-1]=E
		elif K==M and J==T:B._width[0]=G;B._elevation[0]=E
		B.__init_helper(check_nan=F,sort=F,check_increasing=D)
	def compute(B):B._wet_a_array,B._wet_per_array=g(B._width,B._elevation,check_nan=F,sort=F,check_increasing=F);B._wet_a,B._wet_per=B._wet_a_array[-1],B._wet_per_array[-1];C=E.deepcopy(B._wet_per_array);D=max(B.__float_atol,B.min_width);C=A.maximum(C,D);B._hydro_rad_array=B._wet_a_array/C;B._hydro_rad=B._hydro_rad_array[-1];B._debitance_array=B._wet_a_array*B._hydro_rad_array**(2./3.);B._debitance=B._debitance_array[-1]
	def plot(B,show=D,plot_title='',fig_width=15,fig_height=5,y_axis_title='elevation (m)',x_axis_title='width (m)',y_lim=C,x_max_lim=C,output_file=C):
		D=x_max_lim;H=E.deepcopy(-B._width[::-1]/2);I=E.deepcopy(B._width/2);J=A.append(H,I);K=E.deepcopy(B._elevation[::-1]);L=E.deepcopy(B._elevation);M=A.append(K,L)
		if D is not C:G=-D/2,D/2
		else:G=C
		l(xs=[J],ys=[M],show=show,title=plot_title,x_axis_title=x_axis_title,y_axis_title=y_axis_title,fig_width=fig_width,fig_height=fig_height,y_lim=y_lim,x_lim=G,add_legend=F,output_file=output_file)
	def combine_with_other(I,other,self_z_up_bound_array=C,self_z_low_bound_array=C,self_w_up_bound_array=C,self_w_low_bound_array=C,other_z_up_bound_array=C,other_z_low_bound_array=C,other_w_up_bound_array=C,other_w_low_bound_array=C,preference='wide'):
		A3='w_up_other';A2='w_low_other';A1='z_up_other';A0='z_low_other';z='w_up_self';y='w_low_self';x='z_up_self';w='z_low_self';p='self';o='other';n=other_w_low_bound_array;m=other_w_up_bound_array;l=other_z_low_bound_array;k=other_z_up_bound_array;j=self_w_low_bound_array;i=self_w_up_bound_array;h=self_z_low_bound_array;g=self_z_up_bound_array;c='w_other';b='w_self';N=other;h,g=J(ref_array=I.elevation,value0_low_bound_array=h,value0_up_bound_array=g);j,i=J(ref_array=I.width,value0_low_bound_array=j,value0_up_bound_array=i);l,k=J(ref_array=N.elevation,value0_low_bound_array=l,value0_up_bound_array=k);n,m=J(ref_array=N.width,value0_low_bound_array=n,value0_up_bound_array=m);s=A.round(E.deepcopy(I.elevation)*10**I.__n_sig_dig,0).astype(A.int64);t=A.round(E.deepcopy(N.elevation)*10**I.__n_sig_dig,0).astype(A.int64);u=A.concatenate([E.deepcopy(s),E.deepcopy(t)]);K=A.full_like(u,fill_value=A.nan,dtype=N.width.dtype);B=q.DataFrame({T:E.deepcopy(u),b:K,c:K,w:K,x:K,y:K,z:K,A0:K,A1:K,A2:K,A3:K});B.drop_duplicates(subset=T,inplace=D,ignore_index=D);B.sort_values(T,inplace=D);B.set_index(T,drop=D,inplace=D);U=B.index[A.isin(B.index.to_numpy(),s).nonzero()[0]];W=B.index[A.isin(B.index.to_numpy(),t).nonzero()[0]];B.loc[U,b]=E.deepcopy(I.width);B.loc[U,w]=E.deepcopy(h);B.loc[U,x]=E.deepcopy(g);B.loc[U,y]=E.deepcopy(j);B.loc[U,z]=E.deepcopy(i);B.loc[W,c]=E.deepcopy(N.width);B.loc[W,A0]=E.deepcopy(l);B.loc[W,A1]=E.deepcopy(k);B.loc[W,A2]=E.deepcopy(n);B.loc[W,A3]=E.deepcopy(m)
		for L in B.columns:
			if A.any(A.isfinite(B[L].to_numpy())):B[L]=r(values_in_array=B[L].to_numpy(),base_in_array=B.index.to_numpy(),limits=C,check_nan=F,float_atol=I.__float_atol)
			elif'up'in L:B[L]=A.inf
			else:B[L]=-A.inf
		M,X,Y,Z,a=[],[],[],[],[]
		for G in d(B.shape[0]):
			if A.isnan(B.iloc[G][b]):H=o
			elif A.isnan(B.iloc[G][c]):H=p
			elif'wi'in preference:
				if B.iloc[G][b]>=B.iloc[G][c]:H=p
				else:H=o
			elif B.iloc[G][b]<=B.iloc[G][c]:H=p
			else:H=o
			if G==0:M.append(B.iloc[G][f"w_{H}"]);X.append(B.iloc[G][f"z_low_{H}"]);Y.append(B.iloc[G][f"z_up_{H}"]);a.append(B.iloc[G][f"w_low_{H}"]);Z.append(B.iloc[G][f"w_up_{H}"])
			elif B.iloc[G][f"w_{H}"]>=M[-1]:M.append(B.iloc[G][f"w_{H}"]);X.append(B.iloc[G][f"z_low_{H}"]);Y.append(B.iloc[G][f"z_up_{H}"]);a.append(B.iloc[G][f"w_low_{H}"]);Z.append(B.iloc[G][f"w_up_{H}"])
			else:M.append(M[-1]);X.append(X[-1]);Y.append(Y[-1]);a.append(a[-1]);Z.append(Z[-1])
		B.index=B.index.astype(I.elevation.dtype);B.index=B.index*10**-I.__n_sig_dig;I._elevation=B.index.to_numpy(dtype=I.elevation.dtype);I._width=A.array(M,dtype=I.width.dtype)
		try:I.__init_helper(check_nan=F,sort=F,check_increasing=D)
		except V as v:
			A4=e(v.args[0])
			if f in A4:raise O(f"")
			else:raise v
		return{P:X,Q:Y,R:a,S:Z}
	def add_wet_bathy_from_other(B,other,self_z_up_bound_array=C,self_z_low_bound_array=C,self_w_up_bound_array=C,self_w_low_bound_array=C,other_z_up_bound_array=C,other_z_low_bound_array=C,other_w_up_bound_array=C,other_w_low_bound_array=C):
		b=other_w_low_bound_array;a=other_w_up_bound_array;Z=other_z_low_bound_array;Y=other_z_up_bound_array;X=self_w_low_bound_array;W=self_w_up_bound_array;N=self_z_low_bound_array;M=self_z_up_bound_array;C=other;N,M=J(ref_array=B.elevation,value0_low_bound_array=N,value0_up_bound_array=M);X,W=J(ref_array=B.width,value0_low_bound_array=X,value0_up_bound_array=W);Z,Y=J(ref_array=C.elevation,value0_low_bound_array=Z,value0_up_bound_array=Y);b,a=J(ref_array=C.width,value0_low_bound_array=b,value0_up_bound_array=a)
		if B.min_elevation<C.min_elevation:return{P:N,Q:M,R:X,S:W}
		O,H=[],[];T,U,I,K=[],[],[],[]
		for E in d(C.elevation.size):
			if C.elevation[E]>=B.min_elevation:break
			if C.width[E]>=B.min_width:break
			O.append(C.elevation[E]);H.append(C.width[E]);T.append(Z[E]);U.append(Y[E]);I.append(b[E]);K.append(a[E])
		if len(O)==0:return{P:N,Q:M,R:X,S:W}
		c=C.get_width_by_elevation(B.min_elevation);g=B.min_width
		if c<g:O.append(B.min_elevation);H.append(c)
		else:O.append(B.min_elevation);H.append(g)
		T.append(-A.inf);U.append(A.inf);I.append(-A.inf);K.append(A.inf);L=(B._width>H[-1]).nonzero()[0]
		if L.size>0:T+=G(N[L]);U+=G(M[L]);I+=G(X[L]);K+=G(W[L]);B._elevation=A.append(A.array(O),B._elevation[L]);B._width=A.append(A.array(H),B._width[L])
		else:T=[*T,N[-1]];U=[*U,M[-1]];I=[*I,I[-1]];K=[*K,K[-1]];B._elevation=A.append(A.array(O),B._elevation[-1]);B._width=A.append(A.array(H),H[-1])
		try:B.__init_helper(check_nan=F,sort=F,check_increasing=D)
		except V as h:
			i=e(h.args[0])
			if f in i:0
			else:raise h
		return{P:A.array(T,dtype=B._elevation.dtype),Q:A.array(U,dtype=B._elevation.dtype),R:A.array(I,dtype=B._elevation.dtype),S:A.array(K,dtype=B._elevation.dtype)}
	def add_dry_bathy_from_other(B,other,self_z_up_bound_array=C,self_z_low_bound_array=C,self_w_up_bound_array=C,self_w_low_bound_array=C,other_z_up_bound_array=C,other_z_low_bound_array=C,other_w_up_bound_array=C,other_w_low_bound_array=C):
		c=other_w_low_bound_array;b=other_w_up_bound_array;a=other_z_low_bound_array;Z=other_z_up_bound_array;N=self_w_low_bound_array;M=self_w_up_bound_array;L=self_z_low_bound_array;K=self_z_up_bound_array;C=other;L,K=J(ref_array=B.elevation,value0_low_bound_array=L,value0_up_bound_array=K);N,M=J(ref_array=B.width,value0_low_bound_array=N,value0_up_bound_array=M);a,Z=J(ref_array=C.elevation,value0_low_bound_array=a,value0_up_bound_array=Z);c,b=J(ref_array=C.width,value0_low_bound_array=c,value0_up_bound_array=b)
		if B.max_elevation>C.max_elevation:return{P:L,Q:K,R:N,S:M}
		T,E=[],[];U,W,X,Y=[],[],[],[]
		for H in d(C.elevation.size):
			g=C.elevation[H];h=C.width[H]
			if g<=B.max_elevation:continue
			if h<=B.max_width:continue
			T.append(g);E.append(h);U.append(a[H]);W.append(Z[H]);X.append(c[H]);Y.append(b[H])
		if len(T)==0:return{P:L,Q:K,R:N,S:M}
		i=C.get_width_by_elevation(B.max_elevation);j=B.max_width;T=[B.max_elevation]+T
		if i>j:E=[i]+E
		else:E=[j]+E
		U.append(-A.inf);W.append(A.inf);X.append(-A.inf);Y.append(A.inf);I=B._width<E[0];U=G(L[I])+U;W=G(K[I])+W;X=G(N[I])+X;Y=G(M[I])+Y;B._elevation=A.append(B._elevation[I],T);B._width=A.append(B._width[I],E)
		try:B.__init_helper(check_nan=F,sort=F,check_increasing=D)
		except V as k:
			l=e(k.args[0])
			if f in l:raise O(f"")
			else:raise k
		return{P:A.array(U,dtype=B._elevation.dtype),Q:A.array(W,dtype=B._elevation.dtype),R:A.array(X,dtype=B._elevation.dtype),S:A.array(Y,dtype=B._elevation.dtype)}
	def modify_by_elevation(A,value,loc_modif=C):A.__ext_shrink_by_var(I,value=value,loc_modif=loc_modif)
	def modify_by_width(A,value,loc_modif=C):A.__ext_shrink_by_var(H,value=value,loc_modif=loc_modif)
	def extend_top_by_elevation(A,value):
		B=value
		if B>A.max_elevation:A.modify_by_elevation(B)
		if A.__v_close(B,A.max_elevation):return
		else:raise N(f"")
	def extend_top_by_width(A,value):
		B=value
		if B>A.max_width:A.modify_by_width(B)
		if A.__v_close(B,A.max_width):return
		else:raise N(f"")
	def extend_bottom_by_elevation(A,value):
		B=value
		if B<A.min_elevation:A.modify_by_elevation(B)
		if A.__v_close(B,A.min_elevation):return
		else:raise N(f"")
	def extend_bottom_by_width(A,value):
		B=value
		if B<A.min_width:raise N(f"")
		if A.__v_close(B,A.min_width):return
		raise V(f"")
	def shrink_top_by_elevation(A,value):
		B=value
		if A.min_elevation<B<A.max_elevation:A.modify_by_elevation(B,loc_modif=L);return
		if A.__v_close(B,A.max_elevation):return
		if A.__v_close(B,A.min_elevation):B=A.min_elevation;A.modify_by_elevation(B,loc_modif=L);return
		raise N(f"")
	def shrink_top_by_width(A,value):
		B=value
		if A.min_width<B<A.max_width:A.modify_by_width(B,loc_modif=L);return
		if A.__v_close(B,A.max_width):return
		if A.__v_close(B,A.min_width):B=A.min_width;A.modify_by_width(B,loc_modif=L);return
		else:raise N(f"")
	def shrink_bottom_by_elevation(A,value):
		B=value
		if A.min_elevation<B<A.max_elevation:A.modify_by_elevation(B,loc_modif=M);return
		if A.__v_close(B,A.min_elevation):return
		if A.__v_close(B,A.max_elevation):B=A.max_elevation;A.modify_by_elevation(B,loc_modif=M);return
		raise N(f"")
	def shrink_bottom_by_width(A,value):
		B=value
		if A.min_width<B<A.max_width:A.modify_by_width(B,loc_modif=M);return
		if A.__v_close(B,A.min_width):return
		if A.__v_close(B,A.max_width):B=A.max_elevation;A.modify_by_elevation(B,loc_modif=M);return
		else:raise N(f"")
	def get_width_by_elevation(A,value):return A.__get_var1_by_w_z(var0_value=value,var0_ref=I,var1_ref=H)
	def get_elevation_by_width(A,value):return A.__get_var1_by_w_z(var0_value=value,var0_ref=H,var1_ref=I)
	def get_depth_by_width(A,value):B=A.get_elevation_by_width(value);return B-A._min_elevation
	def get_depth_by_elevation(B,value):
		C=value;D=B.get_width_by_elevation(C);E=C-B._min_elevation
		if D>B.min_width:return E
		return A.nan
	def get_area_by_elevation(A,value):return A.__get_var1_by_w_z(var0_value=value,var0_ref=I,var1_ref=U)
	def get_area_by_width(A,value):return A.__get_var1_by_w_z(var0_value=value,var0_ref=H,var1_ref=U)
	def get_perimeter_by_elevation(A,value):return A.__get_var1_by_w_z(var0_value=value,var0_ref=I,var1_ref=Z)
	def get_perimeter_by_width(A,value):return A.__get_var1_by_w_z(var0_value=value,var0_ref=H,var1_ref=Z)
	def get_hydraulic_radius_by_elevation(A,value):return A.__get_var1_by_w_z(var0_value=value,var0_ref=I,var1_ref=h)
	def get_hydraulic_radius_by_width(A,value):return A.__get_var1_by_w_z(var0_value=value,var0_ref=H,var1_ref=h)
	def get_debitance_by_elevation(A,value):return A.__get_var1_by_w_z(var0_value=value,var0_ref=I,var1_ref=i)
	def get_debitance_by_width(A,value):return A.__get_var1_by_w_z(var0_value=value,var0_ref=H,var1_ref=i)
	def __eq__(B,other):
		C=other
		if C.elevation.size!=B.elevation.size:return F
		if A.allclose(C.elevation,B.elevation,rtol=K,atol=B.__float_atol):
			if A.allclose(C.width,B.width,rtol=K,atol=B.__float_atol):return D
		return F
	def __gt__(A,other):
		if A.max_wetted_area>other.max_wetted_area:return D
		return F
	def __ge__(A,other):
		if A.max_wetted_area>=other.max_wetted_area:return D
		return F
	def __ne__(A,other):return not A.__eq__(other)
	def __lt__(A,other):return not A.__ge__(other)
	def __le__(A,other):return not A.__gt__(other)
	@B
	def width(self):return E.deepcopy(self._width)
	@B
	def elevation(self):return E.deepcopy(self._elevation)
	@B
	def wetted_perimeter(self):
		A=self
		if A._wet_per is C:A.compute()
		return E.deepcopy(A._wet_per_array)
	@B
	def wetted_area(self):
		A=self
		if A._wet_a is C:A.compute()
		return E.deepcopy(A._wet_a_array)
	@B
	def hydraulic_radius(self):
		A=self
		if A._hydro_rad is C:A.compute()
		return E.deepcopy(A._hydro_rad_array)
	@B
	def debitance(self):
		A=self
		if A._debitance is C:A.compute()
		return E.deepcopy(A._debitance_array)
	@B
	def depth(self):return E.deepcopy(self._depth)
	@B
	def min_width(self):
		B=self
		if B._min_width is C:B._min_width=A.min(B._width)
		return B._min_width
	@B
	def max_width(self):
		B=self
		if B._max_width is C:B._max_width=A.max(B._width)
		return B._max_width
	@B
	def min_elevation(self):
		B=self
		if B._min_elevation is C:B._min_elevation=A.min(B.elevation)
		return B._min_elevation
	@B
	def max_elevation(self):
		B=self
		if B._max_elevation is C:B._max_elevation=A.max(B.elevation)
		return B._max_elevation
	@B
	def max_depth(self):return A.max(self.depth)
	@B
	def max_wetted_perimeter(self):
		A=self
		if A._wet_per is C:A.compute()
		return A._wet_per
	@B
	def max_wetted_area(self):
		A=self
		if A._wet_a is C:A.compute()
		return A._wet_a
	@B
	def max_hydraulic_radius(self):
		A=self
		if A._hydro_rad is C:A.compute()
		return A._hydro_rad