s='debitance'
p='hydraulic_radius'
o='wide'
n=''
m=IndexError
h=''
g='perimeter'
f=str
e=range
d=any
Y='area'
X='z'
V='w_up_bound'
U='w_low_bound'
T='z_up_bound'
S='z_low_bound'
R=list
Q='bottom'
O='top'
N=TypeError
M=AssertionError
J=.0
I='elevation'
G=property
H='width'
F=False
D=True
C=None
import copy as E,pathlib
from typing import Iterable,Literal,Tuple
import numpy as A,pandas as q
from shapely import LineString as t
from sic4dvar_classes.sic4dvar_0_defaults import SIC4DVarLowCostDefaults as i
from sic4dvar_functions.helpers.helpers_arrays import arrays_rmv_nan_pair as j,arrays_check_increase as Z,iterable_to_flattened_array as a,array_fix_next_same as k,arrays_bounds as K
from sic4dvar_functions.helpers.helpers_generic import pairwise as u
from sic4dvar_functions.helpers.helpers_plot import helper_plot_lines as v
from sic4dvar_functions.D560 import D as r
def w(width_array,elevation_array,comp_area=D,comp_per=D,check_nan=D,sort=D,check_increasing=D):
	G=comp_per;E=comp_area;C=elevation_array;B=width_array
	if check_nan:B,C,W=j(B,C)
	else:B=a(B);C=a(C)
	B=B.astype(A.float32);C=C.astype(A.float32)
	if B.size<2:raise L(n)
	if sort:C=C[A.argsort(B)];B.sort()
	if check_increasing:
		if not Z(C,remove_nan=F):raise L(f"")
		if not Z(B,remove_nan=F):raise L(f"")
	H,I=[B[0]/2],[J]
	for((K,N),(O,P))in u(zip(B/2,C)):
		if G:
			D=t([[K,N],[O,P]]).length
			if A.isnan(D)or not A.isfinite(D)or D<J:raise M('')
			H.append(D)
		if E:S=K;T=O;U=P-N;V=(S+T)*U;I.append(V)
	if G:Q=A.array(H)*2
	else:Q=A.empty(0,dtype=A.float32)
	if E:R=A.array(I)
	else:R=A.empty(0,dtype=A.float32)
	return R,Q
def l(width_array,elevation_array,comp_area=D,comp_per=D,check_nan=D,sort=D,check_increasing=D):
	E=comp_area;D=width_array;B=elevation_array;H,I=w(width_array=D,elevation_array=B,comp_area=E,comp_per=comp_per,check_nan=check_nan,sort=sort,check_increasing=check_increasing);C,J=A.nancumsum(H),A.nancumsum(I)
	if E:
		F=D[-1];G=B[-1]-B[0]
		if A.round(C[-1],2)>A.round(G*F,2)+1:raise M(f"")
	return C,J
class L(RuntimeError):
	def __init__(A,*B):
		if B:A.message=B[0]
		else:A.message=C
	def __str__(A):
		if A.message:return f""
		else:return f""
class b:
	def __init__(B,width,elevation,check_nan=D,sort=D,check_increasing=D,float_atol=i().def_cs_float_atol):D=float_atol;B.__n_sig_dig=int(abs(A.log10(D)));B.__float_atol=D;B._width=E.deepcopy(width);B._elevation=E.deepcopy(elevation);B._wet_a_array=A.empty(0,dtype=A.float32);B._wet_per_array=A.empty(0,dtype=A.float32);B._hydro_rad_array=A.empty(0,dtype=A.float32);B._debitance_array=A.empty(0,dtype=A.float32);B._wet_a=C;B._wet_per=C;B._hydro_rad=C;B._debitance=C;B._min_width=C;B._max_width=C;B._min_elevation=C;B._max_elevation=C;B._depth=C;B.__init_helper(check_nan=check_nan,sort=sort,check_increasing=check_increasing)
	@classmethod
	def from_dataframe(B,df,check_nan=D,sort=D,check_increasing=D,elevation_col=X,width_col='w',float_atol=i().def_cs_float_atol):return B(width=df[width_col].to_numpy().astype(A.float32),elevation=df[elevation_col].to_numpy().astype(A.float32),check_nan=check_nan,sort=sort,check_increasing=check_increasing,float_atol=float_atol)
	def __v_close(B,v0,v1):return A.isclose(v0,v1,rtol=J,atol=B.__float_atol)
	def __init_helper(B,check_nan,sort,check_increasing):
		if check_nan:B._width,B._elevation,D=j(B._width,B._elevation)
		else:B._width=a(B._width);B._elevation=a(B._elevation)
		if B._width.size!=B._elevation.size:raise m('')
		B._width=B._width.astype(A.float32);B._elevation=B._elevation.astype(A.float32);B._width=A.round(B._width,B.__n_sig_dig);B._elevation=A.round(B._elevation,B.__n_sig_dig)
		if B._width.size<2:raise L(n)
		if sort:B._elevation=B._elevation[A.argsort(B._width)];B._width=A.sort(B._width)
		if check_increasing:
			if not Z(B._elevation,remove_nan=F):raise L(f"")
			if not Z(B._width,remove_nan=F):raise L(f"")
		B._width[B.__v_close(B._width,J)]=J
		if A.any(B._width<J):raise L(f"")
		B._width=k(B._width,float_atol=B.__float_atol);B._elevation=k(B._elevation,float_atol=B.__float_atol);B._wet_a_array=A.empty(0,dtype=A.float32);B._wet_per_array=A.empty(0,dtype=A.float32);B._hydro_rad_array=A.empty(0,dtype=A.float32);B._debitance_array=A.empty(0,dtype=A.float32);B._wet_a=C;B._wet_per=C;B._hydro_rad=C;B._debitance=C;B._min_width=B._width[0];B._max_width=B._width[-1];B._min_elevation=B._elevation[0];B._max_elevation=B._elevation[-1];B._depth=B.elevation-B.min_elevation
	def __get_var1_by_w_z(G,var0_value,var0_ref,var1_ref):
		n='';k='perim';j='radius';i='comp_per';h='comp_area';f='elev';c='debit';U=var0_ref;R='var0_ref';P=var0_value;L=var1_ref;K='var1_ref'
		if U not in[I,H]:raise N('')
		if L in[Y,g]:L=f""
		if U==L:return P
		try:P[0]
		except N:V=D;O=A.array([P],dtype=A.float32)
		except m:V=D;O=A.array([P],dtype=A.float32)
		else:V=F;O=A.array(E.deepcopy(P),dtype=A.float32)
		if U==H:W=A.concatenate([G.width,O],axis=0);X=A.concatenate([G.elevation,A.full_like(O,fill_value=A.nan)],axis=0);S=E.deepcopy(W);Z=I
		else:W=A.concatenate([G.width,A.full_like(O,fill_value=A.nan)],axis=0);X=A.concatenate([G.elevation,O],axis=0);S=E.deepcopy(X);Z=H
		O=(A.round(O,G.__n_sig_dig)*10**G.__n_sig_dig).astype(A.int32);S=(A.round(S,G.__n_sig_dig)*10**G.__n_sig_dig).astype(A.int32)
		try:B=q.DataFrame({R:S,H:W,I:X})
		except ValueError as o:raise o
		B.drop_duplicates(R,keep='first',inplace=D);B[R]=B[R].astype(A.int32);B.set_index(R,inplace=D);B.sort_index(inplace=D);B[Z]=r(values_in_array=B[Z].to_numpy(),base_in_array=B.index,limits='linear',check_nan=F,float_atol=G.__float_atol);C=B[H].to_numpy()>=G.min_width
		if d([A in L for A in[f,H]]):
			if f in L:B[K]=B[I]
			else:B[K]=B[H]
			if A.any(~C):B.loc[~C,K]=A.nan
		else:
			B[K]=A.full(B.shape[0],A.nan);a={h:F,i:F}
			if d([A in L for A in[c,j,Y]]):a[h]=D
			if d([A in L for A in[c,j,k]]):a[i]=D
			b,T=l(width_array=B[H].to_numpy()[C],elevation_array=B[I].to_numpy()[C],check_nan=F,sort=F,check_increasing=F,**a);p=max(G.__float_atol,G.min_width);T=A.maximum(T,p)
			if Y in L:
				B.loc[C,K]=b
				if A.any(~C):B.loc[~C,K]=J
			elif k in L:
				B.loc[C,K]=T
				if A.any(~C):B.loc[~C,K]=J
			else:
				B.loc[C,K]=b/T
				if A.any(~C):B.loc[~C,K]=J
				if c in L:s=B.loc[C,K];t=b*s**(2./3.);B.loc[C,K]=t
		e=B.loc[O].index
		if e.size!=O.size:raise M(n)
		Q=B.loc[e,K].values
		if Q.size!=O.size:raise M(n)
		Q=A.round(Q,G.__n_sig_dig)
		if V:return Q[0]
		return Q
	def __ext_shrink_by_var(B,var_ref,value,loc_modif=C):
		W='adapt';V='??';U=value;T=var_ref;S='shrink';R='extend';K=loc_modif
		if T==I:E=U
		else:
			E=B.__get_var1_by_w_z(U,T,I)
			if A.isnan(E):raise L(f"")
		if E>=B.max_elevation or B.__v_close(E,B.max_elevation):K=O
		elif E<=B.min_elevation or B.__v_close(E,B.min_elevation):K=Q
		else:
			if O in K.lower():K=O
			elif'bot'in K.lower():K=Q
			if K is C:raise N('')
		if T==H:G=U
		else:
			G=B.__get_var1_by_w_z(U,T,H)
			if B.__v_close(G,B.min_width):G=B.min_width
		if A.isnan(G):J=R;G=B.min_width
		elif B.min_width<G<B.max_width:J=S
		elif B.__v_close(G,B.min_width):
			if E<B.min_elevation:J=R
			elif B.min_elevation<E<B.max_elevation:J=S
			elif E>B.max_elevation:raise M(V)
			else:J=W
		elif B.__v_close(G,B.max_width):
			if E>B.max_elevation:J=R
			elif B.min_elevation<E<B.max_elevation:J=S
			elif E<B.min_elevation:raise M(V)
			else:J=W
		elif G>B.max_width:J=R
		else:raise M(V)
		if K==O and J==R:B._width=A.append(B._width,G);B._elevation=A.append(B._elevation,E)
		elif K==Q and J==R:B._elevation=A.append(E,B._elevation);B._width=A.append(G,B._width)
		elif K==O and J==S:
			P=(B._width<=G)&(B._elevation<E)
			if A.any(P):
				B._elevation=B._elevation[P];B._width=B._width[P]
				if not B.__v_close(B._elevation[-1],E)or not B.__v_close(B._width[-1],G):
					B._elevation=A.sort(A.append(B._elevation,E))
					if G>B._width[0]:B._width=A.append(B._width,G)
					else:B._width=A.append(B._width,B._width[-1])
			else:raise M(V)
		elif K==Q and J==S:
			P=(B._width>=G)&(B._elevation>E)
			if A.any(P):
				B._elevation=B._elevation[P];B._width=B._width[P]
				if not B.__v_close(B._elevation[0],E)or not B.__v_close(B._width[0],G):
					B._elevation=A.sort(A.append(E,B._elevation))
					if G<B._width[0]:B._width=A.append(G,B._width)
					else:B._width=A.append(B._width[0],B._width)
			else:raise M('')
		elif K==O and J==W:B._width[-1]=G;B._elevation[-1]=E
		elif K==Q and J==W:B._width[0]=G;B._elevation[0]=E
		B.__init_helper(check_nan=F,sort=F,check_increasing=D)
	def compute(B):B._wet_a_array,B._wet_per_array=l(B._width,B._elevation,check_nan=F,sort=F,check_increasing=F);B._wet_a,B._wet_per=B._wet_a_array[-1],B._wet_per_array[-1];C=E.deepcopy(B._wet_per_array);D=max(B.__float_atol,B.min_width);C=A.maximum(C,D);B._hydro_rad_array=B._wet_a_array/C;B._hydro_rad=B._hydro_rad_array[-1];B._debitance_array=B._wet_a_array*B._hydro_rad_array**(2./3.);B._debitance=B._debitance_array[-1]
	def plot(B,show=D,plot_title='',fig_width=15,fig_height=5,y_axis_title='elevation (m)',x_axis_title='width (m)',y_lim=C,x_max_lim=C,output_file=C):
		D=x_max_lim;H=E.deepcopy(-B._width[::-1]/2);I=E.deepcopy(B._width/2);J=A.append(H,I);K=E.deepcopy(B._elevation[::-1]);L=E.deepcopy(B._elevation);M=A.append(K,L)
		if D is not C:G=-D/2,D/2
		else:G=C
		v(xs=[J],ys=[M],show=show,title=plot_title,x_axis_title=x_axis_title,y_axis_title=y_axis_title,fig_width=fig_width,fig_height=fig_height,y_lim=y_lim,x_lim=G,add_legend=F,output_file=output_file)
	def combine_with_other(I,other,self_z_up_bound_array=C,self_z_low_bound_array=C,self_w_up_bound_array=C,self_w_low_bound_array=C,other_z_up_bound_array=C,other_z_low_bound_array=C,other_w_up_bound_array=C,other_w_low_bound_array=C,preference=o):
		A4='w_up_other';A3='w_low_other';A2='z_up_other';A1='z_low_other';A0='w_up_self';z='w_low_self';y='z_up_self';x='z_low_self';p='self';o='other';n=other_w_low_bound_array;m=other_w_up_bound_array;l=other_z_low_bound_array;k=other_z_up_bound_array;j=self_w_low_bound_array;i=self_w_up_bound_array;g=self_z_low_bound_array;d=self_z_up_bound_array;c='w_other';b='w_self';P=other;g,d=K(ref_array=I.elevation,value0_low_bound_array=g,value0_up_bound_array=d);j,i=K(ref_array=I.width,value0_low_bound_array=j,value0_up_bound_array=i);l,k=K(ref_array=P.elevation,value0_low_bound_array=l,value0_up_bound_array=k);n,m=K(ref_array=P.width,value0_low_bound_array=n,value0_up_bound_array=m);s=A.round(E.deepcopy(I.elevation)*10**I.__n_sig_dig,0).astype(A.int64);t=A.round(E.deepcopy(P.elevation)*10**I.__n_sig_dig,0).astype(A.int64);u=A.concatenate([E.deepcopy(s),E.deepcopy(t)]);J=A.full_like(u,fill_value=A.nan,dtype=P.width.dtype);B=q.DataFrame({X:E.deepcopy(u),b:J,c:J,x:J,y:J,z:J,A0:J,A1:J,A2:J,A3:J,A4:J});B.drop_duplicates(subset=X,inplace=D,ignore_index=D);B.sort_values(X,inplace=D);B.set_index(X,drop=D,inplace=D);Q=B.index[A.isin(B.index.to_numpy(),s).nonzero()[0]];R=B.index[A.isin(B.index.to_numpy(),t).nonzero()[0]];B.loc[Q,b]=E.deepcopy(I.width);B.loc[Q,x]=E.deepcopy(g);B.loc[Q,y]=E.deepcopy(d);B.loc[Q,z]=E.deepcopy(j);B.loc[Q,A0]=E.deepcopy(i);B.loc[R,c]=E.deepcopy(P.width);B.loc[R,A1]=E.deepcopy(l);B.loc[R,A2]=E.deepcopy(k);B.loc[R,A3]=E.deepcopy(n);B.loc[R,A4]=E.deepcopy(m)
		for N in B.columns:
			if A.any(A.isfinite(B[N].to_numpy())):B[N]=r(values_in_array=B[N].to_numpy(),base_in_array=B.index.to_numpy(),limits=C,check_nan=F,float_atol=I.__float_atol)
			elif'up'in N:B[N]=A.inf
			else:B[N]=-A.inf
		O,W,Y,Z,a=[],[],[],[],[]
		for G in e(B.shape[0]):
			if A.isnan(B.iloc[G][b]):H=o
			elif A.isnan(B.iloc[G][c]):H=p
			elif'wi'in preference:
				if B.iloc[G][b]>=B.iloc[G][c]:H=p
				else:H=o
			elif B.iloc[G][b]<=B.iloc[G][c]:H=p
			else:H=o
			if G==0:O.append(B.iloc[G][f"w_{H}"]);W.append(B.iloc[G][f"z_low_{H}"]);Y.append(B.iloc[G][f"z_up_{H}"]);a.append(B.iloc[G][f"w_low_{H}"]);Z.append(B.iloc[G][f"w_up_{H}"])
			elif B.iloc[G][f"w_{H}"]>=O[-1]:O.append(B.iloc[G][f"w_{H}"]);W.append(B.iloc[G][f"z_low_{H}"]);Y.append(B.iloc[G][f"z_up_{H}"]);a.append(B.iloc[G][f"w_low_{H}"]);Z.append(B.iloc[G][f"w_up_{H}"])
			else:O.append(O[-1]);W.append(W[-1]);Y.append(Y[-1]);a.append(a[-1]);Z.append(Z[-1])
		B.index=B.index.astype(I.elevation.dtype);B.index=B.index*10**-I.__n_sig_dig;I._elevation=B.index.to_numpy(dtype=I.elevation.dtype);I._width=A.array(O,dtype=I.width.dtype)
		try:I.__init_helper(check_nan=F,sort=F,check_increasing=D)
		except L as v:
			w=f(v.args[0])
			if h in w:raise M(f"")
			else:raise v
		return{S:W,T:Y,U:a,V:Z}
	def add_wet_bathy_from_other(B,other,self_z_up_bound_array=C,self_z_low_bound_array=C,self_w_up_bound_array=C,self_w_low_bound_array=C,other_z_up_bound_array=C,other_z_low_bound_array=C,other_w_up_bound_array=C,other_w_low_bound_array=C):
		c=other_w_low_bound_array;b=other_w_up_bound_array;a=other_z_low_bound_array;Z=other_z_up_bound_array;Y=self_w_low_bound_array;X=self_w_up_bound_array;O=self_z_low_bound_array;N=self_z_up_bound_array;C=other;O,N=K(ref_array=B.elevation,value0_low_bound_array=O,value0_up_bound_array=N);Y,X=K(ref_array=B.width,value0_low_bound_array=Y,value0_up_bound_array=X);a,Z=K(ref_array=C.elevation,value0_low_bound_array=a,value0_up_bound_array=Z);c,b=K(ref_array=C.width,value0_low_bound_array=c,value0_up_bound_array=b)
		if B.min_elevation<C.min_elevation:return{S:O,T:N,U:Y,V:X}
		P,G=[],[];Q,W,H,I=[],[],[],[]
		for E in e(C.elevation.size):
			if C.elevation[E]>=B.min_elevation:break
			if C.width[E]>=B.min_width:break
			P.append(C.elevation[E]);G.append(C.width[E]);Q.append(a[E]);W.append(Z[E]);H.append(c[E]);I.append(b[E])
		if len(P)==0:return{S:O,T:N,U:Y,V:X}
		d=C.get_width_by_elevation(B.min_elevation);g=B.min_width
		if d<g:P.append(B.min_elevation);G.append(d)
		else:P.append(B.min_elevation);G.append(g)
		Q.append(-A.inf);W.append(A.inf);H.append(-A.inf);I.append(A.inf);J=(B._width>G[-1]).nonzero()[0]
		if J.size>0:Q+=R(O[J]);W+=R(N[J]);H+=R(Y[J]);I+=R(X[J]);B._elevation=A.append(A.array(P),B._elevation[J]);B._width=A.append(A.array(G),B._width[J])
		else:Q=[*Q,O[-1]];W=[*W,N[-1]];H=[*H,H[-1]];I=[*I,I[-1]];B._elevation=A.append(A.array(P),B._elevation[-1]);B._width=A.append(A.array(G),G[-1])
		try:B.__init_helper(check_nan=F,sort=F,check_increasing=D)
		except L as i:
			j=f(i.args[0])
			if h in j:raise M(f"")
			else:raise i
		return{S:A.array(Q,dtype=B._elevation.dtype),T:A.array(W,dtype=B._elevation.dtype),U:A.array(H,dtype=B._elevation.dtype),V:A.array(I,dtype=B._elevation.dtype)}
	def add_dry_bathy_from_other(B,other,self_z_up_bound_array=C,self_z_low_bound_array=C,self_w_up_bound_array=C,self_w_low_bound_array=C,other_z_up_bound_array=C,other_z_low_bound_array=C,other_w_up_bound_array=C,other_w_low_bound_array=C):
		c=other_w_low_bound_array;b=other_w_up_bound_array;a=other_z_low_bound_array;Z=other_z_up_bound_array;O=self_w_low_bound_array;N=self_w_up_bound_array;J=self_z_low_bound_array;I=self_z_up_bound_array;C=other;J,I=K(ref_array=B.elevation,value0_low_bound_array=J,value0_up_bound_array=I);O,N=K(ref_array=B.width,value0_low_bound_array=O,value0_up_bound_array=N);a,Z=K(ref_array=C.elevation,value0_low_bound_array=a,value0_up_bound_array=Z);c,b=K(ref_array=C.width,value0_low_bound_array=c,value0_up_bound_array=b)
		if B.max_elevation>C.max_elevation:return{S:J,T:I,U:O,V:N}
		P,E=[],[];Q,W,X,Y=[],[],[],[]
		for G in e(C.elevation.size):
			d=C.elevation[G];g=C.width[G]
			if d<=B.max_elevation:continue
			if g<=B.max_width:continue
			P.append(d);E.append(g);Q.append(a[G]);W.append(Z[G]);X.append(c[G]);Y.append(b[G])
		if len(P)==0:return{S:J,T:I,U:O,V:N}
		i=C.get_width_by_elevation(B.max_elevation);j=B.max_width;P=[B.max_elevation]+P
		if i>j:E=[i]+E
		else:E=[j]+E
		Q.append(-A.inf);W.append(A.inf);X.append(-A.inf);Y.append(A.inf);H=B._width<E[0];Q=R(J[H])+Q;W=R(I[H])+W;X=R(O[H])+X;Y=R(N[H])+Y;B._elevation=A.append(B._elevation[H],P);B._width=A.append(B._width[H],E)
		try:B.__init_helper(check_nan=F,sort=F,check_increasing=D)
		except L as k:
			l=f(k.args[0])
			if h in l:raise M(f"")
			else:raise k
		return{S:A.array(Q,dtype=B._elevation.dtype),T:A.array(W,dtype=B._elevation.dtype),U:A.array(X,dtype=B._elevation.dtype),V:A.array(Y,dtype=B._elevation.dtype)}
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
		raise L(f"")
	def shrink_top_by_elevation(A,value):
		B=value
		if A.min_elevation<B<A.max_elevation:A.modify_by_elevation(B,loc_modif=O);return
		if A.__v_close(B,A.max_elevation):return
		if A.__v_close(B,A.min_elevation):B=A.min_elevation;A.modify_by_elevation(B,loc_modif=O);return
		raise N(f"")
	def shrink_top_by_width(A,value):
		B=value
		if A.min_width<B<A.max_width:A.modify_by_width(B,loc_modif=O);return
		if A.__v_close(B,A.max_width):return
		if A.__v_close(B,A.min_width):B=A.min_width;A.modify_by_width(B,loc_modif=O);return
		else:raise N(f"")
	def shrink_bottom_by_elevation(A,value):
		B=value
		if A.min_elevation<B<A.max_elevation:A.modify_by_elevation(B,loc_modif=Q);return
		if A.__v_close(B,A.min_elevation):return
		if A.__v_close(B,A.max_elevation):B=A.max_elevation;A.modify_by_elevation(B,loc_modif=Q);return
		raise N(f"")
	def shrink_bottom_by_width(A,value):
		B=value
		if A.min_width<B<A.max_width:A.modify_by_width(B,loc_modif=Q);return
		if A.__v_close(B,A.min_width):return
		if A.__v_close(B,A.max_width):B=A.max_elevation;A.modify_by_elevation(B,loc_modif=Q);return
		else:raise N(f"")
	def get_width_by_elevation(A,value):return A.__get_var1_by_w_z(var0_value=value,var0_ref=I,var1_ref=H)
	def get_elevation_by_width(A,value):return A.__get_var1_by_w_z(var0_value=value,var0_ref=H,var1_ref=I)
	def get_depth_by_width(A,value):B=A.get_elevation_by_width(value);return B-A._min_elevation
	def get_depth_by_elevation(B,value):
		C=value;D=B.get_width_by_elevation(C);E=C-B._min_elevation
		if D>B.min_width:return E
		return A.nan
	def get_area_by_elevation(A,value):return A.__get_var1_by_w_z(var0_value=value,var0_ref=I,var1_ref=Y)
	def get_area_by_width(A,value):return A.__get_var1_by_w_z(var0_value=value,var0_ref=H,var1_ref=Y)
	def get_perimeter_by_elevation(A,value):return A.__get_var1_by_w_z(var0_value=value,var0_ref=I,var1_ref=g)
	def get_perimeter_by_width(A,value):return A.__get_var1_by_w_z(var0_value=value,var0_ref=H,var1_ref=g)
	def get_hydraulic_radius_by_elevation(A,value):return A.__get_var1_by_w_z(var0_value=value,var0_ref=I,var1_ref=p)
	def get_hydraulic_radius_by_width(A,value):return A.__get_var1_by_w_z(var0_value=value,var0_ref=H,var1_ref=p)
	def get_debitance_by_elevation(A,value):return A.__get_var1_by_w_z(var0_value=value,var0_ref=I,var1_ref=s)
	def get_debitance_by_width(A,value):return A.__get_var1_by_w_z(var0_value=value,var0_ref=H,var1_ref=s)
	def __eq__(B,other):
		C=other
		if C.elevation.size!=B.elevation.size:return F
		if A.allclose(C.elevation,B.elevation,rtol=J,atol=B.__float_atol):
			if A.allclose(C.width,B.width,rtol=J,atol=B.__float_atol):return D
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
	@G
	def width(self):return E.deepcopy(self._width)
	@G
	def elevation(self):return E.deepcopy(self._elevation)
	@G
	def wetted_perimeter(self):
		A=self
		if A._wet_per is C:A.compute()
		return E.deepcopy(A._wet_per_array)
	@G
	def wetted_area(self):
		A=self
		if A._wet_a is C:A.compute()
		return E.deepcopy(A._wet_a_array)
	@G
	def hydraulic_radius(self):
		A=self
		if A._hydro_rad is C:A.compute()
		return E.deepcopy(A._hydro_rad_array)
	@G
	def debitance(self):
		A=self
		if A._debitance is C:A.compute()
		return E.deepcopy(A._debitance_array)
	@G
	def depth(self):return E.deepcopy(self._depth)
	@G
	def min_width(self):
		B=self
		if B._min_width is C:B._min_width=A.min(B._width)
		return B._min_width
	@G
	def max_width(self):
		B=self
		if B._max_width is C:B._max_width=A.max(B._width)
		return B._max_width
	@G
	def min_elevation(self):
		B=self
		if B._min_elevation is C:B._min_elevation=A.min(B.elevation)
		return B._min_elevation
	@G
	def max_elevation(self):
		B=self
		if B._max_elevation is C:B._max_elevation=A.max(B.elevation)
		return B._max_elevation
	@G
	def max_depth(self):return A.max(self.depth)
	@G
	def max_wetted_perimeter(self):
		A=self
		if A._wet_per is C:A.compute()
		return A._wet_per
	@G
	def max_wetted_area(self):
		A=self
		if A._wet_a is C:A.compute()
		return A._wet_a
	@G
	def max_hydraulic_radius(self):
		A=self
		if A._hydro_rad is C:A.compute()
		return A._hydro_rad

