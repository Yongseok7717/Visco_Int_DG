#!/usr/bin/python
from __future__ import print_function
from dolfin import *
import numpy as np
import math
import getopt, sys
import matplotlib.pyplot as plt
from mpltools import annotation
import matplotlib as mpl
#mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

s1Hu=np.loadtxt("H1_error_quad_disp_S1.txt")
s1Lu=np.loadtxt("L2_error_quad_disp_S1.txt")
s1Lw=np.loadtxt("L2_error_quad_velo_S1.txt")
s1Hw=np.loadtxt("H1_error_quad_velo_S1.txt")


s2Hu=np.loadtxt("H1_error_quad_disp_S2.txt")
s2Lu=np.loadtxt("L2_error_quad_disp_S2.txt")
s2Lw=np.loadtxt("L2_error_quad_velo_S2.txt")
s2Hw=np.loadtxt("H1_error_quad_velo_S2.txt")


N,M=np.shape(s1Hu)

xh=s1Hu[0,1:M]

#print()
plt.figure(1)
line_D1=plt.plot(np.log(xh),np.log(s1Hu.diagonal()[1::]),label=r'(S1) $H^1$ norm of $e$',ls='-.')
line_D2=plt.plot(np.log(xh),np.log(s1Lu.diagonal()[1::]),label=r'(S1) $L_2$ norm of $ e$',ls='--')
line_D3=plt.plot(np.log(xh),np.log(s1Hw.diagonal()[1::]),label=r'(S1) $H^1$ norm of $\tilde{e}$',ls=':')
line_D4=plt.plot(np.log(xh),np.log(s1Lw.diagonal()[1::]),label=r'(S1) $L_2$ norm of $\tilde{e}$')

annotation.slope_marker((4.8,-14),-2,invert=True)

line_V1=plt.plot(np.log(xh),np.log(s2Hu.diagonal()[1::]),label=r'(S2) $H^1$ norm of $ e$',marker='o',ls='-.')
line_V2=plt.plot(np.log(xh),np.log(s2Lu.diagonal()[1::]),label=r'(S2) $L_2$ norm of $ e$',marker='D',ls='--')
line_V3=plt.plot(np.log(xh),np.log(s2Hw.diagonal()[1::]),label=r'(S2) $H^1$ norm of $\tilde{ e}$',marker='p',ls=':')
line_V4=plt.plot(np.log(xh),np.log(s2Lw.diagonal()[1::]),label=r'(S2) $L_2$ norm of $\tilde{ e}$',marker='s')

plt.xlabel(r'$\log(1/h)$')
plt.legend(loc='best')

plt.ylabel('log(error)')
plt.title('Numerical results of quadratic basis')
plt.savefig('graph_quad.png')
