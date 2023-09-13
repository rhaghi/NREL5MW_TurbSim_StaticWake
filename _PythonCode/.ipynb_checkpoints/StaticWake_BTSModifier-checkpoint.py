import os

import numpy
import time
import pandas as pd
import numpy as np
import h5py
import chaospy as cp
import glob

import time as tm

from scipy import signal
import datetime
import sys
from pyFAST.input_output import FASTOutputFile
from pyFAST.input_output import TurbSimFile
from pyFAST.input_output import FASTInputFile
from scipy.interpolate import interp2d
from scipy.stats import hmean
import random

###################

def Sobol_Samp_u_a_TI(nos):
    a_LB = 0.15
    a_UB = 0.22
    D = 126
    R = D/2
    U_max = 25
    z = 90
    U = cp.Uniform(lower = 3, upper=25)
    TI = cp.Uniform(lower = 0.04, upper=0.18*(0.75+5.6/U))
    alpha_lb = a_LB-0.23*(U_max/U)*(1-(0.4*np.log10(R/z))**2)
    alpha_ub = a_UB + 0.4*(R/z)*(U_max/U)
    alpha = cp.Uniform(lower =alpha_lb , upper=alpha_ub)
    joint_dist1 = cp.J(U,alpha)
    joint_dist2 = cp.J(U,TI)
    joint_dist1_T = cp.Trunc(joint_dist1,lower=-0.3,upper=30)
    s1 = cp.generate_samples(nos, joint_dist1_T,rule='S')
    samp1 = s1[:,np.argsort(s1[0,:])]
    s2 = cp.generate_samples(nos, joint_dist2,rule='S')
    samp2 = s2[:,np.argsort(s2[0,:])]
    '''Stack alpha and TI together. In the main sampling array, the first row is the
    wind speed, the second row is the shear (alpha) and the third row is TI 
    '''
    samp = np.vstack((samp1,samp2[1,:]))
    return samp

def sigma_D(x,D,I_a,Ct):
    #From Eq 28 of [1]:
    k_star= 0.11*(Ct**1.07)*(I_a**0.2)
    epsilon = 0.23*(Ct**(-0.25))*(I_a**0.2)
    #From Eq 24 of [1]:
    sD = k_star*x/D + epsilon
    return sD

def vel_def(Ct,sigma_D):
    #From Eq 5 of [2]
    vd = (1-np.sqrt(1-Ct/(8*(sigma_D**2))))
    return vd

def get_phi(y,y_wake,z,z_hub,u_0,alpha,vel_def,sigma_D,D):
    #From Eq 4 of [2]
    u=u_0*(z/z_hub)**alpha
    sigma = sigma_D*D
    gauss_power = (-1/(sigma**2))*((z-z_hub)**2+(y-y_wake)**2)
    guassian = vel_def*np.exp(gauss_power)
    phi_yz = u*(1-guassian)
    return phi_yz

def delta_I(x,y,z,HH,D,I_a,Ct,sigma_D):
    # From Eq 35 of [1] and formulas in Table 2
    sgm = sigma_D*D
    pi = np.pi
    r = np.sqrt(y**2 + (z-HH)**2)
    d = 2.3*Ct**(-1.2)
    e = I_a**(0.1)
    f = 0.7*Ct**(-3.2)*I_a**(-0.45)
    k1=np.nan
    k2=np.nan
    if r/D <= 0.5:
        k1 = np.cos((pi/2)*(r/D-0.5))**2
    elif r/D > 0.5:
        k1 =1
    if r/D <= 0.5:
        k2 = np.cos((pi/2)*(r/D+0.5))**2
    elif r/D > 0.5:
        k2 =0
    if z>=HH:
        drk = 0
    elif z<HH:
        drk = I_a*np.sin(pi*(HH-z)/HH)
        #drk = 0
    frc1 = 1/(d+e*x/D+f*(1+x/D)**(-2))
    frc2 = -((r-D/2)**2)/(2*sgm**2)
    frc3 = -((r+D/2)**2)/(2*sgm**2)
    d_I1 = frc1*(k1*np.exp(frc2) + k2*np.exp(frc3)) - drk
    return d_I1


BTSFileName = sys.argv[1]

BTS_data = TurbSimFile(BTSFileName)

TurbSim_org_folder = "TurbSim_Org"

if  os.path.exists('../'+TurbSim_org_folder) == False:
    os.mkdir('../'+TurbSim_org_folder)

TurbSimFile.write(BTS_data,'../'+TurbSim_org_folder+'/'+BTSFileName[11:])

y = BTS_data['y']
z = BTS_data['z']
BTS_Seed = int(BTSFileName[BTSFileName.find('.bts')-5:BTSFileName.find('.bts')])

# making mesh using y,z for plotting and averaging purposes
Y,Z= np.meshgrid(y,z)

NREL_5MW_Ct = pd.read_csv('../_PythonCode/NREL_Ct.csv')
NREL_5MW_Ct=NREL_5MW_Ct.drop(labels=0,axis=0)
NREL_5MW_Ct = NREL_5MW_Ct.astype(float)

D = 126     #Rotor diameter [m]
x = 7*D     #Distance down the first turbine
y_wake = float(sys.argv[2])  #The center of the wake y
z_hub = 90  #Hub height
mod_sobol = pd.read_csv('../_PythonCode/mod_sobol.csv',index_col=0)


u_0 = BTS_data['uRef']

if u_0 < 3:
    u_0=3

Ct = NREL_5MW_Ct.loc[NREL_5MW_Ct['WindVxi']==np.round(u_0)]['Ct'].to_numpy()

u_0 = BTS_data['uRef']

alpha = mod_sobol.loc[mod_sobol['Seed Number'] == BTS_Seed]['alpha'].to_numpy()
I_a = mod_sobol.loc[mod_sobol['Seed Number'] == BTS_Seed]['TI'].to_numpy()

# The wake width based on the linear wake propogation
S_ovr_D = sigma_D(x=x,D=D,I_a=I_a,Ct=Ct)
# The maximum velocity defici
vd = vel_def(Ct,S_ovr_D)
phi = np.zeros((15,15))
for i,y_i in zip(range(0,15),y):
    for j,z_i in zip(range(0,15),z):
        phi[j,i] = get_phi(y_i,y_wake,z_i,z_hub,u_0,alpha,vel_def=vd,sigma_D=S_ovr_D,D=D)
        

u = BTS_data['u'][0,:,:,:]
u_bar = np.mean(u,axis=0)
u_tilda = u - u_bar
u_wake = u_tilda+phi
BTS_data['u'][0:,:,:]=u_wake
TurbSimFile.write(BTS_data,BTSFileName)

TurbSim_wake_folder = "TurbSim_Wake"

if  os.path.exists('../'+TurbSim_wake_folder) == False:
    os.mkdir('../'+TurbSim_wake_folder)
    
TurbSimFile.write(BTS_data,'../'+TurbSim_wake_folder+'/'+BTSFileName[11:])