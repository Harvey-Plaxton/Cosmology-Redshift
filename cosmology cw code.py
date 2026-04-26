#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 09:55:32 2023

@author: bking
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import random
import math
from sklearn.linear_model import LinearRegression

df = pd.read_csv('./sn-data-CW.csv');



plt.figure(1);

plt.plot(df["Redshift"],df["Distance Modulus"],'.');
plt.xlabel("Redshift z")
plt.ylabel("Distance modulus")
plt.show()

def luminosity_dist(x):
    return (10**((x/5) + 1))*(10**(-6))

c = 300000

def yval(x):
    return (luminosity_dist(x)*100)/c

distance_modulus = df["Distance Modulus"].tolist()

y_values = [yval(x) for x in distance_modulus]


x_vals = df["Redshift"].tolist()

plt.figure()
plt.plot(x_vals,y_values,'.')
plt.xlabel("Redshift z")
plt.ylabel("d/l")
plt.show()

zmin = 0.05
zmax = 0.2

redshift_limited = df[df["Redshift"].between(0.05,0.2)]["Redshift"].tolist()
distance_modulus_limited = df[df["Redshift"].between(0.05,0.2)]["Distance Modulus"].tolist()


luminosity_over_l = [yval(x) for x in distance_modulus_limited]

plt.plot(redshift_limited,luminosity_over_l,'.')
plt.xlabel("Redshift (0.05<z<0.2)")
plt.ylabel("d/l")
plt.show()

x = np.array(redshift_limited).reshape((-1,1))
y = np.array(luminosity_over_l)

model = LinearRegression().fit(x, y)
c_1 = float(model.coef_)
c_0 = float(model.intercept_)
print(f'Slope: {c_1}')
print(f'Intercept: {c_0}')


def h(z):
    return z/(c_0+c_1*z)
    
zs = np.linspace(0.05,0.2,30)
hs = [h(i) for i in zs]
plt.plot(zs,hs)
plt.xlabel("Redshift (0.05<z<0.2)")
plt.ylabel("Hubble parameter h")
plt.show()

redshift_limited_c = df[df["Redshift"].between(0.1,0.7)]["Redshift"].tolist()
distance_modulus_limited_c = df[df["Redshift"].between(0.1,0.7)]["Distance Modulus"].tolist()

h_c = 0.7

def l(x,z):
    m = yval(x)
    return h_c*m*(1/z)

lhs = []
for i in range(len(redshift_limited_c)):
    mu = distance_modulus_limited_c[i]
    re = redshift_limited_c[i]
    lval = l(mu,re)
    lhs.append(lval)
print(len(lhs) == len(redshift_limited_c))

plt.plot(redshift_limited_c, lhs, '.')
plt.xlabel("Redshift (0.1<z<0.7)")
plt.ylabel("hd/zl")
plt.show()

xc = np.array(redshift_limited_c).reshape(-1,1)
yc = np.array(lhs)

model_c = LinearRegression().fit(xc,yc)
c_1_c = float(model_c.coef_)
c_0_c = float(model_c.intercept_)
print(f"intercept: {c_0_c}")
print(f"slope: {c_1_c}")

def q(z):
    return 1 - 2*(c_0_c - 1 + c_1_c*z)/z

zc = np.linspace(0.1,0.7,60)
qc = [q(i) for i in zc]
plt.plot(zc,qc)
plt.ylabel("q")
plt.xlabel("Redshift (0.1<z<0.7)")
plt.show()
        










