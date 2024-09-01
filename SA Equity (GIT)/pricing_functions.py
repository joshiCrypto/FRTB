#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 11:24:39 2024

@author: joshuakaji
"""
import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt 

def black_scholes_call(S, sigma, K, T, r):
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate the call option price using the Black-Scholes formula
    call_price = (S * si.norm.cdf(d1, 0.0, 1.0)) - (K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    
    return call_price

def vega_call(S, sigma, K, T, r):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    N_prim_d1 = np.exp(-d1**2 / 2) / np.sqrt(2* np.pi)
    vega = S * np.sqrt(T) * N_prim_d1
    return vega 

def delta_call(S, sigma, K, T, r):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    delta = si.norm.cdf(d1, 0.0, 1.0)
    return delta 

def sensi_by_finite_difference(V, x, shock=0.01, shock_type='relative'):
    if shock_type == 'relative':
        x_shocked = x * (1 + shock)
        sensi = (V(x_shocked) - V(x))/ (x*shock)
    if shock_type == 'absolute':
        x_shocked = x + shock
        sensi = (V(x_shocked) - V(x))/ shock
    return sensi 

def eqt_future(S0, r, cy, T): # conveniance yield includes repo rate 
    return S0 * np.exp((r - c)*T)



# Example parameters
S = 80     # Current stock price
K = 100     # Strike price
T = 1/12       # Time to maturity in years
r = 0.05    # Risk-free interest rate
sigma = 0.3 # Volatility

S = np.linspace(70, 130, 100)


########################################################
## Vega analytical VS finite difference approximation
# Vega - analytical 
vega_analytical = vega_call(S, sigma, K, T, r)
# Vega - finite difference approximation
params = [(lambda x : black_scholes_call(S, x, K, T, r)), sigma, 0.01, 'absolute']
vega_by_shock = sensi_by_finite_difference(*params)

plt.plot(S, vega_by_shock)
plt.plot(S, vega_analytical)
plt.show()

# error of approximation (in BPs)
plt.plot(S, (vega_by_shock - vega_analytical)/vega_analytical*1000)
plt.show()
######################################################## 
## Delta analytical VS finite difference approximation
# Delta - analytical 
delta_analytical = delta_call(S, sigma, K, T, r)
# Delta - finite difference approximation
params = [(lambda x : black_scholes_call(x, sigma, K, T, r)), S, 0.01, 'relative']
delta_by_shock = sensi_by_finite_difference(*params)

plt.plot(S, delta_by_shock)
plt.plot(S, delta_analytical)
plt.show()

# error of approximation (in BPs)
plt.plot(S, (delta_by_shock - delta_analytical)/delta_analytical*1000)
plt.show()




