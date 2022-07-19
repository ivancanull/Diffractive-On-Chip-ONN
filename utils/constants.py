"""
    This file defines important paramenters.
"""

import numpy as np

# changable general parameters

Lambda0 = 1.55e-6
n_sub = 2.78
TM02_phi_coupling = 0.58
TM02_phi_decoupling = 0.35 # -0.5 or -1?
TM02_eta_decoupling = 1.1
TM02_n_eff = 2.8
w0 = 0.7e-6
w_Si = 0.5e-6
k_GBM = 0.8
k_RSM = 0.5
eta_norm_coupling = 1
delta = 0

x0_bound = 0.55e-6 / Lambda0
# fixed parameters

# infered parameters
Lambda = Lambda0 / n_sub
k0 = 2 * np.pi / Lambda0
k_sub = 2 * np.pi * n_sub / Lambda0
TM02_beta = TM02_n_eff * k0
TM02_k_eff = 2 * np.pi * TM02_n_eff / Lambda0

F_coupling_coeff = eta_norm_coupling * np.exp(1j * TM02_phi_coupling) \
                   * np.sqrt(w_Si / w0 / np.sqrt(np.pi))

# kind of useless parameters

Coupling_TM02 = 1e-3 # coupling efficinecy for TM02


# not in use anymore

# R_E0 = 0.58  # define reflectance
# R_phi = 0 #phase shift for reflectance
# TM04_phi_coupling = -5.7
# TM04_phi_decoupling = 0.35
# TM04_n_eff = 1.54
# TM04_k_eff = 2 * np.pi * TM04_n_eff / Lambda0


