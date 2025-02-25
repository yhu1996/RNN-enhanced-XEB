# simulate M_C random circuits and compute the cross entropy chi_C for each circuit realizations by sampling M measurements


import numpy as np
import pickle

########################################################################################

import os
import sys

sys.path.append(os.path.abspath("./classical_simulation/XEB/"))

from MIPT_circuit_exact import *


########################################################################################
# circuit params

L = 8
p = 0.2
M_C = 10 # number of random circuits
M_max = 10 # number of measurement shots

########################################################################################

# sigma = |000000...0>
def init_sigma(L):
    sigma = [0]*(2**L - 1)
    sigma.append(1)
    sigma = np.array(sigma, dtype = np.complex128)
    return sigma


# rho = |+>^L ~ (|0> + |1>)^L
def init_rho(L):
    plus_state = 1/np.sqrt(2)*np.array([1,1])
    rho = 1/np.sqrt(2)*np.array([1,1])
    for i in range(L-1):
        rho = np.kron(rho, plus_state)
    rho = np.array(rho, dtype = np.complex128)
    return rho


def numerator_simul(rho_encoding, sigma_encoding, T, L, phi_rec, measurement_rec):
    p_c_lst = []

    for step_r in range(M_max):
        # rho-circuit, get outcome
        state_q, outcome_rec = MIPT_hybrid_circuit_bulk(rho_encoding, T, L, phi_rec, measurement_rec)
        # sigma-circuit 
        state_c = MIPT_hybrid_circuit_proj(sigma_encoding, T, L, phi_rec, measurement_rec, outcome_rec)
        # compute numerator
        p_c = get_trace(state_c, L)
        p_c_lst.append(p_c) # shape [M_max]
    
    return np.mean(p_c_lst)


def denominator_simul(sigma_encoding, T, L, phi_rec, measurement_rec):
    p_c_lst = []

    for step_r in range(M_max):
        # sigma-circuit, get outcome
        state_q, outcome_rec = MIPT_hybrid_circuit_bulk(sigma_encoding, T, L, phi_rec, measurement_rec)
        # sigma-circuit 
        state_c = MIPT_hybrid_circuit_proj(sigma_encoding, T, L, phi_rec, measurement_rec, outcome_rec)
        # compute denominator
        p_c = get_trace(state_c, L)
        p_c_lst.append(p_c)
    
    return np.mean(p_c_lst)



########################################################################################
# simulation 

T = 2*L
    
sigma = init_sigma(L)
rho = init_rho(L)
    
chiC_lst = []
circuit_dic_lst = []

for step_C in range(M_C):

    # circuit: encoding T + bulk T
    phi_rec_encoding = generate_circuit_encoding(L, T)
    phi_rec, measurement_rec = generate_circuit_rand(L, T, p)
    rho_encoding = MIPT_hybrid_circuit_encoding(rho, T, L, phi_rec_encoding)
    sigma_encoding = MIPT_hybrid_circuit_encoding(sigma, T, L, phi_rec_encoding)

    circuit_dic = {'phi_rec_encoding': phi_rec_encoding, 'phi_rec':phi_rec, 'measurement_rec':measurement_rec}
    circuit_dic_lst.append(circuit_dic) # M_C dicts

    # numerator of chi_C
    numerator = numerator_simul(rho_encoding, sigma_encoding, T, L, phi_rec, measurement_rec)

    # denominator of chi_C
    denominator = denominator_simul(sigma_encoding, T, L, phi_rec, measurement_rec)

    # chi_C 
    chiC_lst.append(numerator/denominator) # shape [M_C]


# save data: one can save the circuit parameters and chiC for each circuit
# circuit parameters:
file = open('./classical_simulation/result/circuit_param(p=%5.2f,L=%d,MC=%d,M=%d).pickle'%(p,L,M_C,M_max), 'wb')
dic = {'circuit_dic_lst': circuit_dic_lst}
pickle.dump(dic, file)
file.close()

# chiC for each circuit:
file = open('./classical_simulation/result/chi_C(p=%5.2f,L=%d,MC=%d,M=%d).pickle'%(p,L,M_C,M_max), 'wb')
dic = {'chiC_lst': chiC_lst}
pickle.dump(dic, file)
file.close()