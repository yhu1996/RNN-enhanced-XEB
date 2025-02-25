# compute chi_C evolution (output numerators and denominators) & generate outcome dataset

import numpy as np
import pickle

########################################################################################

import os
import sys

sys.path.append(os.path.abspath("./classical_simulation/XEB/"))

from MIPT_circuit_exact import *


########################################################################################
# circuit params

p = 0.1
M_max = 10 # the number of measurement outcomes to be generated

# get circuit params
file = open('./classical_simulation/data/circuit_param(p=%5.2f).pickle'%p, 'rb')
data = pickle.load(file)
file.close()

L_lst = data['L_lst']
circuit_dic_lst = data['circuit_dic_lst']

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


def numerator_evol(rho_encoding, sigma_encoding, T, L, phi_rec, measurement_rec):
    p_c_lst = []
    rho_outcome = []
    for step_r in range(M_max):
        # rho-circuit, get outcome
        state_q, outcome_rec = MIPT_hybrid_circuit_bulk(rho_encoding, T, L, phi_rec, measurement_rec)
        # sigma-circuit 
        state_c = MIPT_hybrid_circuit_proj(sigma_encoding, T, L, phi_rec, measurement_rec, outcome_rec)
        # compute numerator
        p_c = get_trace(state_c, L)
        p_c_lst.append(p_c) # shape [M_max]
        rho_outcome.append(outcome_rec) # M_max outcomes

    
    return p_c_lst, rho_outcome 



def denominator_evol(sigma_encoding, T, L, phi_rec, measurement_rec):
    p_c_lst = []
    sigma_outcome = []

    for step_r in range(M_max):
        # sigma-circuit, get outcome
        state_q, outcome_rec = MIPT_hybrid_circuit_bulk(sigma_encoding, T, L, phi_rec, measurement_rec)
        # sigma-circuit 
        state_c = MIPT_hybrid_circuit_proj(sigma_encoding, T, L, phi_rec, measurement_rec, outcome_rec)
        # compute denominator
        p_c = get_trace(state_c, L)
        p_c_lst.append(p_c)
        sigma_outcome.append(outcome_rec) # M_max outcomes
    
    return p_c_lst, sigma_outcome



########################################################################################


for i in range(len(L_lst)):

    #st = time.time()
    L = L_lst[i]
    T = 2*L
    
    sigma = init_sigma(L)
    rho = init_rho(L)

   
    # circuit: encoding T + bulk T
    circuit_dic = circuit_dic_lst[i]
    measurement_rec = circuit_dic['measurement_rec']
    phi_rec_encoding = circuit_dic['phi_rec_encoding']
    phi_rec = circuit_dic['phi_rec']
    
    rho_encoding = MIPT_hybrid_circuit_encoding(rho, T, L, phi_rec_encoding)
    sigma_encoding = MIPT_hybrid_circuit_encoding(sigma, T, L, phi_rec_encoding)


    # numerator of chi_C
    numerator_lst, rho_outcome = numerator_evol(rho_encoding, sigma_encoding, T, L, phi_rec, measurement_rec)
    

    # denominator of chi_C
    denominator_lst, sigma_outcome = denominator_evol(sigma_encoding, T, L, phi_rec, measurement_rec)

    # save outcome data
    file = open('./classical_simulation/data/training_data(p=%5.2f,L=%d,M=%d).pickle'%(p,L,M_max), 'wb')
    dic = {'rho_outcome': rho_outcome, 'sigma_outcome': sigma_outcome}
    pickle.dump(dic, file)
    file.close()

    # save data
    file = open('./classical_simulation/result/histogram(L=%d,p=%5.2f,M=%d).pickle'%(L,p,M_max), 'wb')
    dic = {'numerator_lst': numerator_lst, 'denominator_lst': denominator_lst}
    pickle.dump(dic, file)
    file.close()