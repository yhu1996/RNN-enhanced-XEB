# compute the cross entropy by enumerating outcomes / sampling outcomes

import numpy as np
import torch
from torch.nn.functional import one_hot

########################################################################################

def ten_to_two_string(n, k):
    # output: a binary string of n with k digits
    bit = ''
    while n//2 >= 1 or n%2 >=1:
        bit = bit + str(n%2)
        n = n//2
    while len(bit) < k:
        bit = bit + str(0)
    return bit[::-1]

def ten_to_two_list(n, k):
    # output: a list 
    bit = ten_to_two_string(n, k)
    lst = []
    for i in range(k):
        if bit[i] == '0':
            lst.append(0)
        if bit[i] == '1':
            lst.append(1)
    return lst

def change_shape(one_d_lst, shape):
    
    idx = 0
    output_lst = []
    for layer in range(len(shape)):
        layer_lst = shape[layer]
        output_layer = []
        for i in range(len(layer_lst)):
            output_layer.append(one_d_lst[idx])
            idx += 1
        output_lst.append(output_layer)
    return output_lst


########################################################################################

# all possible outcome_rec 
def all_outcome(N):

    outcome_rec_lst = []
    for i in range(2**N):
        outcome_rec_lst.append(ten_to_two_list(i, N))
    outcome_rec_lst = np.array(outcome_rec_lst)

    outcome_rec_lst = torch.from_numpy(outcome_rec_lst)
    outcome_rec_lst = one_hot(outcome_rec_lst.long(), 2)

    return outcome_rec_lst

########################################################################################

def probability(config_lst, model): # config_lst is Torch tensor with shape [N_sample, N_spin, 2]

    config_lst, y = model.eval().forward(config_lst)
    y = np.exp(y.detach().numpy())
    config_lst = config_lst.detach().numpy()
    
    prob_lst = []
    for i in range(config_lst.shape[0]):
        prob_config = 1
        for n in range(config_lst.shape[1]):
            prob_config *= np.dot(y[i, n], config_lst[i, n])
        prob_lst.append(prob_config)
    return np.array(prob_lst)


def chi_C_from_ndarray(P_rho, P_sigma): # inputs are ndarray

    numerator = np.sum(P_rho * P_sigma)
    denominator = np.sum (P_sigma * P_sigma)

    return numerator/denominator


def chi_C_from_model(config_lst, model_rho, model_sigma):

    P_rho = probability(config_lst, model_rho)
    P_sigma = probability(config_lst, model_sigma)
    
    return chi_C_from_ndarray(P_rho, P_sigma)



########################################################################################
def chi_C_from_model_sampling(model_rho, model_sigma, N_sample): 
    # estimate chiC using M samples
    
    s = model_sigma.sample(N_sample)
    r = model_rho.sample(N_sample)

    numerator = np.mean(probability(r, model_sigma))
    denominator = np.mean(probability(s, model_sigma))

    return numerator/denominator