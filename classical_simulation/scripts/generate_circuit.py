# generate a circuit for a list of L's and p
# check the total number of measurements

import numpy as np
import pickle

########################################################################################

import os
import sys

sys.path.append(os.path.abspath("./classical_simulation/XEB/"))

from MIPT_circuit_exact import *


########################################################################################
# circuit params

L_lst = [8, 12, 16]
p = 0.1 
circuit_dic_lst = []


########################################################################################
# circuit: encoding T + bulk T
for L in L_lst:
    
    T = 2*L

    phi_rec_encoding = generate_circuit_encoding(L, T)
    phi_rec, measurement_rec = generate_circuit_rand(L, T, p)

    circuit_dic = {'phi_rec_encoding': phi_rec_encoding, 'phi_rec':phi_rec, 'measurement_rec':measurement_rec}
    circuit_dic_lst.append(circuit_dic)

    N = 0
    for time in range(T):
        for spin in range(L):
            # measurement: yes = 1, no = 0
            if measurement_rec[time,spin] == 1:
                N += 1
    print(L,N) 


# save data
file = open('./classical_simulation/data/circuit_param(p=%5.2f).pickle'%p, 'wb')
dic = {'L_lst':L_lst, 'circuit_dic_lst': circuit_dic_lst}
pickle.dump(dic, file)
file.close()