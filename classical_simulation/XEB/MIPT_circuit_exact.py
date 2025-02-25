import numpy as np
import random
from ncon import *


# trace from a pure state
def get_trace(state, L):
    state = state.reshape(2**(L))
    trace = np.dot(np.conjugate(state),state)
    return trace.real


########################
### Random Unitaries ###
########################


def MS_gate(theta):
	MS = np.array([[np.cos(theta),0,0,-np.sin(theta)*1j],[0,np.cos(theta),-np.sin(theta)*1j,0],[0,-np.sin(theta)*1j,np.cos(theta),0],[-np.sin(theta)*1j,0,0,np.cos(theta)]])
	return MS



def Rotation_gate(theta, phi):
	R = np.array([[np.cos(theta), -1j*np.sin(theta)*np.exp(-1j*phi)],[-1j*np.sin(theta)*np.exp(1j*phi),np.cos(theta)]])
	return R



##################################
### Build Circuits: each layer ###
##################################


# Impletment random single-qubit rotation at each time step given the phi's
def insert_rotation_layer(state, L, phi_lst):
	theta = np.pi/4
	for spin in range(1,L+1):
		phi = phi_lst[spin-1]
		R = Rotation_gate(theta, phi)
		state = state.reshape(2**(spin-1), 2, 2**(L-spin))
		state = ncon([R,state],[[-2,1],[-1,1,-3]])
    
	return state



def MS_even(state, L):
	theta = np.pi/4
	unitary_op = MS_gate(theta)

	for i in range(L//2-1): 
		idx = 1 + 2*i
		state = state.reshape([2**(idx-1), 4, 2**(L-idx-1)])
		state = ncon([unitary_op,state],[[-2,1],[-1,1,-3]])

	# the last block
	state = state.reshape([2**(L-2),4])
	state = np.matmul(state, unitary_op.transpose())
	
	return state



def MS_odd(state, L):
	theta = np.pi/4
	unitary_op = MS_gate(theta)

	for i in range(L//2-1):
		state = state.reshape([2**(1+2*i), 4, 2**(L-3-2*i)])
		state = ncon([unitary_op,state],[[-2,1],[-1,1,-3]])

	# the last block (if periodic bndy condition)
	#state = state.reshape([2,2**(L-2),2])
	#unitary_op = unitary_op.reshape([2,2,2,2])
	#state = ncon([unitary_op,state],[[-1,-3,1,2],[1,-2,2]])

	return state


def insert_measurement_layer(state, L, measurement_lst):
	
	outcome_lst = []
	for spin in range(1, L+1):
		if measurement_lst[spin-1] == 1:
			state, outcome = insert_measurement_site(state, spin, L)
			outcome_lst.append(outcome)

	return state, outcome_lst



def insert_proj_layer(state, L, measurement_lst, outcome_lst):
	idx = 0
	for spin in range(1, L+1):
		if measurement_lst[spin-1] == 1:
			state = insert_proj_site(state, spin, L, outcome_lst[idx])
			idx += 1

	return state



#################################
### Build Circuits: each site ###
#################################



# do measurement on a single qubit, record measurement outcome: plus->1; minus->0
def insert_measurement_site(state, spin, L):

	# compute Pplus and Pminus, namely the expectation value of sigma_axis(spin) 	
	random_num = random.uniform(0, 1)
	proj_plus = np.array([[1,0],[0,0]])
	proj_minus = np.array([[0,0],[0,1]])

	state = state.reshape(2**(spin-1), 2, 2**(L-spin))
	proj_plus_state = ncon([proj_plus,state],[[-2,1],[-1,1,-3]])
	proj_minus_state = ncon([proj_minus,state],[[-2,1],[-1,1,-3]])
	
	state = state.reshape(2**(L))
	proj_plus_state = proj_plus_state.reshape(2**(L))
	Pplus = np.dot(np.conjugate(state),proj_plus_state)

	if random_num < Pplus: 
		return proj_plus_state/np.sqrt(Pplus), 1
	else:
		proj_minus_state = proj_minus_state.reshape(2**(L))
		return proj_minus_state/np.sqrt(1-Pplus), 0



def insert_proj_site(state, spin, L, outcome):
	proj_plus = np.array([[1,0],[0,0]])
	proj_minus = np.array([[0,0],[0,1]])
	state = state.reshape(2**(spin-1), 2, 2**(L-spin))

	if outcome == 0:
		proj_minus_state = ncon([proj_minus,state],[[-2,1],[-1,1,-3]])
		return proj_minus_state
	
	if outcome == 1:
		proj_plus_state = ncon([proj_plus,state],[[-2,1],[-1,1,-3]])
		return proj_plus_state




################################
### Build Circuits: assemble ###
################################



def generate_circuit_rand(L, T, p):
	
	measurement_rec = np.zeros((T,L), dtype=int)
	phi_rec = np.zeros((T,L))
	phi_lst = [0, np.pi/2, np.pi/4]
	
	for time in range(T):
		for spin in range(L):
			# measurement yes = 1, no = 0
			r = random.uniform(0, 1)
			if r < p:
				measurement_rec[time,spin] = 1
			# phi_i
			phi_rec[time,spin] = random.choice(phi_lst)
	return phi_rec, measurement_rec



def generate_circuit_encoding(L, T):
	
	phi_rec = np.zeros((T,L))
	phi_lst = [0, np.pi/2, np.pi/4]
	
	for time in range(T):
		for spin in range(L):
			# phi_i
			phi_rec[time,spin] = random.choice(phi_lst)
	return phi_rec


# bulk = with measurements
def MIPT_hybrid_circuit_bulk(initial_state, T, L, phi_rec, measurement_rec):

	state = initial_state
	outcome_rec = []

	for time in range(T):
		
		# implement random single-qubit rotation
		phi_lst = phi_rec[time]
		state = insert_rotation_layer(state, L, phi_lst)

		# implement MS transforms 
		if time%2 == 0:
			state = MS_even(state, L)
		else:
			state = MS_odd(state, L)
		
		# insert measurements
		measurement_lst = measurement_rec[time]
		state, outcome_lst = insert_measurement_layer(state, L, measurement_lst)
		outcome_rec.append(outcome_lst)

#	state = state.reshape(2**(L))
	
	return state, outcome_rec




# proj = with projections
def MIPT_hybrid_circuit_proj(initial_state, T, L, phi_rec, measurement_rec, outcome_rec):

	state = initial_state

	for time in range(T):
		
		# implement random single-qubit rotation
		phi_lst = phi_rec[time]
		state = insert_rotation_layer(state, L, phi_lst)

		# implement MS transforms 
		if time%2 == 0:
			state = MS_even(state, L)
		else:
			state = MS_odd(state, L)
		
		# insert projections
		measurement_lst = measurement_rec[time]
		outcome_lst = outcome_rec[time]
		state = insert_proj_layer(state, L, measurement_lst, outcome_lst)

#	state = state.reshape(2**(L))
	
	return state


# encoding = only random unitaries
def MIPT_hybrid_circuit_encoding(initial_state, T, L, phi_rec):

	state = initial_state

	for time in range(T):
		
		# implement random single-qubit rotation
		phi_lst = phi_rec[time]
		state = insert_rotation_layer(state, L, phi_lst)

		# implement MS transforms 
		if time%2 == 0:
			state = MS_even(state, L)
		else:
			state = MS_odd(state, L)

#	state = state.reshape(2**(L))
	
	return state