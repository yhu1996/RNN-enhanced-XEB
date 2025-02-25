# test the training 

import os
import sys

sys.path.append(os.path.abspath("./vanilla_rnn"))

########################################################################################

import numpy as np
import pickle

import torch
from torch.optim import lr_scheduler
from torch.optim import Adam

import time


########################################################################################

import paseq as psq
from paseq.nn import RNNSpinModel

from XEntropy import *

########################################################################################

# params
L = 8
N = 12
p = 0.1

# retrieve arguments from command line
M = int(sys.argv[1]) 
batch_size = int(sys.argv[2])
val_ratio = float(sys.argv[3])
N_h = int(sys.argv[4])
N_epochs = int(sys.argv[5])
dropout_flag = sys.argv[6].lower() == "true"
dropout_rate = float(sys.argv[7])

# if N is large, uncomment the following line
#N_sample = int(sys.argv[8]) 

lr = 1e-3
log_every = 10

all_outcome_rec = all_outcome(N)

########################################################################################


def train_2models(model_rho, model_sigma, train_loader_rho, val_loader_rho, train_loader_sigma, val_loader_sigma):

    optimizer_rho = Adam(model_rho.parameters(), lr=lr)
    optimizer_sigma = Adam(model_sigma.parameters(), lr=lr)
    schedule_params = {"factor": 1}

    schedule_rho = lr_scheduler.ConstantLR(optimizer_rho, **schedule_params)  
    schedule_sigma = lr_scheduler.ConstantLR(optimizer_sigma, **schedule_params)

    running_loss_lst_rho = []
    val_loss_lst_rho = []
    running_loss_lst_sigma = []
    val_loss_lst_sigma = []
    epoch_lst = []
    chiC_lst = []

    for epoch in range(N_epochs):
        

        optimizer_rho.zero_grad()
        running_loss_rho = 0.0 

        optimizer_sigma.zero_grad()
        running_loss_sigma = 0.0
        
        for i, databatch in enumerate(train_loader_rho): 
            
            (xbatch,) = databatch 

            batch_loss_rho = model_rho.train().dataloss(xbatch)
            batch_loss_rho.backward() 

            running_loss_rho += batch_loss_rho

        optimizer_rho.step() 
        running_loss_rho /= i + 1  
        schedule_rho.step() 

        for i, databatch in enumerate(train_loader_sigma): 
            
            (xbatch,) = databatch 

            batch_loss_sigma = model_sigma.train().dataloss(xbatch)
            batch_loss_sigma.backward() 

            running_loss_sigma += batch_loss_sigma

        optimizer_sigma.step() 
        running_loss_sigma /= i + 1  
        schedule_sigma.step() 


        if epoch % log_every == 0 or epoch == N_epochs - 1: 
            
            val_loss_rho = 0
            for i, databatch in enumerate(val_loader_rho):
                (xbatch,) = databatch

                val_loss_rho += model_rho.eval().dataloss(xbatch) 
            val_loss_rho /= i + 1 

            val_loss_sigma = 0
            for i, databatch in enumerate(val_loader_sigma):
                (xbatch,) = databatch

                val_loss_sigma += model_sigma.eval().dataloss(xbatch) 
            val_loss_sigma /= i + 1 

            # compute chi_C
            # if N is small, compute via exact computation
            chiC = chi_C_from_model(all_outcome_rec, model_rho, model_sigma)
            # if N is large, compute via sampling outcomes
            # chiC = chi_C_from_model_sampling(model_rho, model_sigma, N_sample)
            chiC_lst.append(chiC)

            # record data
            epoch_lst.append(epoch+1)
            running_loss_lst_rho.append(running_loss_rho.detach().numpy().item())
            val_loss_lst_rho.append(val_loss_rho.detach().numpy().item())
            running_loss_lst_sigma.append(running_loss_sigma.detach().numpy().item())
            val_loss_lst_sigma.append(val_loss_sigma.detach().numpy().item())


        ########################################################################################

        print(
            "{:<100}".format(
                "\r" 
                + "[{:<60}] ".format(
                    "=" * ((np.floor((epoch + 1) / N_epochs * 60)).astype(int) - 1) + ">"
                    if epoch + 1 < N_epochs
                    else "=" * 60
                ) # progress bar
                + "{:<40}".format(
                    "Epoch {}/{}: Loss(Train) = {:.3f}, Loss(Val) = {:.3f}, chiC = {:.3f}".format(
                        epoch + 1, N_epochs, running_loss_rho, val_loss_rho, chiC
                    )
                )
            ),
            end="", 
        )
        sys.stdout.flush() 

    
    # save data 
    if dropout_flag:
        file = open('./vanilla_rnn/plot_data/train_result(N=%d,M=%d,Nh=%d,dropout%.2f,bs=%d,vr=%.2f).pickle'%(N_epochs,M,N_h,dropout_rate,batch_size,val_ratio), 'wb')
    else:
        file = open('./vanilla_rnn/plot_data/train_result(no_dropout)(N=%d,M=%d,Nh=%d,bs=%d,vr=%.2f).pickle'%(N_epochs,M,N_h,batch_size,val_ratio), 'wb')
    dic = {'epoch_lst':epoch_lst,
           'running_loss_lst_rho': running_loss_lst_rho,
           'val_loss_lst_rho': val_loss_lst_rho,
           'running_loss_lst_sigma': running_loss_lst_sigma,
           'val_loss_lst_sigma': val_loss_lst_sigma,
           'chiC_lst': chiC_lst}
    pickle.dump(dic, file)
    file.close()

    return model_rho, model_sigma

########################################################################################

# load data 
datafolder = "./training_data"


# sigma 
datapath = os.path.join(
        datafolder, "sigma_outcome_p%.2f(M=60000).txt"%p
    )
train_loader_sigma, val_loader_sigma = psq.data.load_data_cut(
        datapath, M, batch_size=batch_size, val_ratio=val_ratio
    )
model_sigma = RNNSpinModel(N=N, N_h=N_h, dropout_flag = dropout_flag, dropout_rate = dropout_rate)


# rho
datapath = os.path.join(
        datafolder, "rho_outcome_p%.2f(M=60000).txt"%p
    )
train_loader_rho, val_loader_rho = psq.data.load_data_cut(
        datapath, M, batch_size=batch_size, val_ratio=val_ratio
    )
model_rho = RNNSpinModel(N=N, N_h=N_h, dropout_flag = dropout_flag, dropout_rate = dropout_rate)


model_rho, model_sigma = train_2models(model_rho, model_sigma, train_loader_rho, val_loader_rho, train_loader_sigma, val_loader_sigma)
