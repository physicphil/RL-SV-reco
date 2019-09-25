# imports
import sys
import time

import numpy as np
from numpy import linalg as LA
#import ROOT
from array import array
import matplotlib
matplotlib.use('Agg')
import pylab
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

#ROOT.gROOT.ProcessLine(".L Objects.h+")
    
#from ROOT import PFCandidateType

import Intersection_finder_absoluteCoordinates_Module as I
import VertexObject as VO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# set up training as testing data
'''
mepz_opts = ['hasTHETA_CHI2', 'hasTHETA_CHI2_dlCut', 'noMEPZ', 'hasEPZ', 'hasMEPZ']
mepz = 'hasTHETA_CHI2'
poca = "pocas"
scal = "scal"
pTcut = 1.0
ptc = "pTcut:" + str(pTcut) + "GeV"
#could be for loops
iserr = 'noErr'
'''
# load data in torch format
X_data = np.load('PFC_data_pocas.npy')
y_data = np.load('PV_true_pocas.npy')

print(np.transpose(y_data[0]))


#['pt', 'eta', 'phi', 'charge', 'dxy', 'dz', 'pvIndex', 'pdgId']#, 'POCA_z', 'chi2', 'ndof', 'pdgId']
#  0       1    2       3         4      5     6            7

# apply cuts on data (pT > 1, pdgId != 0)
for i in range(len(X_data)): # event loop
    for j in range(len(X_data[i])):
        if X_data[i,j,0] < 1 or X_data[i,j,6] != 0 or X_data[i,j,7] == 0:
            X_data[i,j] = np.zeros(len(X_data[i,j]))

maxp = 0
minp = 100
for j in range (X_data.shape[0]):
    nump = 0
    for k in range(X_data.shape[1]):
        if (X_data[j,k,0] != 0): #this means we have an actual particle, not zero padding
            nump += 1
            #convert pdgId to theta using theta = 2*atan(exp(-eta))
            X_data[j,k,-1] = 2 * np.arctan(np.exp(-1*X_data[j,k,1]))
        if (X_data[j,k,0] < 0 or (X_data[j,k,0] < 1 and X_data[j,k,0] != 0)):
            print(j, k, "We have a pT problem!")
    if (nump > maxp): 
        maxp = nump
    if (nump < minp):
        minp = nump
    #order in pT so the all-zero particles will always be the ones lost when i cull
    X_data[j] = X_data[j, np.flip((X_data[j,:,0].argsort()), 0)]

print(maxp)
X_data = X_data[:,:maxp,:]
print(X_data.shape)
# assign training and testing data
idx = np.random.choice(X_data.shape[0], X_data.shape[0],replace=False)
print(idx)
X_data = X_data[idx]
y_data = y_data[idx]
print(X_data.shape)
X_train = X_data[:10000]
y_train = y_data[:10000]

X_test = X_data[10000:13000]
y_test = y_data[10000:13000]

X_val = X_data[13000:]
y_val = y_data[13000:]


class DQN(nn.Module):
    """DQN class with h input nodes and output output nodes"""
    def __init__(self, h, outputs):
        super(DQN, self).__init__()
        self.fcn1 = nn.Linear(h,512)
        self.fcn2 = nn.Linear(512,512)
        self.fcn3 = nn.Linear(100,50)
        '''
        self.fcn4 = nn.Linear(50,10)
        self.fcn6 = nn.Linear(10,10)
        self.fcn5 = nn.Linear(10,outputs)
        '''

    def forward(self, x):
        x = F.relu(self.fcn1(x))
        x = F.relu(self.fcn2(x))
        x = self.fcn3(x)
        '''
        x = F.relu(self.fcn4(x))
        x = F.relu(self.fcn6(x))
        x = F.relu(self.fcn6(x))
        x = self.fcn5(x)
        '''
        return x
        
def select_action_DQN(state):
    """Selects an action either based on policy or randomly"""
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # return the index of the max in output tensor
            return int(policy_net(torch.tensor(state)).argmax())
    else:
        # return random bool
        random_actions.append(i_episode)
        return np.random.choice(n_actions, 1)

def init_weights(m):
    """Inits weights of m by random for linear layers"""
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

# maybe change reward such that it can be calculated after the experience
# this way training can be made faster by keeping memories
# num steps is a bad thing as it is not         
def optimise_model_memory(batch_size):
    """
    This function performs one training step on policy net.
    
    State action value Q(s,a) is compared to r + Q_t(s',a')
    and a step of the optimiser is taken.
    """
    x_batch, y_batch = [], []
    action_batch = []
    batch_size = min(len(memory), batch_size)
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        y_target = reward if done else reward + GAMMA \
                              * (target_net(torch.ones((1,1))*next_state).max())
        action_batch.append(int(action))
        x_batch.append(state)
        y_batch.append(y_target)
        
    x_batch = torch.tensor(x_batch).type(torch.FloatTensor).reshape((batch_size,1))
    y_batch = torch.tensor(y_batch).type(torch.FloatTensor).reshape((batch_size,1))
    action_batch = torch.tensor(action_batch).type(torch.LongTensor).reshape((batch_size,1))
    optimizer.zero_grad()
    #out = policy_net(x_batch).type(torch.FloatTensor).max()
    out = policy_net(x_batch).type(torch.FloatTensor).gather(1, action_batch)
    loss = F.smooth_l1_loss(out, y_batch)
    #loss = F.mse_loss(out, y_batch)
    loss.backward()
    # clip error values to values between -1 and 1
    #for param in policy_net.parameters():
     #   param.grad.data.clamp_(-1, 1)
    optimizer.step()

print(X_data.shape)
n_actions = 1 + 2 * X_data.shape[1]
n_inputs = X_data.shape[1] * (X_data.shape[2] + 1)
GAMMA = 0.999
EPS_START = 1
EPS_END = 0.00
EPS_DECAY = 50000
steps_done = 0

TARGET_UPDATE = 30
num_episodes = 70

memory = []


policy_net = DQN(290, n_actions).to(device)
target_net = DQN(290, n_actions).to(device)
init_weights(policy_net)
optimizer = optim.Adam(policy_net.parameters(),lr=0.001)

# loop over training data
counter = 0
for evnt in X_train:
    Env = VO.TrackEnvironment(torch.tensor(evnt))
    if counter < 10:
        print(Env.state)
    counter += 1
    

# how to loop over jets in X_train
# build reward function
