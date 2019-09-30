# imports
import sys
import time

import numpy as np
from numpy import linalg as LA
import math
import random

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

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
print("Device: ", device)


np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# set up training as testing data

# load data in torch format
X_data = np.load('PFC_data_pocas.npy')
y_data = np.load('PV_true_pocas.npy')

#['pt', 'eta', 'phi', 'charge', 'dxy', 'dz', 'pvIndex', 'pdgId']#, 'chi2', pvx, pvy, pvz needed!!!
#  0       1    2       3         4      5     6            7
print(y_data.shape)
y_data = np.repeat(y_data, X_data.shape[1]).reshape((y_data.shape[0], X_data.shape[1],3))
print(y_data.shape)
print(X_data.shape)
X_data = np.concatenate((X_data, y_data), axis=2)

# apply cuts on data (pT > 1, pdgId != 0)
for i in range(len(X_data)): # event loop
    for j in range(len(X_data[i])):
        if X_data[i,j,0] < 1 or X_data[i,j,6] != 0 or X_data[i,j,7] == 0 or X_data[i,j,3] == 0:
            X_data[i,j] = np.zeros(len(X_data[i,j]))

maxp = 0
minp = 100
for j in range (X_data.shape[0]):
    nump = 0
    for k in range(X_data.shape[1]):
        if (X_data[j,k,0] != 0): #this means we have an actual particle, not zero padding
            nump += 1
            #convert pdgId to theta using theta = 2*atan(exp(-eta))
            X_data[j,k,7] = 2 * np.arctan(np.exp(-1*X_data[j,k,1]))
        if (X_data[j,k,0] < 0 or (X_data[j,k,0] < 1 and X_data[j,k,0] != 0)):
            print(j, k, "We have a pT problem!")
    if (nump > maxp): 
        maxp = nump
    if (nump < minp):
        minp = nump
    #order in pT so the all-zero particles will always be the ones lost when i cull
    X_data[j] = X_data[j, np.flip((X_data[j,:,0].argsort()), 0)]

print("Max number of valid tracks", maxp)
#X_data = X_data[:,:maxp,:]
num_pfc_cut = 7 # maxp
X_data = X_data[:,:num_pfc_cut,:]
print(X_data.shape)

# assign training and testing data
# randomize data
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
        self.fcn3 = nn.Linear(512,outputs)
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
    a = 0
    if sample > eps_threshold:
        with torch.no_grad():
            # return the index of the max in output tensor
            state = state.to(device)
            a = int(policy_net(state.flatten()).argmax())
    else:
        # return random bool
        a = int(np.random.choice(n_actions, 1))
    #print("Action selected:", a)
    return a

def init_weights(m):
    """Inits weights of m by random for linear layers"""
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.00)

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
        x_batch.append(state.flatten())
        state = state.to(device)
        next_state = next_state.to(device)
        y_target = reward if done else reward + GAMMA \
                              * float((target_net(next_state.flatten()).max()).to('cpu'))
        action_batch.append(int(action))
        y_batch.append(y_target)
        #.type(torch.FloatTensor)
    x_batch = torch.cat(x_batch).reshape((batch_size, n_inputs)).to(device)
    y_batch = torch.tensor(y_batch).type(torch.FloatTensor).squeeze().to(device)
    action_batch = torch.tensor(action_batch).reshape((batch_size,1)).to(device)
    optimizer.zero_grad()
    out = policy_net(x_batch).reshape((batch_size, n_actions))
    out = out.gather(1, action_batch).squeeze()
    #loss = F.smooth_l1_loss(out, y_batch)
    loss = F.mse_loss(out, y_batch)
    loss.backward()
    # clip error values to values between -1 and 1
    #for param in policy_net.parameters():
     #   param.grad.data.clamp_(-1, 1)
    optimizer.step()

print(X_train.shape)
n_actions = 1 + 2 * X_data.shape[1]
print("# actions: ", n_actions)

n_inputs = X_data.shape[1] * (X_data.shape[2] + 1)
print("# inputs", n_inputs)


memory = []

policy_net = DQN(n_inputs, n_actions).to(device)
target_net = DQN(n_inputs, n_actions).to(device)
init_weights(policy_net)
optimizer = optim.Adam(policy_net.parameters(),lr=0.01)


GAMMA = 0.8
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 5000
steps_done = 0

MINI_BATCH = 100
TARGET_UPDATE = 20
epochs = 3
# num_episodes = 1000

# loop over training data
counter = 0
steps_done = 0

final_poca_dist = []
ntracks_used = []
episode_lengths = []

av_test_ntracks_used = []
av_test_poca_dist = []

rewards = []

# print("Start plotting")
# fig, ax = I.helices_plot(X_train[1], [1,2], barrel=True)
# fig.savefig("helix0.png")
# plt.close()

print("Start training")

for i_epoch in range(epochs):
    # go through training data once per episode
    for i in range(X_train.shape[0]):
        if X_train[i, 0, 0] == 0:
            continue
        counter += 1
        print("Currently at event: ", )
        Env = VO.TrackEnvironment(X_train[i])
        state = Env.state
        if len(memory) > 10000:
            #l = len(memory)
            #idx = np.random.choice(l, l,replace=False)
            #memory = memory[idx]
            memory = memory[:5000]
        if counter < 100:
            print(Env.state)
            print(Env.vertex.track_indices)
            print(Env.vertex.track_indices)
            print(Env.vertex.x)
            
        evnt_counter = 0    
        for t in count():
            steps_done += 1
            evnt_counter += 1
            print("Steps done: ", steps_done)
            err_prev = 10
            true_PV = np.array([y_train[i,0], y_train[i,1], y_train[i,2]])
            if type(Env.vertex.x) == np.ndarray:
                err_prev = LA.norm(Env.vertex.x-true_PV)
            action = select_action_DQN(state)
            next_state, vertex_x, uncertainty, n, dflag, pflag = Env.take_action(action)
            print(Env.vertex.track_indices)
            #print(Env.state)
            # set up reward, if a vertex can be computed, set it to change in displacement
            reward = -500 # if no vertex, this should be positive
            if type(vertex_x) == np.ndarray:
                reward = err_prev - LA.norm(vertex_x-true_PV) \
                         + 50 *len(Env.vertex.track_indices)
            if dflag:
                reward += 10
            if pflag:
                reward -= 10000
            rewards.append(reward)
            print(f"Reward for action {action}: ", reward)
            memory.append((state, action, reward, next_state, dflag))
            state = next_state
            optimise_model_memory(MINI_BATCH)
            if steps_done%TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
            if dflag or evnt_counter > 5000:
                if i < 10:
                    fig, ax = I.helices_plot(X_test[i],
                                 Env.vertex.track_indices,
                                 pocas=Env.vertex.pocas, barrel=True)
                    fig.savefig(f"helix_train_{i}.png")
                    plt.close()
                if type(Env.vertex.x) == np.ndarray:
                    final_poca_dist.append(LA.norm(Env.vertex.x-true_PV))
                #else:
                    #final_poca_dist.append(1000000)
                ntracks_used.append(len(Env.vertex.track_indices))
                episode_lengths.append(t+1)
                if dflag:
                    print("Episode ended naturally")
                else:
                    print("Episode ended forcefully")
                break
                
    # test performance on test data once per episode (no random actions!)
    test_ntracks_used = []
    test_poca_dist = []
    print("Started testing")
    for i in range(X_test.shape[0]):
        Env = VO.TrackEnvironment(X_test[i])
        state = Env.state
        test_counter = 0
        for t in count():
            steps_done += 1
            test_counter += 1
            state = state.flatten().to(device)
            action = int(policy_net(state).argmax())
            next_state, vertex_x, uncertainty, n, dflag, pflag = Env.take_action(action)
            true_PV = np.array([y_test[i,0], y_test[i,1], y_test[i,2]])
            if dflag or test_counter > 1000:
                if i < 10:
                    fig, ax = I.helices_plot(X_test[i],
                                 Env.vertex.track_indices,
                                 pocas=Env.vertex.pocas, barrel=True)
                    fig.savefig(f"helix_test_{i}.png")
                    plt.close()

                if type(Env.vertex.x) == np.ndarray:
                    test_poca_dist.append(LA.norm(Env.vertex.x-true_PV))
                #else:
                    #test_poca_dist.append(1000000)
                test_ntracks_used.append(len(Env.vertex.track_indices))
                episode_lengths.append(t+1)
                if dflag:
                    print("Episode ended naturally")
                else:
                    print("Episode ended forcefully")
                break
        plt.hist(test_ntracks_used)
        plt.xlabel("Num tracks used")
        plt.ylabel("Events")
        plt.title("Test sample")
        plt.savefig(f"RL_oldtree_test_ntracks_epi{i_epoch}.png")
        plt.close()
        
        plt.hist(test_poca_dist)
        plt.xlabel("Vertexing error")
        plt.ylabel("Events")
        plt.title("Test sample")
        plt.yscale('log')
        plt.savefig(f"RL_oldtree_test_displacement_epi{i_epoch}.png")
        plt.close()

    av_test_ntracks_used.append(np.mean(test_ntracks_used))
    av_test_poca_dist.append(np.mean(test_poca_dist))

print("Training ended")
plt.hist(final_poca_dist)
plt.xlabel("Vertexiong error")
plt.ylabel("Attempts")
#plt.title("Vertexing error")
plt.yscale('log')
plt.savefig("RL_oldtree_poca_displacement.png")
plt.close()

plt.scatter(range(len(final_poca_dist)), final_poca_dist)
plt.xlabel("Attempt")
plt.ylabel("Vertexing error")
#plt.title("Vertexing error")
plt.yscale('log')
plt.savefig("RL_oldtree_poca_displacement.png")
plt.close()


plt.hist(ntracks_used)
#plt.title("Number of tracks used")
plt.xlabel("Number of tracks used")
plt.ylabel("Attempts")
plt.savefig("RL_oldtree_ntracks.png")
plt.close()

plt.plot(episode_lengths)
#plt.title("Actions in an episode")
plt.xlabel("Episode")
plt.ylabel("Actions")
plt.savefig("RL_oldtree_epilength.png")
plt.close()

plt.plot(av_test_ntracks_used)
plt.xlabel("Epoch")
plt.ylabel(r"# tracks used")
plt.savefig("RL_oldtree_test_ntracks.png")
plt.close()

plt.plot(av_test_poca_dist)
plt.xlabel("Epoch")
plt.ylabel("Av vertex displacement")
plt.savefig("RL_oldtree_test_displacement.png")
plt.close()

plt.scatter(range(len(rewards)), rewards)
plt.xlabel("Step")
plt.ylabel("Reward")
plt.savefig("RL_oldtree_test_rewards.png")
plt.close()

print(counter)
print(len(memory))

