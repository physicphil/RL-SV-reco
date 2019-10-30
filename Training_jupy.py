#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# imports
import sys
import time
from IPython.core.debugger import set_trace

import numpy as np
from numpy import linalg as LA
import math
import random

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from itertools import count

import torch#
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import Intersection_finder_absoluteCoordinates_Module as I
import VertexObject as VO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# In[ ]:


class DQN(nn.Module):
    """DQN class with h input nodes and output output nodes"""
    def __init__(self, n_inputs, n_parameters, outputs):
        super(DQN, self).__init__()
        self.n_inputs = n_inputs
        self.cn1 = nn.Conv2d(1, 32, (1,n_parameters))
        self.cn2 = nn.Conv2d(32, 32, (1,1))
        self.cn3 = nn.Conv2d(32, 16, (1,1))
        self.cn4 = nn.Conv2d(16, 8, (1,1))
        self.fcn1 = nn.Linear(self.n_inputs * 8,256)
        self.fcn4 = nn.Linear(256,128)
        self.fcn6 = nn.Linear(128,64)
        self.fcn5 = nn.Linear(64,outputs)

    def forward(self, x):
        x = F.relu(self.cn1(x))
        x = F.relu(self.cn2(x))
        x = F.relu(self.cn3(x))
        x = self.cn4(x)
        x = x.view((-1, 8*self.n_inputs))
        x = F.relu(self.fcn1(x))
        x = F.relu(self.fcn4(x))
        x = F.relu(self.fcn6(x))
        x = self.fcn5(x)
        return x


def select_action_DQN(state):
    """Selects an action either based on policy or randomly"""
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    a = 0
    if sample > eps_threshold:
        with torch.no_grad():
            # return the index of the max in output tensor
            state = state.to(device)
            a = (policy_net(state).argsort()).cpu()[0].numpy()
            a = np.flip(a)
    else:
        # return random bool
        a = np.random.choice(n_actions, n_actions,replace=False)
    #print("Action selected:", a)
    return [int(i) for i in a]

def init_weights(m):
    """Inits weights of m by random for linear layers"""
    if type(m) == nn.Linear:
        torch.nn.init.zeros_(m.weight)
        m.bias.data.fill_(0.00)

# maybe change reward such that it can be calculated after the experience
# this way training can be made faster by keeping experiences
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
        x_batch.append(state)
        next_state = next_state.to(device)
        y_target = reward if done else reward + gamma * float((target_net(next_state).max()).to('cpu'))
        action_batch.append(int(action))
        y_batch.append(y_target)
        #.type(torch.FloatTensor)
    with torch.no_grad():
        x_batch = torch.cat(x_batch).to(device)
        y_batch = torch.tensor(y_batch).type(torch.FloatTensor).to(device)
        action_batch = torch.tensor(action_batch).reshape((batch_size,1)).to(device)
    optimizer.zero_grad()
    out = policy_net(x_batch).reshape((batch_size, n_actions))
    out = out.gather(1, action_batch).squeeze()
    loss = F.smooth_l1_loss(out, y_batch)
    #loss = F.mse_loss(out, y_batch)
    loss.backward()
    # clip error values to values between -1 and 1
    #for param in policy_net.parameters():
     #   param.grad.data.clamp_(-1, 1)
    optimizer.step()



np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# set up training as testing data

# load data in torch format
X_data = np.load('PFC_data_pocas_wrtJ_sign_cut.npy')
y_data = np.load('SV_true_pocas_wrtJ_sign_cut.npy')
print("Data loaded")
np.load = np_load_old

# PFCatts = 'pt', 'eta', 'phi', 'charge', 'dxy', 'dz', 'pv_x', 'pv_y', 'pv_z',
# 
#             0     1     2        3          4    5    6         7       8 
#             
#            'theta', 'chi2', 'normalizedChi2','ndof', 'nPixelHits', 'deltaEta',
#            
#               9        10            11         12       13            14
#               
#            'deltaPhi', 'jetPtFrac', 
#            
#                 15            16 
#                 
#            'ptError', 'etaError', 'phiError',  'dxyError', 'dzError'
#            
#                17          18            19         20         21
# 

# In[ ]:


num_pfc_cut = 10 # maxp
# do not use the most complicated parameters
X_data = X_data[:,:num_pfc_cut,:17] # do not include error for track params
print(f"Number of jets: {X_data.shape}")


print(f"Number of jets: {X_data.shape}")
# split data to training, testing, validation data
X_train = X_data[:25000]
y_train = y_data[:25000]

X_test = X_data[25000:]
y_test = y_data[25000:]
if len(X_test) == 0:
    print("To much training data")

'''
X_val = X_data[32000:]
y_val = y_data[32000:]
if len(X_val) == 0:
    print("To much testing data")
'''

# build scaler, which is passed to the Vertexing Envirnoment
X_scale = X_train.reshape(X_train.shape[0]*X_train.shape[1], X_train.shape[2])
print(X_scale.shape)
print(X_scale[:5])
print(f"Before scaling: {X_scale[0]}")
remove_mask = []
for i in range(X_scale.shape[0]):
    if X_scale[i,0] == 0:
        remove_mask.append(i)

X_scale = np.delete(X_scale, np.array(remove_mask), 0)

scaler = StandardScaler()
scaler.fit(X_scale)
X_scale = scaler.transform(X_scale)
print(X_scale[:5])
print(X_scale.shape)


for i in range(X_scale.shape[1]):
    plt.hist(X_scale[:,i], bins=100)
    plt.title(f"{i}th attr of scaled tracks")
    plt.close()

#print(X_train.shape)
Env_test = VO.TrackEnvironment(X_train[0])
print(Env_test.state)
Env_test = VO.TrackEnvironment(X_train[0], scaler=scaler)
n_actions = 1 + 2 * Env_test.state.shape[2]
print(f"# actions: {n_actions}")

n_parameters =  Env_test.state.shape[3]
print(f"# inputs per track: {n_parameters}")
n_tracks = Env_test.state.shape[2]
h = Env_test.state.shape[3]
print(f"# tracks: {n_tracks}")
#for i in range(X_train[:50].shape[0]):
 #   print(X_train[i, :5, :14])
print(Env_test.state)
print(type(scaler) == StandardScaler)

memory = []

policy_net = DQN(n_tracks,n_parameters,n_actions).to(device)
target_net = DQN(n_tracks,n_parameters,n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
#target_net.load_state_dict(nn.init.zeros_(target_net.weight.size()))
optimizer = optim.Adam(policy_net.parameters(),lr=0.0001)

print("NN initialised")

gamma = 0
GAMMA = 1.2 #0.9
EPS_START = 1
EPS_END = 0.2
EPS_DECAY = 50000
steps_done = 0

MINI_BATCH = 256
TARGET_UPDATE = 20

epochs = 2
num_episodes = 10000 #number of jets to train on
num_test_episodes = 15000# number of jets to test on
max_episode_length = 35 #when to end a vertex attempt
run_test = True


#directory = './Plots_training/1013/06/'
directory = '/home/rinckeph/work/Post_write_work/02/'
save_format = 'pdf' #png
# loop over training data
steps_done = 0

episode_lengths = []
rewards = []

print("Start training")
start_time = time.time()

for i_epoch in range(epochs):

    # go through training data once per epoch
    episode_counter = 0
    train_ntracks_used = []
    train_vertex_error = []
    train_pflags = []
    train_steps = []
    for i in range(X_train.shape[0]):
        if X_train[i, 0, 0] == 0 or X_train[i, 1, 0] == 0:
            continue
        episode_counter += 1
        #print(f"Currentlyin  episode {episode_counter}")
        if episode_counter > num_episodes:
            print("Reached required number of episodes")
            break
        
        Env = VO.TrackEnvironment(X_train[i], scaler=scaler)
        true_SV = np.array([y_train[i,0], y_train[i,1], y_train[i,2]])
        print(f"Currently at event: {i} \nNumber of valid tracks: {Env.n_valid_tracks}" )
        state = Env.state
        if len(memory) > 20000:
            print("Memory update")
            #l = len(memory)
            #idx = np.random.choice(l, l,replace=False)
            #memory = memory[idx]
            memory_r = random.sample(memory, len(memory))
            memory = memory_r[:10000]
            
        pflags_till_valid = -1
        steps_till_done = 0
        skipped_stopping = False
        for t in count():
            print(f"Steps done: {steps_done}")
            #print(f"SV: {true_SV}")
            actions = select_action_DQN(Env.state)
            next_state, vertex_x, uncertainty, n, pflag, dflag = -1, -1, -1, -1, True, False
            actions_index = 0
            action = actions[actions_index] # need to store action outside of loop            
            '''
            while pflag:
                pflags_till_valid += 1
                action = actions[actions_index]
                if t > max_episode_length:
                    action = n_actions - 1
                    print("Episode will have ended forcefully")
                next_state, vertex_x, uncertainty, n, pflag, dflag = Env.take_action(action)
                actions_index += 1
            '''
            
            print(f"Attempting action: {action}")
            while pflag:
                pflags_till_valid += 1
                action = actions[actions_index]
                actions_index += 1
                if action == n_actions - 1 and not skipped_stopping:
                    print("Illegal stopping attempt!")
                    action = actions[actions_index]
                    print(f"Trying instead: {action}")
                if t > max_episode_length:
                    action = n_actions - 1
                    print("Episode will have ended forcefully")
                next_state, vertex_x, uncertainty, n, pflag, dflag = Env.take_action(action)
            skipped_stopping = True # first action was valid
            train_pflags.append(pflags_till_valid)
            steps_till_done += 1
            #print(Env.vertex.track_indices)
            #print(Env.state)
            # set up reward, if a vertex can be computed, set it to change in displacement
            reward = -10000 # if no vertex, this should be positive
            if type(vertex_x) == np.ndarray:
                reward = - np.sum((vertex_x-true_SV)**2)
                if dflag:
                    reward += 5/(-reward+0.1)
            if pflag:
                reward -= 10000

            rewards.append(reward)
            print(f"Reward for action {action}: {reward}")
            memory.append((state, action, reward, next_state, dflag))
            #if i < 10:
             #   print(Env.state)
            optimise_model_memory(MINI_BATCH)
            optimise_model_memory(MINI_BATCH)
            #print("Did model update")
            if steps_done%TARGET_UPDATE == 0 and steps_done!= 0:
                target_net.load_state_dict(policy_net.state_dict())
                gamma = GAMMA
                print("Did target update")
            if dflag:
                print("Done flag")
                train_steps.append(steps_till_done)
                high_displacement = False
                if type(vertex_x) == np.ndarray:
                    if np.sum((vertex_x-true_SV)**2) > 14000:
                        high_displacement = True
                if False:#i < 1 or high_displacement:
                    print("Trying to plot")
                    error_str = "no vertex"
                    if type(vertex_x) == np.ndarray:
                        error_str = f"{LA.norm(vertex_x-true_SV):.2f}"
                    uncer_str = "uncertainty"
                    if uncertainty != None:
                        uncer_str = f"{uncertainty:.4f}"
                    textstr = f"decay length: {y_train[i,3]:.2f}"+f"\nreward: {reward:.1f} \nerror: {error_str} \nuncertainty: {uncer_str}"
                    fig, ax = I.helices_plot(Env.track_data,
                                 Env.vertex.track_indices, textstr=textstr, 
                                 pocas=Env.vertex.pocas, barrel=True,
                                 reco_vertex=vertex_x,true_vertex=true_SV)
                    fig.savefig(f"{directory}helix_train_{i}_ep_{i_epoch}.pdf")
                    plt.close()
                    
                    fig, ax = I.helices_plot_xy(Env.track_data,
                                 Env.vertex.track_indices, textstr=textstr, 
                                 pocas=Env.vertex.pocas, barrel=True,
                                 reco_vertex=vertex_x,true_vertex=true_SV)
                    fig.savefig(f"{directory}helix_xy_train_{i}_ep_{i_epoch}.pdf")
                    plt.close()
                    print("Closed figure")
                if type(Env.vertex.x) == np.ndarray:
                    train_vertex_error.append(LA.norm(Env.vertex.x-true_SV))
                #print("Optional stuff done")
                train_ntracks_used.append(len(Env.vertex.track_indices))
                episode_lengths.append(t)
                if dflag:
                    print("Episode ended naturally")
                else:
                    print("Episode ended forcefully")
                #print("Break loop now")
                break
                
    torch.save(policy_net.state_dict(), f"{directory}model_{i_epoch}")
    
    plt.hist(train_ntracks_used, bins=range(n_actions//2))
    plt.xlabel("Num tracks used")
    plt.ylabel("Events")
    plt.title("Train sample")
    plt.savefig(f"{directory}RL_train_ntracks_epo{i_epoch}.pdf")
    plt.close()
        
    plt.hist(train_vertex_error)
    plt.xlabel("Vertexing error")
    plt.ylabel("Events")
    plt.title("Train sample")
    plt.yscale('log')
    plt.savefig(f"{directory}RL_train_displacement_epo{i_epoch}.pdf")
    plt.close()
    
    plt.hist(train_pflags, bins=range(n_actions//2))
    plt.xlabel("Number of attempts for valid action")
    plt.ylabel("Events")
    plt.title("Train sample")
    plt.savefig(f"{directory}RL_train_pflags_epo{i_epoch}.pdf")
    plt.close()
    
    plt.hist(train_steps, bins=range(max_episode_length))
    plt.xlabel("Number of steps before done")
    plt.ylabel("Events")
    plt.title("Train sample")
    plt.savefig(f"{directory}RL_train_nsteps_epo{i_epoch}.pdf")
    plt.close()   
    
    # test performance on test data once per episode (no random actions!)
    print("Should I do testing?")
    if not run_test:
        print("No :/")
        break
    print("Yes!")
    episode_counter = 0
    test_ntracks_used = []
    test_vertex_error = []
    test_decay_length = []
    test_pflags = []
    test_steps = []
    test_tracks_used = []
    test_uncertainty = []
    test_tracks_avail = []
    test_Q0s = []
    print("Started testing")
    for i in range(X_test.shape[0]):
        if episode_counter > num_test_episodes:
            print("Reached required number of test episodes")
            break
        episode_counter += 1
        Env = VO.TrackEnvironment(X_test[i], scaler=scaler)
        true_SV = np.array([y_test[i,0], y_test[i,1], y_test[i,2]])
        state = Env.state
        test_Q0s.append(policy_net(state.to(device)).detach().squeeze().cpu().numpy())
        test_counter = 0
        n_pflags = -1
        n_steps = 0
        skipped_stopping = True
        for t in count():
            steps_done += 1
            state = Env.state.to(device)
            agent_out = policy_net(state)
            actions = policy_net(state).argsort().cpu().numpy()
            actions = np.flip(actions)
            actions = [int(i) for i in actions[0]]
            if t == 0 or action == n_actions-1:
                print(f"Predicted value of all actions: {agent_out}")    
            # init return values
            next_state, vertex_x, uncertainty, n, pflag, dflag = -1, -1, -1, -1, True, False
            actions_index = 0
            action = actions[actions_index]
            while pflag:
                n_pflags += 1
                action = actions[actions_index]
                actions_index += 1
                if action == n_actions - 1 and not skipped_stopping:
                    action = actions[actions_index]
                if t > max_episode_length:
                    action = n_actions - 1
                    print("Episode will have ended forcefully")
                next_state, vertex_x, uncertainty, n, pflag, dflag = Env.take_action(action)
            skipped_stopping = True   
            test_pflags.append(n_pflags)
            n_steps += 1
            current_error = 100
            if type(vertex_x) == np.ndarray:
                current_error = LA.norm(vertex_x-true_SV)
            
            if dflag:
                print(f"It took {t+1} steps in testing")
                test_steps.append(n_steps)
                for track_ind in Env.vertex.track_indices:
                    test_tracks_used.append(track_ind)
                test_tracks_avail.append(Env.n_valid_tracks)
                test_uncertainty.append(uncertainty)
                high_uncertainty = False
                if uncertainty != None:
                    if uncertainty > 5:
                        high_uncertainty = True
                if False:#(i < 100 and current_error < 0.1) or current_error<0.01 or (current_error < 5 and high_uncertainty):
                    error_str = "no vertex"
                    if type(vertex_x) == np.ndarray:
                        error_str = f"{current_error:.2f}"
                    uncer_str = "uncertainty"
                    if uncertainty != None:
                        uncer_str = f"{uncertainty:.4f}"
                    textstr = f"decay length: {y_test[i,3]:.2f} \nerror: {error_str} \nuncertainty: {uncer_str}"
                    fig, ax = I.helices_plot(Env.track_data,
                                 Env.vertex.track_indices,
                                 textstr=textstr, 
                                 pocas=Env.vertex.pocas, barrel=True,
                                 reco_vertex=vertex_x,true_vertex=true_SV)

                    fig.savefig(f"{directory}helix_test_{i}_ep_{i_epoch}.pdf")
                    plt.close()
                    
                    fig, ax = I.helices_plot_xy(Env.track_data,
                                 Env.vertex.track_indices,
                                 textstr=textstr, 
                                 pocas=Env.vertex.pocas, barrel=True,
                                 reco_vertex=vertex_x,true_vertex=true_SV)
                    fig.savefig(f"{directory}helix_xy_test_{i}_ep_{i_epoch}.pdf")
                    plt.close()

                if type(Env.vertex.x) == np.ndarray:
                    test_vertex_error.append(current_error)
                    test_decay_length.append(y_test[i,3])
                #else:
                    #test_vertex_error.append(1000000)
                test_ntracks_used.append(len(Env.vertex.track_indices))
                if dflag:
                    print("Episode ended naturally")
                else:
                    print("Episode ended forcefully")
                break

    # save output and model
    np.save(f"{directory}test_ntracks_used_epo{i_epoch}.npy",np.array(test_ntracks_used))
    np.save(f"{directory}test_vertex_error_epo{i_epoch}.npy",np.array(test_vertex_error))
    np.save(f"{directory}test_decay_length_epo{i_epoch}.npy",np.array(test_decay_length))
    np.save(f"{directory}test_pflags_epo{i_epoch}.npy",np.array(test_pflags))
    np.save(f"{directory}test_steps_epo{i_epoch}.npy",np.array(test_steps))
    np.save(f"{directory}test_tracks_used_epo{i_epoch}.npy",np.array(test_tracks_used))
    np.save(f"{directory}test_uncertainty_epo{i_epoch}.npy",np.array(test_uncertainty))
    np.save(f"{directory}test_tracks_avail_epo{i_epoch}.npy",np.array(test_tracks_avail))
    np.save(f"{directory}test_Q0s_epo{i_epoch}.npy",np.array(test_Q0s))


    torch.save(policy_net.state_dict(), f"{directory}model_{i_epoch}")

    plt.hist(test_ntracks_used, bins=range(n_actions//2))
    plt.xlabel("Num tracks used")
    plt.ylabel("Events")
    plt.title(f"Test sample, episode: {i_epoch}")
    plt.savefig(f"{directory}RL_test_ntracks_epo{i_epoch}.pdf")
    plt.close()
        
    plt.hist(test_vertex_error)
    plt.xlabel("Vertexing error")
    plt.ylabel("Events")
    plt.title(f"Test sample, episode: {i_epoch}")
    plt.yscale('log')
    plt.savefig(f"{directory}RL_test_error_epo{i_epoch}.pdf")
    plt.close()

    plt.hist(test_pflags, bins=range(n_actions//2))
    plt.xlabel("Number of attepts till valid action")
    plt.ylabel("Events")
    plt.title(f"Test sample, episode: {i_epoch}")
    plt.savefig(f"{directory}RL_test_npflags_epo{i_epoch}.pdf")
    plt.close()
                      
    plt.hist(test_steps, bins=range(max_episode_length))
    plt.xlabel("Number of steps till done")
    plt.ylabel("Events")
    plt.title(f"Test sample, episode: {i_epoch}")
    plt.savefig(f"{directory}RL_test_nsteps_epo{i_epoch}.pdf")
    plt.close()
    
    plt.hist(test_tracks_used, bins=range(n_actions//2))
    plt.xlabel("Indices of tracks for vertexing")
    plt.ylabel("Occurances")
    #plt.title(f"Test sample, episode: {i_epoch}")
    plt.savefig(f"{directory}RL_test_tracks_used_epo{i_epoch}.pdf")
    plt.close()
    
    plt.scatter(test_decay_length, test_vertex_error, marker='.',s=15,c='k')
    plt.xlabel("Decay length (cm)")
    plt.ylabel("Vertex error (cm)")
    plt.title(f"Test sample, epoch: {i_epoch}")
    plt.savefig(f"{directory}RL_test_decay_error_scatter_epo{i_epoch}.pdf")
    plt.close()
    
    plt.hist2d(test_decay_length, test_vertex_error, bins=(np.linspace(0,20,200), np.linspace(0,20,200)))
    plt.xlabel("Decay length (cm)")
    plt.ylabel("Vertex error (cm)")
    plt.title(f"Test sample, epoch: {i_epoch}")
    plt.savefig(f"{directory}RL_test_decay_error_hist_epo{i_epoch}.pdf")
    plt.close()

    Q0s_sorted = []
    for n in range(max(test_tracks_avail)+1):
        Q0s_sorted.append([])
    for i in range(len(test_Q0s)):
        Q0s_sorted[test_tracks_avail[i]].append(test_Q0s[i])
        
    means = []
    stds = []
    lens = []
    
    latex_table_improv = 'n available tracks & average value $\pm$ std \\ '
    for n in range(max(test_tracks_avail)+1):
        mean_n = np.mean(Q0s_sorted[n], axis=0)
        means.append(mean_n)
        std_n = np.std(Q0s_sorted[n], axis=0)
        stds.append(std_n)
        len_n = len(Q0s_sorted[n])
        lens.append(len_n)
        out_str = f"For {n} valid tracks the average error improvement is {mean_n} +- {std_n}, {len_n} out of {len(test_Q0s)}"
        print(out_str)
        
        #for i in range(3, len(means)):
    #    latex_table_improv += f"{lens[i]} "
    '''
    for i in range(2, len(means)-1):
        print(i)
        latex_table_improv += f'\n {i+1} '
        for j in range(3, len(means)):
            latex_table_improv += f' & {means[j][i]:.2f} $\pm$ {stds[j][i]:.2f} '
        latex_table_improv += r'\\'
    '''
    for i in range(17):
        print(i)
        latex_table_improv += f'\n {i} '
        for j in range(3, len(means)):
            latex_table_improv += f' & {means[j][i]:.2f} $\pm$ {stds[j][i]:.2f} '
        latex_table_improv += r'\\'
        
    f = open(f"{directory}latex_table_average_out_all.txt", 'w')
    f.write(latex_table_improv)
    f.close()

print(f"Training ended, it took {time.time()-start_time:.2f} seconds for {epochs} epochs with {num_episodes} jets totalling {steps_done} steps")

plt.plot(range(len(episode_lengths)),episode_lengths)
plt.xlabel("Episode")
plt.ylabel("Actions")
plt.savefig(f"{directory}RL_epilength.png")
plt.close()

plt.scatter(range(len(rewards)), -np.log(-np.array(rewards)), marker='.',s=15,c='k')
plt.xlabel("Step")
plt.ylabel("Reward")
plt.savefig(f"{directory}RL_rewards.png")
plt.close()


print(len(memory))


torch.save(policy_net.state_dict(), f"{directory}model_final")

