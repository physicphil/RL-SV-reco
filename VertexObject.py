"""
Assemble a vertex based on quality and from a list of coordinates.
Return rewards?

Pseudocode at the moment, the actual layout of the jet data is needed next.
Imagine track index is enough as an argument, set up functions acordingly.
"""
import numpy as np
from numpy import linalg as LA

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from sklearn.preprocessing import StandardScaler

import Intersection_finder_absoluteCoordinates_Module as I

class Poca(object):
    """A point of closest approach lies on a track i.
    
    j gives track index to which it is closest.
    weight is based on quality.
    """
    def __init__(self, i, t, j, X):
        #Vertex computation here
        self.x = I.helix_by_index(t, i, X) # point in space (numpy)
        self.track = i # track index of poca
        self.next_to = j # next to this track
        self.weight = 1 # weigth determined by uncertainty


class VertexCandidate(object):
    """Vertex candidate object
    This object stores the current position of the vertex.
    Tracks can be added or removed.
    If penalty is returned, an unintended action has been taken
    The tracks and their weights are stored from which the vertex is
    recalculated evert time a track is either added or removed.
    If only one track is included, None is stored as position.
    This needs to be remembered for the reward function!
    """
    def __init__(self, i, track_data, state):
        """ Initialises the vertex object with one track
        
        No position is initially returned.
        Has 5 attributes and 3 methods.
        Atrr:
        .track_indices indices of added tracks (list)
        .vertex vertex position numpy array)
        .uncertainty
        .num_steps, the number of steps taken
        .pocas list of Poca objects
        """
        self.track_indices = [i] # list of track indices
        self.x = None # 3d coordinates of vertex
        self.uncertainty = None # measure of spread of pocas
        self.num_steps = 0 # how many steps have been done by the agent
        self.pocas = [] # list of Poca objects
        state[0,0, i, -1] = 1
        
    def add_track(self, i, track_data, state):
        """Adds a track to the vertex candidate.
        
        The pocas to all other tracks are computed and the weighted
        midpoint is computed.
        Return values are the position of the vertex, the uncertainty,
        the number of steps taken and if the track was added before.
        If flag the state will not be changed. Additionally a penalty
        can be given.
        """
        self.num_steps += 1
        # if track was already added or is from 0 padding, do nothing and give a penalty
        #                              pt == 0?
        if i in self.track_indices or track_data[i, 0] == 0:
            return self.x, self.uncertainty, self.num_steps, True
        # check if track is from zero padding, if so, do the same as with an
        # already added track.
        if state[0,0,i,-1] == 1:
            print("Vertex bookkeeping broken!!!!")
        # compute pocas to all other already added tracks
        for j in self.track_indices:
            t_i, t_j, poca_sep, iter_counter = I.vertexing_by_index(i, j, track_data)
            self.pocas.append(Poca(i, t_i, j, track_data))
            self.pocas.append(Poca(j, t_j, i, track_data))
            #self.pocas.append(Poca(i, 0, j, state))
            #self.pocas.append(Poca(j, 0, i, state))
        
        # add track index to indices list
        self.track_indices.append(i)
        # compute vertex from pocas
        self.x = self.calc_vertex()
        self.uncertainty = self.calc_uncertainty()
        return self.x, self.uncertainty, self.num_steps, False

    def rm_track(self, i, track_data, state):
        """A track is removed from vertex candidate.
        
        This track must have been added before.
        The track index is deleted, so are the pocas and weights.
        Recomputation of vertex position and uncertainties.
        Return values are the position of the vertex, the uncertainty,
        the number of steps taken and if the track was added before.
        If flag the state will not be changed. Additionally a penalty
        can be given.
        If the only two or one tracks are in index list, the vertex is 
        set to none.
        """
        '''
        self.num_steps += 1
        if len(self.track_indices) < 3 and i in self.track_indices:
            self.track_indices.remove(i)
            new_poca_list = []
            for j in range(len(self.pocas)):
                if self.pocas[j].track != i and self.pocas[j].next_to != i:
                    new_poca_list.append(self.pocas[j])
            self.pocas = new_poca_list
            self.x = self.calc_vertex()
            self.uncertainty = self.calc_uncertainty()
            return self.x, self.uncertainty, self.num_steps, False
        
        elif len(self.track_indices) == 1 and i in self.track_indices:
            self.track_indices.remove(i)
            self.x = self.calc_vertex()
            self.uncertainty = self.calc_uncertainty()
            self.pocas = []
            return self.x, self.uncertainty, self.num_steps, False
        '''
        self.num_steps += 1
        if i not in self.track_indices or len(self.track_indices) < 3:
            return self.x, self.uncertainty, self.num_steps, True
        # check if track was already added. If not -> penalty
        self.track_indices.remove(i)
        new_poca_list = []
        for j in range(len(self.pocas)):
            if self.pocas[j].track != i and self.pocas[j].next_to != i:
                new_poca_list.append(self.pocas[j])
        self.pocas = new_poca_list
        self.x = self.calc_vertex()
        self.uncertainty = self.calc_uncertainty()
        return self.x, self.uncertainty, self.num_steps, False
        
    def vertex_stop(self):
        return self.x, self.uncertainty, self.num_steps, False
        
    def calc_vertex(self):
        """Takes list of pocas and computes the weighted midpoint."""
        if len(self.pocas) < 2:
            return None
        norm_fac = 0
        vertex = np.zeros(3)
        for poca in self.pocas:
            norm_fac += poca.weight
        for poca in self.pocas:
            vertex += (poca.weight/norm_fac) * poca.x
        return vertex

    def calc_uncertainty(self):
        """This method computes the spread of pocas around the vertex"""
        if len(self.pocas) < 2:
            return None
        norm_fac = 0
        rms = 0
        for poca in self.pocas:
            norm_fac += poca.weight
        for poca in self.pocas:
            rms += np.sum((self.x - poca.x)**2)*(poca.weight/norm_fac) 
        return np.sqrt(rms)


class TrackEnvironment(object):
    """Track environment containing all valid tracks.
    Another dimension keeps track of which tracks have been added.
    """
    def __init__(self, jet, scaler=None, para_list=[0,3,4,5,6,7,8,9,13,14,15], soft_start=False):
        """Make a tensor of all valid track parameters (pT > 1GeV).
        Add dim to flag all added tracks.
        Zero flag to match input dimension of policy net.
        """
        self.has_sv = True #jet.sv_flag
        # also include nPixelHits, theta for learning etc.
        self.track_data = torch.tensor(jet,dtype=torch.float) # with a gather or where? # padding (done in input)
        self.n = self.track_data.shape[0]
        self.n_valid_tracks = self.num_valid_tracks()
        data = jet
        # if a scaler is provided, scale the jet data, then convert to tensor
        dummies = np.zeros(max(data.shape[1], 17))
        dummies[[0,1,2,3,4,5,6,7,8]] = -5, -4, -2, -2, -5,-10,-10, -10, -10
        dummies[[9,10,11,12,13,14,15,16]] = -5, -1, -1,-1, -2, -10, -5, -5
        if type(scaler) == StandardScaler:
            for track in data:
                if track[0] >= 1:
                    track = scaler.transform([track])
                else:
                    track = dummies[:data.shape[1]]
        data = torch.tensor(data, dtype=torch.float)
        # here things can be varied to change the data included
        self.track_variable_mask = torch.tensor(para_list)
        needed_track_data = data[:, self.track_variable_mask]
        states = torch.zeros((self.track_data.shape[0],1))
        self.state = torch.cat((needed_track_data, states), 1).unsqueeze(0).unsqueeze(0)# add a dimension or column with 0
        self.vertex = VertexCandidate(0, self.track_data,self.state)
        self.take_action(1)
        if soft_start:
            self.vertex = VertexCandidate(self.n_valid_tracks-1, self.track_data,self.state)
            self.take_action(self.n_valid_tracks-2)
  
    def take_action(self, a):
        """ There are 2n + 1 actions: add track 0 to n-1, remove track
        from 0 to n-1 and stop vertexing.
        Returns current state and information
        needed to calculate reward.
        """
        # pflag: penalty flag (nothing was done)
        # dflag: done flag, when agent select stop action this is set to true
        vertex, uncer, numsteps, pflag, dflag = -1, -1, -1, -1, False
        if a < self.n:
            vertex, uncer, numsteps, pflag = self.vertex.add_track(a, self.track_data, self.state)
            if not pflag:
                self.state[0,0,a, -1] = 1
        elif a < 2*self.n:
            vertex, uncer, numsteps, pflag = self.vertex.rm_track(a-self.n, self.track_data, self.state)
            if not pflag:
                self.state[0,0,a-self.n, -1] = 0
        elif a == 2*self.n:
            vertex, uncer, numsteps, pflag = self.vertex.vertex_stop()
            dflag = True
        else:
            print("Choose a proper action")
        #      next state  reward   reward  reward   penalty  done
        return self.state, vertex , uncer, numsteps, pflag, dflag
        
    def num_valid_tracks(self):
        for i in range(self.track_data.shape[0]):
            if self.track_data[i,0] == 0:
                if i > 1:
                   return i
        return self.n
        
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
