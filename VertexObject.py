"""
Assemble a vertex based on quality and from a list of coordinates.
Classes?
Return rewards?
"""   
        
class Poca(object):
    """A point of closest approach lies on a track i.
    
    j gives track index to which it is closest.
    weight is based on quality.
    """
    def __init__(self, i, t, j):
        #Vertex computation here
        self.x = helix(t, i)
        self.track = i
        self.next_to = j
        self.weight = 1


class VertexCandidate(object):
    """Vertex candidate object
    This object stores the current position of the vertex.
    Tracks can be added or removed.
    If penalty is returned, an unintended action has been taken
    The tracks and their weights are stored from which the vertex is
    recalculated evert time a track is either added or removed.
    If only one track is included, None is stored as position.
    This needs to be remembered for the reward function!
    How is vertex aware of track variables and environment?
    """
    def __init__(self, track):
        """ Initialises the vertex object with one track
        
        No position is initially returned.
        """
        self.track_indices = [track]
        self.vertex = None
        self.uncertainty = None
        self.num_steps = 0
        self.pocas = []
        
    def add_track(self, i):
        """Adds a track to the vertex candidate.
        
        The pocas to all other tracks are computed and the weighted
        midpoint is computed.
        Return values are the position of the vertex, the uncertainty,
        the number of steps taken and if the track was added before.
        If flag the state will not be changed. Additionally a penalty
        can be given.
        """
        self.num_steps += 1
        if i in self.track_indices:
            return self.vertex, self.uncertainty, self.num_steps, True
        # check if track is from zero padding, if so, do the same as with an
        # already added track.
        for j in self.track_indices:
            t_i, t_j = vertexing(i, j)
            self.pocas.append(Poca(i, t_i, j))
            self.pocas.append(Poca(j, t_j, i))
        self.track_indices.append(i)
        self.vertex = calc_vertex(self.pocas)
        self.uncertainty = calc_uncertainty(self.pocas)
        return self.vertex, self.uncertainty, self.num_steps, False
        
    def rm_track(self, i):
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
        self.num_steps += 1
        if len(self.track_indices) == 2 and i in self.track_indices:
            self.vertex = None
            self.uncertainty = None
            self.track_indices.remove(i)
            new_poca_list = []
            new_weight_list = []
            for j in range(len(self.pocas)):
                if self.pocas[j].track != i and self.pocas[j].next_to != i:
                    new_poca_list.append(self.pocas[j])
                    new_weight_list.append(self.weights[j])
            self.pocas = new_poca_list
            self.weights = new_weight_list
            return self.vertex, self.uncertainty, self.num_steps, False
            
        elif len(self.track_indices) == 1 and i in self.track_indices:
            self.vertex = None
            self.uncertainty = None
            self.track_indices = []
            self.pocas = []
            self.weights = []
            return self.vertex, self.uncertainty, self.num_steps, False
            
        elif i not in self.track_indices:
            return self.vertex, self.uncertainty, self.num_steps, True
        # check if track is from zero padding, if so, do the same as with an
        # already added track.
        self.track_indices.remove(i)
        new_poca_list = []
        new_weight_list = []
        for j in range(len(self.pocas)):
            if self.pocas[j].track != i and self.pocas[j].next_to != i:
                new_poca_list.append(self.pocas[j])
                new_weight_list.append(self.weights[j])

        self.pocas = new_poca_list
        self.weights = new_weight_list
        self.vertex = calc_vertex(self.pocas)
        self.uncertainty = calc_uncertainty(self.pocas, self_weights)
        return self.vertex, self.uncertainty, self.num_steps, False
        
    def vertex_stop(self):
        return self.vertex, self.uncertainty, self.num_steps, False
        
    def calc_vertex(self, pocas):
        """Takes list of pocas and computes the weighted midpoint."""
        if len(pocas) < 2:
            return None
        norm_fac = 0
        vertex = np.zeros(3)
        for poca in pocas:
            norm_fac += poca.weight
        for poca in pocas:
            vertex += (poca.weight/form_fac) * poca.x
        
        

class TrackEnvironment(object):
    """Track environment containing all valid tracks.
    Another dimension keeps track of which tracks have been added.
    """
    def __init__(self, jet):
        """Make a tensor of all valid track parameters (pT > 1GeV).
        Add dim to flag all added tracks.
        Zero flag to match input dimension of policy net.
        """
        self.has_sv = jet.sv_flag
        # how to apply pt cut?
        self.track_data = torch.tensor(jet) # with a gather or where? # padding
        self.state = self.track_data # add a dimension or column with 0
        self.vertex = VertexCandidate(0)
  
    def take_action(self, a):
        """ There are 2n + 1 actions: add track 0 to n-1, remove track
        from 0 to n-1 and stop vertexing.
        Returns current state and information
        needed to calculate reward.
        """
        # pflag: penalty flag (nothing was done)
        # dflag: done flag, when agent select stop action this is set to true
        vertex, uncer, numsteps, pflag, dflag = -1, -1, -1, -1, False
        if a < n:
            vertex, uncer, numsteps, pflag = self.vertex.add_track(a)
            self.state[a, 0] = 1
        elif a < 2*n:
            vertex, uncer, numsteps, pflag = self.vertex.rm_track(a-n)
            self.state[a-n, 0] = 0
        elif a == 2*n:
            vertex, uncer, numsteps, pflag = self.vertex.vertex_stop()
            dflag = True
        else:
            print("Choose a proper action")
        return self.state, vertex , uncer, numsteps, pflag, dflag
        

        
        
		
    
		
