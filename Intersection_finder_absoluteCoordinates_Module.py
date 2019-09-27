import numpy as np
from numpy import linalg as LA

r = 833.91024
pdis_init = 5

N = 30
points_n = 8
intervall_frac = 0.4

def helix(t, phi, eta, q, pt, dxy, dz, pvx, pvy, pvz):
    """Returns a point on a helix as numpy array

    The input are a running parameter and CMS track coordinates.
    The output coordinates are relative to the centre of CMS.
    phi, eta define the tangent of the helix at
    the point of closest approach (POCA)
    q is the charge of the particle, pt its transverse momentum.
    dxy and dz are the impact parameters.
    The magnetic field is 4T along the z direction.
    """

    x = -q * dxy * np.sin(phi) + pt * r * q * np.sin(phi)\
        + pt * r * np.cos(-q * t + phi + q * 0.5 * np.pi) + pvx

    y = q * dxy * np.cos(phi) - pt * r * q * np.cos(phi)\
        + pt * r * np.sin(-q * t + phi + q * 0.5 * np.pi) + pvy

    z = dz + t * pt * r / np.tan(2 * np.arctan(np.exp(-eta))) + pvz

    return np.array([x, y, z]) 


def isinbarrel(t, phi, eta, q, pt, dxy, dz, pvx, pvy, pvz):
    """Checks if a point on defined helix is still in the pixel barrel.

    Uses CMS coordinates and primary vertex coordinates to determine if
    point parametrized by t and helix parameters is inside pixel barrel.
    """

    x = helix(t, phi, eta, q, pt, dxy, dz, pvx, pvy, pvz)

    return (((x[0]**2 + x[1]**2) < 110**2) and (abs(x[2]) < 275))


def pointdistance(stepwidth, phi, eta, q, pt, dxy, dz):
    """Returns space stepwidth for points after one running parameter step.

    Returns the distance of neighbouring points on one helix given the
    helix parameters and a running parameter stepwidth.
    """

    return LA.norm(helix(0, phi, eta, q, pt, dxy, dz, 0, 0, 0)
                  - helix(stepwidth, phi, eta, q, pt, dxy, dz, 0, 0, 0))


def optstepwidth(pdis, phi, eta, q, pt, dxy, dz):
    """Returns a helix running paramter for a resultion of pdis.

    As long as the current stepwidth results in a too large separation
    of neighbouring points, the running parameter step is reduced by
    some factor.
    """

    stepwidth = 1.0
    counter = 0

    while pointdistance(stepwidth, phi, eta, q, pt, dxy, dz) > pdis:
        if counter > 1000:
            return -10
        stepwidth *= 0.6
        counter += 1

    return stepwidth


def scanrange(stepwidth, phi, eta, q, pt, dxy, dz, pvx, pvy, pvz):
    """Returns range as touple to scan the whole barrel at stepwidth.
    """

    t0 = 0.0
    counter0 = 0
    t1 = 0.0
    counter1 = 0

    while isinbarrel(t0, phi, eta, q, pt, dxy, dz, pvx, pvy, pvz):
        t0 += stepwidth
        counter0 += 1

    while isinbarrel(t1, phi, eta, q, pt, dxy, dz, pvx, pvy, pvz):
        t1 -= stepwidth
        counter1 += 1
   
    return (t1, t0)


def uncertainty_xyz(t, phi, eta, q, pt, dxy, dz,
                    dphi, deta, dpt, ddxy, ddz):
    """Returns a vector with uncertainties for a given point on a helix.

    Uncertainties from parametrisation of the helix propagate into the
    sampling of the helix. Uncertainties from helix parameters are
    input, uncertainty in x, y, z are output with gaussian error
    propagation.
    """

    x = dphi * ((- q * dxy + r * pt * q) * np.cos(phi)\
    - r * pt * np.sin(-q * t + phi + q * 0.5 * np.pi))\
    + dpt * (q * np.sin(phi) * r\
    + r * np.cos(-q*t + phi + q * 0.5 * np.pi))\
    - ddxy * q * np.sin(phi)

    y = dphi * ((- q * dxy + r * pt * q) * np.sin(phi)\
    + r * pt * np.cos(-q * t + phi + q * 0.5 * np.pi))\
    + dpt * (-q * np.cos(phi) * r + \
    r * np.sin(-q*t + phi + q * 0.5 * np.pi))\
    + ddxy * q * np.cos(phi)

    z = deta *(1/np.sin(2*np.arctan(np.exp(-eta))))**2\
    * np.exp(-eta)/(np.exp(-2*eta)+1)\
    + dpt * (r * t/np.tan(2 * np.arctan(np.exp(-eta)))) + ddz

    return np.array([x, y, z])

def findvertex_rp(tstart0, tend0, tstart1, tend1,
                  tstart0_new, tend0_new, tstart1_new, tend1_new,
                  phi0, eta0, q0, pt0, dxy0, dz0, pvx0, pvy0, pvz0,
                  phi1, eta1, q1, pt1, dxy1, dz1, pvx1, pvy1, pvz1):
    """Returns running parameters to get helices POCAs and distance.

    Helices running parameter space is sampled within tstart and tend
    with stepwidth tstepwith and distance is computed and compared to
    current minimum.
    Running parameters are returned as touple, first entry is t for
    helix0, second entry is t for helix1.
    """

    global N
    # determine the length of scanned range in rp space
    range0 = tend0_new-tstart0_new
    range1 = tend1_new-tstart1_new
    
    # sampling running parameters for the two helices
    rp0 = np.linspace(tstart0_new, tend0_new, num=N)#,dtype=np.double)
    rp1 = np.linspace(tstart1_new, tend1_new, num=N)#,dtype=np.double)
    
    #sampling points as (N,3) array
    points0 = np.transpose(helix(rp0, phi0, eta0, q0, pt0,
                                 dxy0, dz0, pvx0, pvy0, pvz0))
    points1 = np.transpose(helix(rp1, phi1, eta1, q1, pt1,
                                 dxy1, dz1, pvx1, pvy1, pvz1))
    
    # matrices with sampling points, one with copied rows, one with with copied coloumns
    p0_mat = np.tile(points0,(N,1,1))
    p1_mat = np.transpose(np.tile(points1,(N,1,1)),(1, 0, 2))
    
    ds = np.sum((p0_mat-p1_mat)**2, axis=-1)
    min_of_ds = np.amin(ds)
    #ind_sort_x, ind_sort_y = np.unravel_index(np.argsort(ds.flatten())[:4],ds.shape)
    ind_sort_x_0, ind_sort_y_0 = np.unravel_index(np.argsort(ds.ravel()),
                                                  (N,N),order='F')
    x_no_duplicates = []
    y_no_duplicates = []

    for i in ind_sort_x_0:
        if i not in x_no_duplicates:
            x_no_duplicates.append(i)
            
    for i in ind_sort_y_0:
        if i not in y_no_duplicates:
            y_no_duplicates.append(i)

    ind_sort_x_1, ind_sort_y_1 = (x_no_duplicates[:points_n],
                                  y_no_duplicates[:points_n])

    
    poca_min = np.sqrt(ds[ind_sort_x_1[0],ind_sort_y_1[0]])
    
    t0 = rp0[ind_sort_x_0[0]]
    t1 = rp1[ind_sort_y_0[0]]

    t0_min_d_cand = rp0[ind_sort_x_1]
    t1_min_d_cand = rp1[ind_sort_y_1]

    ''' How to return new bounds: take 10 min distances and take
    min and max of the running parameters of this set
    '''

    t0_min = min(t0_min_d_cand)
    t0_max = max(t0_min_d_cand)
    t1_min = min(t1_min_d_cand)
    t1_max = max(t1_min_d_cand)

    if t1 > tend1_new or t1 < tstart1_new:
        print("Alarm!")

    len0 = abs(t0_max - t0_min)*intervall_frac #range0 * 0.33
    len1 = abs(t1_max - t1_min)*intervall_frac #range1 * 0.33
        
    '''
    len0 = range0 * 0.12
    len1 = range1 * 0.12
    '''
    '''
    print "Ratio of intervall which contains close points: ",len0 / range0
    print "Ratio of total intervall which contains close points: ",len0 / (tend0-tstart0)
    '''
    
    #adjust len0 and len1 for pathologic cases
    if len0 == 0 or len0 > range0 * 0.45:
        len0 = range0 * 0.45

    if len1 == 0 or len1 > range1 * 0.45:
        len1 = range1 * 0.45
    
    #get new search intervalls 
    tstart0_newn = 0.
    tstart1_newn = 0.
    tend0_newn = 0.
    tend1_newn = 0.
    #min sourounded with margin
    '''
    if t0 - len0 > tstart0_new:
        tstart0_newn = t0 - len0
    else:
        tstart0_newn = tstart0_new
        
    if t0 + len0 < tend0_new:
        tend0_newn = t0 + len0
    else:
        tend0_newn = tend0_new

    if t1 - len1 > tstart1_new:
        tstart1_newn = t1 - len1
    else:
        tstart1_newn = tstart1_new
        
    if t1 + len1 < tend1_new:
        tend1_newn = t1 + len1
    else:
        tend1_newn = tend1_new
    '''
    #intervall sourounded with margin    
    #'''
    if t0_min - len0 > tstart0_new:
        tstart0_newn = t0_min - len0
    else:
        tstart0_newn = tstart0_new
        
    if t0_max + len0 < tend0_new:
        tend0_newn = t0_max + len0
    else:
        tend0_newn = tend0_new

    if t1_min - len1 > tstart1_new:
        tstart1_newn = t1_min - len1
    else:
        tstart1_newn = tstart1_new
        
    if t1_max + len1 < tend1_new:
        tend1_newn = t1_max + len1
    else:
        tend1_newn = tend1_new
    #'''
    #Min sourounded with global limits as scope
    '''
    if t0 - len0 > tstart0:
        tstart0_newn = t0 - len0
    else:
        tstart0_newn = tstart0
        
    if t0 + len0 < tend0:
        tend0_newn = t0 + len0
    else:
        tend0_newn = tend0

    if t1 - len1 > tstart1:
        tstart1_newn = t1 - len1
    else:
        tstart1_newn = tstart1
        
    if t1 + len1 < tend1:
        tend1_newn = t1 + len1
    else:
        tend1_newn = tend1
    '''    

    return t0, t1, tstart0_newn, tend0_newn, tstart1_newn, tend1_newn,t0_min_d_cand,t1_min_d_cand, poca_min

#    return t0, t1, tstart0_new, tend0_new, tstart1_new, tend1_new,t0_min_d_cand,t1_min_d_cand, poca_min


def vertexing_no_error_adaptive(phi0, eta0, q0, pt0, dxy0, dz0, 
                                pvx0, pvy0, pvz0, chi2_0, 
                                phi1, eta1, q1, pt1, dxy1, dz1,
                                pvx1, pvy1, pvz1, chi2_1,
                                pdis=pdis_init):
    """Returns a vertex candidate position,quality estimates,w/o errors.

    Needs weights function and should also return these.

    Also returns number of sampled points along both tracks. Starts 
    with initial scanning and then refined scanning until the scanning
    range uncertaintyis below resolution given by helix parameter
    uncertainties.
    This function needs still to weight the two points according to
    their respective uncertainties.
    """

    global N
    tstepwidth0 = optstepwidth(pdis, phi0, eta0, q0, pt0, dxy0, dz0)
    tstepwidth1 = optstepwidth(pdis, phi1, eta1, q1, pt1, dxy1, dz1)
    
    tstart0, tend0 = scanrange(tstepwidth0, phi0, eta0, q0, pt0, dxy0,
                               dz0, pvx0, pvy0, pvz0)
    tstart1, tend1 = scanrange(tstepwidth1, phi1, eta1, q1, pt1, dxy1, 
                               dz1, pvx1, pvy1, pvz1)

    t0, t1, tstart0_new, tend0_new, tstart1_new, tend1_new,t0_min_d_cand,t1_min_d_cand, poca_min = findvertex_rp(tstart0, tend0,
                                tstart1, tend1, tstart0, tend0,
                                tstart1, tend1,
                                phi0, eta0, q0, pt0, dxy0, dz0,
                                pvx0, pvy0, pvz0,
                                phi1, eta1, q1, pt1, dxy1, dz1,
                                pvx1, pvy1, pvz1)
    
    '''
    Find out how far points are separated, if above threshold zoom in.
    Then zoom in on suggested bounds from vertexing_rp
    '''
    
    pdis_0 = pointdistance((tend0-tstart0)/float(N),phi0, eta0, q0, pt0, dxy0, dz0)
    pdis_1 = pointdistance((tend1-tstart1)/float(N),phi1, eta1, q1, pt1, dxy1, dz1)
    pdis_max = max(pdis_0, pdis_1)
    counter = 0
    while pdis_max > 0.001:
        #------catch events that take too long---------
        if counter > 25:
            #broken_minimisation.append(jentry)
            poca_sep = LA.norm(helix(t0, phi0, eta0, q0, pt0, dxy0, dz0, pvx0, pvy0, pvz0)
                     - helix(t1, phi1, eta1, q1, pt1, dxy1, dz1, pvx1, pvy1, pvz1))
            return t0, t1, poca_sep, counter

        counter += 1
        t0,t1,tstart0_new,tend0_new,tstart1_new,tend1_new,t0_min_d_cand,t1_min_d_cand, poca_min = findvertex_rp(tstart0, tend0,
                                tstart1, tend1, tstart0_new, tend0_new,                                         
                                tstart1_new, tend1_new,
                                phi0, eta0, q0, pt0, dxy0, dz0,
                                pvx0, pvy0, pvz0,
                                phi1, eta1, q1, pt1, dxy1, dz1,
                                pvx1, pvy1, pvz1)

        pdis_0 = pointdistance((tend0_new-tstart0_new)/float(N),phi0, eta0, q0, pt0, dxy0, dz0)
        pdis_1 = pointdistance((tend1_new-tstart1_new)/float(N),phi1, eta1, q1, pt1, dxy1, dz1)
        pdis_max = max(pdis_0, pdis_1)

    poca_sep = LA.norm(helix(t0, phi0, eta0, q0, pt0, dxy0, dz0, pvx0, pvy0, pvz0)
                     - helix(t1, phi1, eta1, q1, pt1, dxy1, dz1, pvx1, pvy1, pvz1))

    return t0, t1, poca_sep, counter

def get_helix_params(i, X):
    """Returns the helix paramters in relevant order for vertexing.
    
    Input are the track index and a jet (collection of tracks)
    The ith PFC entry is taken and reordered to match the vertexing
    input. Another function can use this to create all arguments needed
    for the vertexing.
    
    This function must be changed in order to handle new input types."""

#['pt', 'eta', 'phi', 'charge', 'dxy', 'dz', 'pvIndex', 'pdgId']#, 'chi2', pvx, pvy, pvz needed!!!
#  0       1    2       3         4      5     6            7
    x0, x1, x2, x3, x4, x5 = X[i,:6]
    x6, x7, x8 = X[i, 8:11]
    # , x9, x10, x11, x12, x13
    phi = float(x2)
    eta = float(x1)
    q = float(x3)
    pt = float(x0)
    dxy  = float(x4)
    dz = float(x5)
    pvx = float(x6) #x6
    pvy = float(x7) #x7
    pvz = float(x8) #x8
    chi2 = 1
    return phi, eta, q, pt, dxy, dz, pvx, pvy, pvz, chi2
    
def helix_by_index(t, i, X):
    phi, eta, q, pt, dxy, dz, pvx, pvy, pvz, chi2_ = get_helix_params(i, X)
    return helix(t, phi, eta, q, pt, dxy, dz, pvx, pvy, pvz)
    
def vertexing_by_index(i, j, X):
    """Calls vertexing with the right argumets from jet data."""
    phi0, eta0, q0, pt0, dxy0, dz0, pvx0, pvy0, pvz0, chi2_0 = get_helix_params(i, X)
    phi1, eta1, q1, pt1, dxy1, dz1, pvx1, pvy1, pvz1, chi2_1 = get_helix_params(j, X)
    t0, t1, poca_sep, counter = vertexing_no_error_adaptive(phi0, eta0, q0, pt0, dxy0, dz0, 
                                pvx0, pvy0, pvz0, chi2_0, 
                                phi1, eta1, q1, pt1, dxy1, dz1,
                                pvx1, pvy1, pvz1, chi2_1,
                                pdis=pdis_init)
    return t0, t1, poca_sep, counter
