import sys
import time

import numpy as np
from numpy import linalg as LA
import ROOT
from array import array
import matplotlib
matplotlib.use('Agg')
import pylab
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


ROOT.gROOT.ProcessLine(".L Objects.h+")
    
from ROOT import PFCandidateType

ctau = 10
inFileName = "VBFH_HToSSTobbbb_MH-125_MS-20_ctauS-%s_TuneCUETP8M1_13TeV-powheg-pythia8_PRIVATE-MC.root"%(ctau)
outFileName = "hist.root"

inFile = ROOT.TFile.Open(inFileName, "READ")
tree = inFile.Get("ntuple/tree")


r = 833.91024
pdis_init = 1

N = 30
points_n = 8
intervall_frac = 0.4
barrel_on = True

broken_minimisation = []

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

    Uses CMS coordinates and primary vertex coordinates to determine if point
    parametrized by t and helix parameters is inside pixel barrel.
    """

    x = helix(t, phi, eta, q, pt, dxy, dz, pvx, pvy, pvz)

    return (((x[0]**2 + x[1]**2) < 110**2) and (abs(x[2]) < 275))


def pointdistance(stepwidth, phi, eta, q, pt, dxy, dz):
    """Returns space stepwidth for points after one running parameter step.

    Returns the distance of neighbouring points on one helix given the helix
    parameters and a running parameter stepwidth.
    """

    return LA.norm(helix(0, phi, eta, q, pt, dxy, dz, 0, 0, 0)
                   - helix(stepwidth, phi, eta, q, pt, dxy, dz, 0, 0, 0))


def optstepwidth(pdis, phi, eta, q, pt, dxy, dz):
    """Returns a helix running paramter to get a resultion of at least pdis.

    As long as the current stepwidth results in a too large separation of
    neighbouring points, the running parameter step is reduced by some factor.
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
    """Returns range as touple to scan the whole barrel at a given stepwidth.

    If too many steps are required to leave pixel barrel, reset something.
    Still needs to be implemented.
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


def uncertainty_xyz(t, phi, eta, q, pt, dxy, dz, dphi, deta, dpt, ddxy, ddz):
    """Returns a vector with uncertainties for a given point on a helix.

    Uncertainties from parametrisation of the helix propagate into the
    sampling of the helix. Uncertainties from helix parameters are input,
    uncertainty in x, y, z are output with gaussian error propagation.
    """

    x = dphi * ((- q * dxy + r * pt * q) * np.cos(phi)\
    - r * pt * np.sin(-q * t + phi + q * 0.5 * np.pi))\
    + dpt * (q * np.sin(phi) * r\
    + r * np.cos(-q*t + phi + q * 0.5 * np.pi))\
    - ddxy * q * np.sin(phi)

    y = dphi * ((- q * dxy + r * pt * q) * np.sin(phi)\
    + r * pt * np.cos(-q * t + phi + q * 0.5 * np.pi))\
    + dpt * (-q * np.cos(phi) * r + r * np.sin(-q*t + phi + q * 0.5 * np.pi))\
    + ddxy * q * np.cos(phi)

    z = deta *(1/np.sin(2*np.arctan(np.exp(-eta))))**2\
    * np.exp(-eta)/(np.exp(-2*eta)+1)\
    + dpt * (r * t/np.tan(2 * np.arctan(np.exp(-eta)))) + ddz

    return np.array([x, y, z])

def findvertex_rp(tstart0, tend0, tstart1, tend1,
                  tstart0_new, tend0_new, tstart1_new, tend1_new,
                  phi0, eta0, q0, pt0, dxy0, dz0, pvx0, pvy0, pvz0,
                  phi1, eta1, q1, pt1, dxy1, dz1, pvx1, pvy1, pvz1,
                  counter,jentry):
    """Returns running parameters of helices to get to their POCA and distance.

    Helices running parameter space is sampled within tstart and tend with
    stepwidth tstepwith and distance is computed and compared to current
    minimum.
    Running parameters are returned as touple, first entry is t for helix0,
    second entry is t for helix1.
    """

    global N
    # determine the length of scanned range in rp space
    range0 = tend0_new-tstart0_new
    range1 = tend1_new-tstart1_new
    
    # sampling running parameters for the two helices
    rp0 = np.linspace(tstart0_new, tend0_new, num=N)#,dtype=np.double)
    rp1 = np.linspace(tstart1_new, tend1_new, num=N)#,dtype=np.double)
    
    #sampling points as (N,3) array
    points0 = np.transpose(helix(rp0, phi0, eta0, q0, pt0, dxy0, dz0, pvx0, pvy0, pvz0))
    points1 = np.transpose(helix(rp1, phi1, eta1, q1, pt1, dxy1, dz1, pvx1, pvy1, pvz1))
    
    # matrices with sampling points, one with copied rows, one with with copied coloumns
    p0_mat = np.tile(points0,(N,1,1))
    p1_mat = np.transpose(np.tile(points1,(N,1,1)),(1, 0, 2))
    
    ds = np.sum((p0_mat-p1_mat)**2, axis=-1)
    min_of_ds = np.amin(ds)
    #ind_sort_x, ind_sort_y = np.unravel_index(np.argsort(ds.flatten())[:4],ds.shape)
    ind_sort_x_0, ind_sort_y_0 = np.unravel_index(np.argsort(ds.ravel()),(N,N),order='F')
    x_no_duplicates = []
    y_no_duplicates = []
    

    
    for i in ind_sort_x_0:
        if i not in x_no_duplicates:
            x_no_duplicates.append(i)
            
    for i in ind_sort_y_0:
        if i not in y_no_duplicates:
            y_no_duplicates.append(i)

    ind_sort_x_1, ind_sort_y_1 = x_no_duplicates[:points_n], y_no_duplicates[:points_n]

    
    poca_min = np.sqrt(ds[ind_sort_x_1[0],ind_sort_y_1[0]])
    print "Poca min from matrix: ", poca_min - np.sqrt(min_of_ds)
    '''
    t0_min_d_cand = [rp0[i] for i in ind_sort_x_1]
    t1_min_d_cand = [rp1[i] for i in ind_sort_y_1]
    '''
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

    t_index = np.unravel_index(np.argmin(ds.ravel()),ds.shape,order='F')
    '''
    t0 = rp0[t_index[0]]
    t1 = rp1[t_index[1]]
    '''
    if t1 > tend1_new or t1 < tstart1_new:
        print("Alarm!", jentry)

    
    if False:#jentry%100==0 or jentry == 0:
        '''
        rps = np.concatenate(rp0,rp1)
        a = np.concatenate(rps,ds, axis=1)
        np.save('plot_raw_%s_%s'%(jentry, counter),a)
        '''
        #np.set_printoptions(precision=4)
        #print ds
        ds_min = np.log(ds[ind_sort_y_0[0],ind_sort_x_0[0]])
        #print ds[ind_sort_x_0[0],ind_sort_y_0[0]]
        fig = plt.figure()
        ax = plt.axes()#projection='3d')
        ax.set_xlabel("ind_x")
        ax.set_ylabel("ind_y")
        #ax.set_zlabel("log(poca_sep)")
        '''
        ax.plot_surface(range(N), range(N), np.log(ds), rstride=1, cstride=1,cmap=cm.coolwarm)
        ax.scatter(ind_sort_x_1[:2],ind_sort_y_1[:2],2*[ds_min], label='4 minimal values')
        ax.scatter(t_index[0],t_index[1],ds_min, label='min')
        '''
        n = len(ind_sort_x_1)
        ax.contour(rp0, rp1, np.log(ds))
        ax.scatter(rp0[ind_sort_x_0[:n]],rp1[ind_sort_y_0[:n]], label='%s minimal values'%(n))
        ax.scatter(rp0[ind_sort_x_1],rp1[ind_sort_y_1], label='%s no duplicate minimal values'%(n)) 
        ax.scatter(rp0[t_index[0]],rp1[t_index[1]], label='min')
        #textstr = 
        #ax.text2D(0.01, 0.99, textstr, transform=ax.transAxes)
        ax.legend()
        #ax.set_title('Separation of points in running parameter space')
        fig.savefig("./index_based_minimising_%s_%s.png"%(jentry, counter))        
        plt.close()

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
    '''
    if jentry%1000 == 0:
        print rp0
        print rp1
        print rp0[ind_sort_x_0[0]], t0
        print rp1[ind_sort_y_0[0]], t1
    '''
    if False:#jentry%1000 == 0:
        ds_min = np.log(ds[ind_sort_x_0[0],ind_sort_y_0[0]])
        fig = plt.figure()
        ax = plt.axes()#projection='3d')
        ax.set_xlabel("t0")
        ax.set_ylabel("t1")
        #ax.set_zlabel("log(poca_sep)")
        #ax.plot_surface(rp0, rp1, np.log(ds), rstride=1, cstride=1,cmap=cm.coolwarm)
        ax.contour(rp0, rp1, np.log(ds))
        '''
        cset = ax.contour(rp0, rp1, np.log(ds), zdir='z')
        cset = ax.contour(rp0, rp1, np.log(ds), zdir='x')
        cset = ax.contour(rp0, rp1, np.log(ds), zdir='y')
        '''
        ax.scatter(t0,t1,ds_min, label='min')
        ax.scatter([tstart0_newn,tstart0_newn,tend0_newn,tend0_newn],
                   [tstart1_newn,tend1_newn,tstart1_newn,tend1_newn],
                   [ds_min]*4,label='area to zoom in')
        ax.legend()
        #ax.set_title('Separation of points in running parameter space')
        fig.savefig("./ds_%s_%s.png"%(jentry, counter))        
        plt.close()

    return t0, t1, tstart0_newn, tend0_newn, tstart1_newn, tend1_newn,t0_min_d_cand,t1_min_d_cand, poca_min

#    return t0, t1, tstart0_new, tend0_new, tstart1_new, tend1_new,t0_min_d_cand,t1_min_d_cand, poca_min


def vertexing_no_error_adaptive(phi0, eta0, q0, pt0, dxy0, dz0, 
                                pvx0, pvy0, pvz0, chi2_0, 
                                phi1, eta1, q1, pt1, dxy1, dz1,
                                pvx1, pvy1, pvz1, chi2_1,
                                jentry, pdis=pdis_init):
    """Returns a vertex candidate position,quality estimates,w/o errors.

    Needs weights function and should also return these.

    Also returns number of sampled points along both tracks. Starts 
    with initial scanning and then refined scanning until the scanning
    range uncertaintyis below resolution given by helix parameter
    uncertainties.
    This function needs still to weight the two points according to their
    respective uncertainties.
    """

    global N
    tstepwidth0 = optstepwidth(pdis, phi0, eta0, q0, pt0, dxy0, dz0)
    tstepwidth1 = optstepwidth(pdis, phi1, eta1, q1, pt1, dxy1, dz1)
    
    tstart0, tend0 = scanrange(tstepwidth0, phi0, eta0, q0, pt0, dxy0, dz0,
                               pvx0, pvy0, pvz0)
    tstart1, tend1 = scanrange(tstepwidth1, phi1, eta1, q1, pt1, dxy1, dz1,
                               pvx1, pvy1, pvz1)
                               
    counter = 0
    t0, t1, tstart0_new, tend0_new, tstart1_new, tend1_new,t0_min_d_cand,t1_min_d_cand, poca_min = findvertex_rp(tstart0, tend0,
                                tstart1, tend1, tstart0, tend0,
                                tstart1, tend1,
                                phi0, eta0, q0, pt0, dxy0, dz0,
                                pvx0, pvy0, pvz0,
                                phi1, eta1, q1, pt1, dxy1, dz1,
                                pvx1, pvy1, pvz1, counter,jentry)
                           
    t0_begin = [tstart0,tstart0_new]
    t0_ends = [tend0,tend0_new]
    t1_begin = [tstart1,tstart1_new]
    t1_ends = [tend1,tend1_new]
    t0_min_d_cands = [t0_min_d_cand]
    t1_min_d_cands = [t1_min_d_cand]
    t0s = [t0]
    t1s = [t1]
    
    print "Off by: ", poca_min - LA.norm(helix(t0, phi0, eta0, q0, pt0,
                                         dxy0, dz0, pvx0, pvy0, pvz0)
                     - helix(t1, phi1, eta1, q1, pt1, dxy1, dz1,
                             pvx1, pvy1, pvz1))

    poca_seps = [LA.norm(helix(t0, phi0, eta0, q0, pt0, dxy0, dz0, pvx0, pvy0, pvz0)
                     - helix(t1, phi1, eta1, q1, pt1, dxy1, dz1, pvx1, pvy1, pvz1))]
    
    
    '''
    Find out how far points are separated, if above threshold zoom in.
    Then zoom in on suggested bounds from vertexing_rp
    '''
    
    pdis_0 = pointdistance((tend0-tstart0)/float(N),phi0, eta0, q0, pt0, dxy0, dz0)
    pdis_1 = pointdistance((tend1-tstart1)/float(N),phi1, eta1, q1, pt1, dxy1, dz1)
    pdis_max = max(pdis_0, pdis_1)
    if jentry == 500:
        print "Pts: ", pt0, pt1
        print "Return of vertexing: ", t0, t1

    
    while pdis_max > 0.001:

        #------catch events that take too long---------
        if counter > 35:
            broken_minimisation.append(jentry)
            poca_sep = LA.norm(helix(t0, phi0, eta0, q0, pt0, dxy0, dz0, pvx0, pvy0, pvz0)
                     - helix(t1, phi1, eta1, q1, pt1, dxy1, dz1, pvx1, pvy1, pvz1))
            if False:#jentry%250==0:#poca_sep > 80: #andjentry > 14500 :
                plt.plot(range(len(t0_begin)),t0_begin, c='r',label='tBegin')
                plt.plot(range(len(t0_ends)),t0_ends, c='b',label='tEnd')
                plt.xlabel("Iteration")
                plt.ylabel("Running parameter")
                plt.legend()
            #plt.scatter(range(len(t0s)),t0s, c='g')
                for i in range(len(t0_min_d_cands)):
                    plt.scatter(i*np.ones(len(t0_min_d_cands[i])),t0_min_d_cands[i], c='g')
                plt.title('Final poca_ sep: %.5f, %s iterations \nTrack 1'%(poca_sep, counter))
                plt.savefig("scanned_intervalls0_jentry_%s.png"%(jentry))
                plt.close()
        
                plt.plot(range(len(t1_begin)),t1_begin, c='r',label='tBegin')
                plt.plot(range(len(t1_ends)),t1_ends, c='b',label='tEnd')
                plt.xlabel("Iteration")
                plt.ylabel("Running parameter")
                plt.legend()
                #plt.scatter(range(len(t1s)),t1s, c='g')
                for i in range(len(t1_min_d_cands)):
                    plt.scatter(i*np.ones(len(t1_min_d_cands[i])),t1_min_d_cands[i], c='g')
                plt.title('Final poca_ sep: %.5f, %s iterations \nTrack 2'%(poca_sep, counter))
                plt.savefig("scanned_intervalls1_jentry_%s.png"%(jentry))
                plt.close()
        

            return t0, t1, poca_sep, counter
                     

        '''
        print "Pdis_max: ", pdis_max
        print "\n"
        '''
        '''
        t0,t1,tstart0_new,tend0_new,tstart1_new,tend1_new,t0_min_d_cand,t1_min_d_cand, poca_min = findvertex_rp(tstart0, tend0,
                                tstart1, tend1, 0.5*(tstart0_new+tstart0), 0.5*(tend0_new+tend0),                                         
                                                                                                      0.5*(tstart1_new+tstart1), 0.5*(tend1_new+tend1),
                                phi0, eta0, q0, pt0, dxy0, dz0,
                                phi1, eta1, q1, pt1, dxy1, dz1,counter,jentry)
        '''
        counter += 1
        t0,t1,tstart0_new,tend0_new,tstart1_new,tend1_new,t0_min_d_cand,t1_min_d_cand, poca_min = findvertex_rp(tstart0, tend0,
                                tstart1, tend1, tstart0_new, tend0_new,                                         
                                tstart1_new, tend1_new,
                                phi0, eta0, q0, pt0, dxy0, dz0,
                                pvx0, pvy0, pvz0,
                                phi1, eta1, q1, pt1, dxy1, dz1,
                                pvx1, pvy1, pvz1, counter,jentry)
                                
        
        pdis_0 = pointdistance((tend0_new-tstart0_new)/float(N),phi0, eta0, q0, pt0, dxy0, dz0)
        pdis_1 = pointdistance((tend1_new-tstart1_new)/float(N),phi1, eta1, q1, pt1, dxy1, dz1)
        pdis_max = max(pdis_0, pdis_1)
        
        if jentry == 500:
            print "Return of vertexing: ", t0, t1
        t0_begin.append(tstart0_new)
        t0_ends.append(tend0_new)
        t0s.append(t0)
        t1s.append(t1)
        t1_begin.append(tstart1_new)
        t1_ends.append(tend1_new)
        t0_min_d_cands.append(t0_min_d_cand)        
        t1_min_d_cands.append(t1_min_d_cand)

        poca_sep = LA.norm(helix(t0, phi0, eta0, q0, pt0, dxy0, dz0, pvx0, pvy0, pvz0)
                     - helix(t1, phi1, eta1, q1, pt1, dxy1, dz1, pvx1, pvy1, pvz1))  
        print "Still off by: ", poca_min-poca_sep
        poca_seps.append(poca_sep)

    poca_sep = LA.norm(helix(t0, phi0, eta0, q0, pt0, dxy0, dz0, pvx0, pvy0, pvz0)
                     - helix(t1, phi1, eta1, q1, pt1, dxy1, dz1, pvx1, pvy1, pvz1))
    

    #print "Min separation of tracks: ", poca_sep

    if jentry%1000==0:#False:#poca_sep > 80: #andjentry > 14500 :
        
        plt.plot(range(len(poca_seps)),poca_seps)
        plt.xlabel("Iteration")
        plt.ylabel("Separation of pocas")
        plt.savefig("poca_change_jentry_%s.png"%(jentry))
        plt.close()
        
        plt.plot(range(len(t0_begin)),t0_begin, c='r',label='tBegin')
        plt.plot(range(len(t0_ends)),t0_ends, c='b',label='tEnd')
        plt.xlabel("Iteration")
        plt.ylabel("Running parameter")
        plt.legend()
        #plt.scatter(range(len(t0s)),t0s, c='g')
        for i in range(len(t0_min_d_cands)):
            plt.scatter(i*np.ones(len(t0_min_d_cands[i])),t0_min_d_cands[i], c='g')
        plt.title('Final poca_ sep: %.5f, %s iterations \nTrack 1'%(poca_sep, counter))
        plt.savefig("scanned_intervalls0_jentry_%s.png"%(jentry))
        plt.close()
        
        plt.plot(range(len(t1_begin)),t1_begin, c='r',label='tBegin')
        plt.plot(range(len(t1_ends)),t1_ends, c='b',label='tEnd')
        plt.xlabel("Iteration")
        plt.ylabel("Running parameter")
        plt.legend()
        #plt.scatter(range(len(t1s)),t1s, c='g')
        for i in range(len(t1_min_d_cands)):
            plt.scatter(i*np.ones(len(t1_min_d_cands[i])),t1_min_d_cands[i], c='g')
        plt.title('Final poca_ sep: %.5f, %s iterations \nTrack 2'%(poca_sep, counter))
        plt.savefig("scanned_intervalls1_jentry_%s.png"%(jentry))
        plt.close()
            
    return t0, t1, poca_sep, counter
    
def helix_distance(t, pvx, pvy, pvz,phi=[0,0], eta=[0,0], q=[1,1], pt =[0,0],
                   dxy=[0,0], dz=[0,0]):
    """Returns the scaled pointwise separation of two helices"""

    return (LA.norm(helix(t[0], phi[0], eta[0], q[0],
                                  pt[0], dxy[0], dz[0], pvx, pvy, pvz)
                   - helix(t[1], phi[1], eta[1], q[1],
                           pt[1], dxy[1], dz[1], pvx, pvy, pvz)))

# -------------------------------Main program --------------------------------
nEvents = 1000 #15000 #tree.GetEntriesFast()

ts = []
qs = []
t0s = []
t1s = []
poca_seps = []
third_track_dis = []
error_counter = 0
broken_minimisation = []
iterations = []
nVertex = 0

pt_fail = []

for jentry in range(nEvents):
    begin_time = time.time()
    entry = tree.GetEntry(jentry)
    highest_pt = [0, 0, 0]
    highest_pt_PFC_i = [0, 0, 0]

    
    if jentry%1 == 0:
        print "\n---------------------", "At event: ", jentry, "\n"
   
    #print(jentry)
    pv0_tracks = 0

    # find 3 highest pt tracks
    for a in range(tree.PFCandidates.size()):
        if (tree.PFCandidates[a].pdgId == 0 or tree.PFCandidates[a].pt < 0.5 or tree.PFCandidates[a].charge == 0):
            continue
        if tree.PFCandidates[a].pvIndex == 0:
            pv0_tracks += 1

            if tree.PFCandidates[a].pt > highest_pt[0]:
                highest_pt[2] = highest_pt[1]
                highest_pt[1] = highest_pt[0]
                highest_pt[0] = tree.PFCandidates[a].pt
                highest_pt_PFC_i[2] = highest_pt_PFC_i[1]
                highest_pt_PFC_i[1] = highest_pt_PFC_i[0]
                highest_pt_PFC_i[0] = a

            elif tree.PFCandidates[a].pt > highest_pt[1]:
                highest_pt[2] = highest_pt[1]
                highest_pt[1] = tree.PFCandidates[a].pt
                highest_pt_PFC_i[2] = highest_pt_PFC_i[1]
                highest_pt_PFC_i[1] = a

            elif tree.PFCandidates[a].pt > highest_pt[2]:
                highest_pt[2] = tree.PFCandidates[a].pt
                highest_pt_PFC_i[2] = a

    if pv0_tracks < 2:
            continue

    nVertex += 1
    # assign track variables
    a = highest_pt_PFC_i[0]
    b = highest_pt_PFC_i[1]
    pvx, pvy, pvz = tree.PrimaryVertices[0].x, tree.PrimaryVertices[0].y, tree.PrimaryVertices[0].z
    track_para0 = np.array([tree.PFCandidates[a].phi, tree.PFCandidates[a].eta,
                   tree.PFCandidates[a].charge, tree.PFCandidates[a].pt,
                   tree.PFCandidates[a].dxy, tree.PFCandidates[a].dz])
    track_para1 = np.array([tree.PFCandidates[b].phi, tree.PFCandidates[b].eta,
                   tree.PFCandidates[b].charge, tree.PFCandidates[b].pt,
                   tree.PFCandidates[b].dxy, tree.PFCandidates[b].dz])
    vertexing_begin = time.time()

    t0, t1, poca_sep, counter = vertexing_no_error_adaptive(track_para0[0], track_para0[1],
                                      track_para0[2], track_para0[3],
                                      track_para0[4], track_para0[5], 
                                      pvx, pvy, pvz, 1,
                                      track_para1[0], track_para1[1],
                                      track_para1[2], track_para1[3],
                                      track_para1[4], track_para1[5], 
                                      pvx, pvy, pvz, 1,
                                      jentry)

    t0s.append(t0)
    t1s.append(t1)

    #only used for plotting!!!!!!!!!!!!
    '''
    tstepwidth0 = optstepwidth(pdis_init, track_para0[0], track_para0[1],
                                      track_para0[2], track_para0[3],
                                      track_para0[4], track_para0[5])
    tstepwidth1 = optstepwidth(pdis_init, track_para1[0], track_para1[1],
                                      track_para1[2], track_para1[3],
                                      track_para1[4], track_para1[5])
    tstart0, tend0 = scanrange(tstepwidth0, track_para0[0], track_para0[1],
                                      track_para0[2], track_para0[3],
                                      track_para0[4], track_para0[5],
                               pvx, pvy, pvz)
    tstart1, tend1 = scanrange(tstepwidth1, track_para1[0], track_para1[1],
                               track_para1[2], track_para1[3],
                               track_para1[4], track_para1[5],
                               pvx, pvy, pvz)
    '''
    vertexing_end = time.time()
    poca_seps.append(poca_sep)

    pv = 0.5 * (helix(t0, track_para0[0], track_para0[1],
                      track_para0[2], track_para0[3],
                      track_para0[4], track_para0[5], pvx, pvy, pvz)\
            + helix(t1, track_para1[0], track_para1[1],
                    track_para1[2], track_para1[3],
                    track_para1[4], track_para1[5], pvx, pvy, pvz))

    pv_q = LA.norm(pv)
    qs.append(pv_q)
    
    if poca_sep > 5:
        #print track_para0[3], track_para1[3]
        pt_fail.append(track_para0[3])
        pt_fail.append(track_para1[3])
        

    if False:#(pv_q > 50 or poca_sep > 4): #jentry%1000==0:#and jentry%5==0: #jentry > 90 and jentry < 95:
        xx = pylab.linspace(tstart0, tend0, 300)
        yy = pylab.linspace(tstart1, tend1, 300)
        zz = pylab.zeros([len(xx), len(yy)])

        for i in xrange(len(xx)):
            for j in xrange(len(yy)):
                zz[i, j] = np.log(helix_distance([xx[i], yy[j]], pvx, pvy, pvz,
             phi=[track_para0[0],track_para1[0]],eta=[track_para0[1],track_para1[1]],
             q=[track_para0[2],track_para1[2]],pt=[track_para0[3],track_para1[3]],
             dxy=[track_para0[4],track_para1[4]],dz=[track_para0[5],track_para1[5]]) + 0.0000000001)
        #----------------------distance plane plot-----------------------
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlabel("t0")
        ax.set_ylabel("t1")
        ax.set_zlabel("log(poca_sep)")
        ax.plot_surface(xx, yy, zz, rstride=1, cstride=1,cmap=cm.coolwarm)
        ax.scatter(t0, t1, np.log(poca_sep+ 0.0000000001), c='r',label='min_sep')
        textstr = 'min track dis=%.4f \npv_q=%.4f' % (poca_sep,pv_q)
        ax.text2D(0.01, 0.99, textstr, transform=ax.transAxes)
        ax.legend()
        #ax.set_title('Separation of points in running parameter space')
        fig.savefig("./d_plot%s.png"%(jentry))        
        plt.close()
        
        #------------------------track plot-------------------------------
        points0 = [helix(t, track_para0[0], track_para0[1],
                         track_para0[2], track_para0[3],
                         track_para0[4], track_para0[5], pvx, pvy, pvz) for t in np.linspace(tstart0, tend0, 1000)] 
        points1 = [helix(t, track_para1[0], track_para1[1],
                         track_para1[2], track_para1[3],
                         track_para1[4], track_para1[5], pvx, pvy, pvz) for t in np.linspace(tstart1, tend1, 1000)] 
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.plot([i[0] for i in points0], [i[1] for i in points0], [i[2] for i in points0],label='track0')
        ax.plot([i[0] for i in points1], [i[1] for i in points1], [i[2] for i in points1],label='track1')
        if False:#pv0_tracks > 2:
            c = highest_pt_PFC_i[2]
            track_para2 = [tree.PFCandidates[c].phi, tree.PFCandidates[c].eta,
                       tree.PFCandidates[c].charge, tree.PFCandidates[c].pt,
                       tree.PFCandidates[c].dxy, tree.PFCandidates[c].dz]
            tstepwidth2 = optstepwidth(pdis_init, track_para2[0], track_para2[1],
                                      track_para2[2], track_para2[3],
                                      track_para2[4], track_para2[5])
            tstart2, tend2 = scanrange(tstepwidth2, track_para2[0], track_para2[1],
                               track_para2[2], track_para2[3],
                               track_para2[4], track_para2[5],
                               pvx, pvy, pvz)
            t02, t20, poca_sep, counter = vertexing_no_error_adaptive(track_para0[0], track_para0[1],
                                      track_para0[2], track_para0[3],
                                      track_para0[4], track_para0[5],
                                      pvx, pvy, pvz, 1,
                                      track_para2[0], track_para2[1],
                                      track_para2[2], track_para2[3],
                                      track_para2[4], track_para2[5],
                                      pvx, pvy, pvz, 1,
                                      jentry)
            t12, t21, poca_sep, counter = vertexing_no_error_adaptive(track_para1[0], track_para1[1],
                                      track_para1[2], track_para1[3],
                                      track_para1[4], track_para1[5],
                                      pvx, pvy, pvz, 1,
                                      track_para2[0], track_para2[1],
                                      track_para2[2], track_para2[3],
                                      track_para2[4], track_para2[5],
                                      pvx, pvy, pvz,1,
                                      jentry)
            points2 = [helix(t, track_para2[0], track_para2[1],
                         track_para2[2], track_para2[3],
                         track_para2[4], track_para2[5], pvx, pvy, pvz,) for t in np.linspace(tstart2, tend2, 1000)] 
            ax.plot([i[0] for i in points2], [i[1] for i in points2], [i[2] for i in points2],label='track2')
            poca20 = helix(t20, track_para2[0], track_para2[1],
                         track_para2[2], track_para2[3],
                         track_para2[4], track_para2[5], pvx, pvy, pvz)
            poca21 = helix(t21, track_para2[0], track_para2[1],
                         track_para2[2], track_para2[3],
                         track_para2[4], track_para2[5], pvx, pvy, pvz)
            poca02 = helix(t02, track_para0[0], track_para0[1],
                         track_para0[2], track_para0[3],
                         track_para0[4], track_para0[5], pvx, pvy, pvz)
            poca12 = helix(t12, track_para1[0], track_para1[1],
                         track_para1[2], track_para1[3],
                         track_para1[4], track_para1[5])
            ax.scatter(poca20[0], poca20[1], poca20[2], c='k',label='poca20')
            ax.scatter(poca21[0], poca21[1], poca21[2], c='yellow',label='poca21')
            ax.scatter(poca02[0], poca02[1], poca02[2],label='poca02')
            ax.scatter(poca12[0], poca12[1], poca12[2],label='poca12')
            
        ax.scatter(pv[0],pv[1],pv[2],c='b',label='reco_vertex')
        
        poca0 = helix(t0, track_para0[0], track_para0[1],
                         track_para0[2], track_para0[3],
                         track_para0[4], track_para0[5], pvx, pvy, pvz)
        poca1 = helix(t1, track_para1[0], track_para1[1],
                         track_para1[2], track_para1[3],
                         track_para1[4], track_para1[5], pvx, pvy, pvz)
        
        ax.scatter(poca0[0], poca0[1], poca0[2], c='g',label='poca0')
        ax.scatter(poca1[0], poca1[1], poca1[2], c='r',label='poca1')
        ax.scatter(pvx, pvy, pvz,c='c',label='gen_vertex')
      
        if barrel_on:
        # Cylinder
            x=np.linspace(-110, 110, 100)
            z=np.linspace(-275, 275, 100)
            Xc, Zc=np.meshgrid(x, z)
            Yc = np.sqrt(110**2-(Xc * np.ones (len(Xc)))**2) * np.ones (len(Xc))
        # Draw parameters
            ax.plot_surface(Xc,-Yc, Zc, alpha=0.2, color='g',rstride=20, cstride=10)
            ax.plot_surface(Xc, Yc, Zc, alpha=0.2, color='g',rstride=20, cstride=10)

        textstr = 'min track dis=%.4f \npv_q=%.4f' % (poca_sep,pv_q)
        ax.text2D(0.05, 0.95, textstr, transform=ax.transAxes)
        ax.legend()
        #ax.set_title('Tracks, pocas and vertices')
        plt.savefig("./bad_helices_n%s.png"%(jentry))
        plt.close()
        

    '''
    if pv0_tracks > 2:
        c = highest_pt_PFC_i[2]
        track_para2 = [tree.PFCandidates[c].phi, tree.PFCandidates[c].eta,
                       tree.PFCandidates[c].charge, tree.PFCandidates[c].pt,
                       tree.PFCandidates[c].dxy, tree.PFCandidates[c].dz]
        t02, t20, poca_sep, counter = vertexing_no_error_adaptive(track_para0[0], track_para0[1],
                                      track_para0[2], track_para0[3],
                                      track_para0[4], track_para0[5], 1,
                                      track_para2[0], track_para2[1],
                                      track_para2[2], track_para2[3],
                                      track_para2[4], track_para2[5], 1,
                                      pvx, pvy, pvz, jentry)
        third_track_dis.append(track_dis)
            pv = 0.33333 * (helix(t0, track_para0[0], track_para0[1],
                      track_para0[2], track_para0[3],
                      track_para0[4], track_para0[5])\
            + helix(t1, track_para1[0], track_para1[1],
                    track_para1[2], track_para1[3],
                    track_para1[4], track_para1[5])\
            + helix(t20, track_para2[0], track_para2[1],
                    track_para2[2], track_para2[3],
                    track_para2[4], track_para2[5]))
    
    '''
    ts.append(vertexing_end - vertexing_begin)
    iterations.append(counter)

    '''
    print "----------------------------------"
    print "Event: ", jentry
    if pv0_tracks > 2:
        print "Vertex displacement, 3 tracks: ", LA.norm(pv)
    else:
        print "Vertex displacement, 2 tracks: ", LA.norm(pv)
    print "Vertexing time: ", vertexing_end - vertexing_begin
    print "----------------------------------\n"
    '''
#print "Different solver!!!!!!"
print "Early terminations: ", error_counter
print "Passed cut: ", nVertex

print "Error rate: ", error_counter/float(len(poca_seps))
print "Maximum separation of POCAs: ",max(poca_seps)


directory = '.'

plt.hist(ts, bins=20)
plt.xlabel("Vertexing time (s)")
plt.title(r"Vertexing time spectrum, ctau = %s, N = %s" %(ctau, nEvents))
plt.savefig("./%s/vertexing_times_minimize_tau%s_n%s_.png"%(directory,ctau, nEvents))
plt.close()

plt.hist(t0s, bins=20)
plt.xlabel("Running paramter (rad)")
plt.title(r"ctau = %s, N = %s" %(ctau, nEvents))
plt.savefig("./%s/vertexing_rp0_minimize_tau%s_n%s_.png"%(directory,ctau, nEvents))
plt.close()

plt.hist(t1s, bins=20)
plt.xlabel("Running paramter (rad)")
plt.title(r"ctau = %s, N = %s" %(ctau, nEvents))
plt.savefig("./%s/vertexing_rp1_minimize_tau%s_n%s_.png"%(directory,ctau, nEvents))
plt.close()

plt.hist(iterations)
plt.xlabel("Iterations")
plt.title(r"Number of Iterations, ctau = %s, N = %s" %(ctau, nEvents))
plt.yscale('log')
plt.savefig("./%s/vertexing_iterations_minimize_tau%s_n%s_.png"%(directory,ctau, nEvents))
plt.close()

plt.hist(qs)
plt.xlabel("Reco to gen PV distance (mm)")
plt.yscale('log')
plt.title("ctau = %s, N = %s"%(ctau, nEvents))
plt.savefig("./%s/vertexing_quality_tau%s_n%s.png"%(directory,ctau, nEvents))
plt.close()

cleaned_qs = [i for i in qs if i <= 1]

plt.hist(cleaned_qs, bins=100)
plt.xlabel("Reco to gen PV distance (mm)")
plt.title("Only for qs <= 1 mm, integral: %s, ctau = %s, N = %s"%(len(cleaned_qs),ctau, nEvents))
plt.savefig("./%s/vertexing_quality_cleaned_tau%s_n%s.png"%(directory,ctau, nEvents))
plt.close()

print "mean cleaned: ", np.mean(cleaned_qs), "\nrms cleaned: ", np.sqrt(np.mean(np.array(cleaned_qs)**2))

textstr = "\n".join(["nentries = %s"%(len(poca_seps)), "mean: %3.2f"%(np.mean(poca_seps)),
                   "rms: %.2f"%(np.sqrt(np.mean(np.array(poca_seps)**2)))])
print(textstr)

plt.hist(poca_seps, bins=range(100))
plt.xlabel("Distance of pocas (mm)")
plt.yscale('log')
plt.title("ctau = %s, N = %s"%(ctau, nEvents))
plt.savefig("./%s/vertexing_poca_dis_tau%s_n%s.png"%(directory,ctau, nEvents))
plt.close()

cleaned_poca_seps = [i for i in poca_seps if i <= 0.1]

plt.hist(cleaned_poca_seps,bins=100)
plt.xlabel("Distance of pocas (mm)")
plt.title("Only for poca_sep <= 0.1 mm, integral: %s, ctau = %s, N = %s"%(len(cleaned_poca_seps),ctau, nEvents))
plt.savefig("./%s/vertexing_poca_dis_cleaned_tau%s_n%s.png"%(directory,ctau, nEvents))
plt.close()


plt.hist(pt_fail,bins=100)
plt.xlabel("Failing pt (GeV)")
plt.title("ctau = %s, N = %s"%(ctau, nEvents))
plt.savefig("./%s/vertexing_fail_pt_tau%s_n%s.png"%(directory,ctau, nEvents))
plt.close()


'''
plt.hist(third_track_dis, bins=20)
plt.xlabel("Distance of third track to PV (mm)")
plt.title(r"Distance of third track, ctau = %s, N = %s"&(ctau,nEvents,solver))
plt.savefig("./%s/vertexing_third_track_tau%s_n%s_solver%s.png"%(directory,ctau, nEvents,solver))
plt.show()
'''


