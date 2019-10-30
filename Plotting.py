import numpy as np
import matplotlib.pyplot as plt

color_dict = {0: 'k', 1: 'b', 2: 'r', 3: 'g', 4: 'm', 5: 'c', 6: 'y', 7: 'm'}

paths = ['Leading_tracks', 'Random_policy', 'Test_Train_restr', 'Test_restr',
         'Train_restr', 'None_restr']

save_path = './Plots_second_try/'
hfont = {'fontname':'Arial'}

class AgentPerformance(object):
    def __init__(self, decay_lengths, errors, uncertainties, n_tracks_used,
                 n_tracks_available, name, bins=np.arange(0, 5, 0.5),
                 uncer_lim=lambda x: 1, error_lim=lambda x: 1):
        self.decay_lengths = decay_lengths
        self.errors = errors
        self.uncertainties = uncertainties
        self.n_tracks_used = n_tracks_used
        self.n_tracks_available = n_tracks_available
        self.name = name
        self.bins = bins
        self.uncer_lim = uncer_lim
        self.error_lim = error_lim
        self.vertex_candidates = None
        self.cut_data()

    def cut_data(self):
        cut_decay = []
        cut_error = []
        cut_n_tracks_used = []
        cut_tracks_avail = []
        for i in range(len(self.decay_lengths)):
            if (self.uncertainties[i] < self.uncer_lim(self.decay_lengths[i])):
                cut_decay.append(self.decay_lengths[i])
                cut_error.append(self.errors[i])
                cut_n_tracks_used.append(self.n_tracks_used[i])
                cut_tracks_avail.append(self.n_tracks_available[i])
        cut_decay = np.array(cut_decay)
        cut_error = np.array(cut_error)
        cut_n_tracks_used = np.array(cut_n_tracks_used)
        cut_tracks_avail = np.array(cut_tracks_avail)
        self.vertex_candidates = np.vstack((cut_decay, cut_error,
                                            cut_n_tracks_used,
                                            cut_tracks_avail)).T
        return 0

    def get_efficiencies(self):
        total_output, bin_edges = np.histogram(self.decay_lengths,
                                               bins=self.bins)
        vertices = []
        for i in range(self.vertex_candidates.shape[0]):
            if self.vertex_candidates[i, 1] < \
            self.error_lim(self.vertex_candidates[i, 0]):
                vertices.append(self.vertex_candidates[i, 0])
        vertex_candidates, bin_edges = np.histogram(vertices,
                                                    bins=self.bins)
        effs = vertex_candidates/total_output
        stds = effs * np.sqrt((1/vertex_candidates)+(1/total_output))
        return effs, bin_edges, self.name, stds

    def get_error_distr(self, displacement, c=''):
        errors = []
        for i in range(self.vertex_candidates.shape[0]):
            if self.vertex_candidates[i, 0] > displacement:
                errors.append(self.vertex_candidates[i, 1])
        values, bin_edges= np.histogram(errors, bins=self.bins)
        stds = 1/np.sqrt(values)
        return values, bin_edges, self.name, stds

    def scatter_plot_candidates(self, c='k', s=8):
        plt.scatter(self.vertex_candidates[:, 0],
                    self.vertex_candidates[:, 1], c=c, marker='.', s=s,
                    label= self.name)
        plt.xlabel("Decay length (cm)", **hfont)
        plt.ylabel("Vertex Error (cm)", **hfont)

def make_eff_comparison(data_sets):
    fig = plt.figure()
    for i in range(len(data_sets)):
        values, bins, name, stds = data_sets[i].get_efficiencies()
        plt.step(bins[:-1], values, c=color_dict[i],#align='edge', 
            #fill=False, width=1,
            label=name, lw=1, where='post')#, ec=color_dict[i])
        plt.errorbar(bins[:-2]+0.5, values[:-1], yerr=stds[:-1], fmt='none', c=color_dict[i])
    plt.xlabel("Decay length (cm)", **hfont)
    plt.ylabel("Efficiency", **hfont)
    return fig

def make_error_distr_comparison(data_sets, displacement):
    fig = plt.figure()
    for i in range(len(data_sets)):
        values, bins, name, stds = data_sets[i].get_error_distr(displacement)
        plt.step(bins[:-1], values, c=color_dict[i],#align='edge', 
            #fill=False, width=1,
            label=f"{name}, decay length > {displacement} (cm)", lw=1, where='post')#, ec=color_dict[i])
        plt.errorbar(bins[:-2]+0.25, values[:-1], yerr=stds[:-1], fmt='none', c=color_dict[i])
    plt.xlabel("Vertex Error (cm)")
    plt.ylabel("Events")
   # plt.title(f"Error distribution above decay length of {displacement} cm")
    return fig


#data_baseline = np.vstack((none_restr_dl, none_restr_err, none_restr_ntracks)).T
    

#    define cuts
    
uncer_cut = 0.025
uncer_cut_str = "0025"
error_cut = 0.5
error_cut_str = "05"
#############            load CMS Baseline data
data_baseline = np.load('SV_true_pocas_wrtJ_sign_cut_nearest_SV.npy')[25000:40000]

CMS_Baseline = AgentPerformance(data_baseline[:,3], data_baseline[:,4], 
                                np.zeros(data_baseline[:,4].shape[0]), 
                                np.zeros(data_baseline[:,4].shape[0]), 
                                np.zeros(data_baseline[:,4].shape[0]), 
                                "CMS_baseline",error_lim=lambda x: error_cut,
                                uncer_lim=lambda x: uncer_cut)


data_sets = [CMS_Baseline]
for path in paths:
    decay_lengths = np.load(f"{path}/test_decay_length_epo0.npy")
    errors = np.load(f"{path}/test_vertex_error_epo0.npy")
    uncerts = np.load(f"{path}/test_uncertainty_epo0.npy")
    n_used = np.load(f"{path}/test_ntracks_used_epo0.npy")
    n_avail = np.load(f"{path}/test_tracks_avail_epo0.npy")

    data_sets.append(AgentPerformance(decay_lengths, errors, uncerts,
                                      n_used, n_avail, path,
                                      error_lim=lambda x: error_cut,
                                      uncer_lim=lambda x: uncer_cut))

print(data_sets[1].vertex_candidates.shape)

for i in range(len(data_sets[1:])):
    print(data_sets[i+1].name)
    plt.hist(data_sets[i+1].uncertainties, fill=False, label=data_sets[i+1].name,
             ec=color_dict[i], bins=np.arange(0, 0.05, 0.0025), histtype='step')
    plt.yscale('log')
plt.xlabel("Uncertainty (cm$^2$/nPoca)")
plt.legend()
plt.savefig(f"{save_path}uncer_uncer_zoomed_{uncer_cut_str}_error_{error_cut_str}.pdf")
plt.show()



#for i in range(len(data_sets)):
#    plt.hist(data_sets[i].uncertainties, bins=np.arange(0, 1, 0.01))
#    plt.show()

#for i in range(len(data_sets)):
#    plt.hist2d(data_sets[i].uncertainties, data_sets[i].errors,
#               bins=20, range=((0,1), (0,1)))
#    plt.show()

print(sum(data_sets[5].n_tracks_available))

fig1 = make_eff_comparison([data_sets[6], data_sets[3], data_sets[4], data_sets[5]])
plt.yscale('log')
plt.legend()
plt.savefig(f"{save_path}efficiencies_4_uncer_{uncer_cut_str}_error_{error_cut_str}.pdf")
plt.show()

fig2 = make_error_distr_comparison([data_sets[0], data_sets[1], data_sets[2], data_sets[5]], 2)
plt.legend()
plt.ylim((10**0, 10**4))
plt.yscale('log')
plt.savefig(f"{save_path}error_distr_2_uncer_{uncer_cut_str}_error_{error_cut_str}.pdf")
plt.show()

fig2 = make_error_distr_comparison([data_sets[0], data_sets[1], data_sets[2], data_sets[5]], 0)
plt.legend()
plt.yscale('log')
plt.savefig(f"{save_path}error_distr_0_uncer_{uncer_cut_str}_error_{error_cut_str}.pdf")
plt.show()

fig5 = make_error_distr_comparison([data_sets[0], data_sets[1], data_sets[2], data_sets[5]], 10)
plt.ylim((0,120))
plt.legend()
plt.savefig(f"{save_path}error_distr_10_uncer_{uncer_cut_str}_error_{error_cut_str}.pdf")
plt.show()

fig4 = make_eff_comparison([data_sets[0], data_sets[1], data_sets[2], data_sets[5]])
plt.yscale('log')
plt.ylim((10**(-3), 10**(1.1)))
plt.legend()
plt.savefig(f"{save_path}efficiencies_baseline_uncer_{uncer_cut_str}_error_{error_cut_str}.pdf")
plt.show()

for i in range(len(data_sets)):
    continue

#plt.hist(np.load(f"{paths[5]}/test_tracks_used_epo0.npy"), bins=range(9))
#plt.show()
#
#plt.hist(np.load(f"{paths[5]}/test_ntracks_used_epo0.npy"), bins=range(9))
#plt.show()
    
def make_normed_error_distr_comparison(data_sets, displacement, norm_fac):
    fig = plt.figure()
    for i in range(len(data_sets)):
        values, bins, name, stds = data_sets[i].get_error_distr(displacement)
        factor = norm_fac/np.sum(values)
        plt.step(bins[:-1], factor * values, c=color_dict[i],#align='edge', 
            #fill=False, width=1,
            label=f"{name}, decay length > {displacement} (cm)", lw=1, where='post')#, ec=color_dict[i])
        plt.errorbar(bins[:-2]+0.5, factor * values[:-1], yerr=stds[:-1], fmt='none', c=color_dict[i])
    plt.xlabel("Vertex Error (cm)")
    plt.ylabel("Events")
   # plt.title(f"Error distribution above decay length of {displacement} cm")
    return fig

norm_train_restr = np.sum(data_sets[5].get_error_distr(10)[0])

fig6 = make_normed_error_distr_comparison([data_sets[0], data_sets[1], data_sets[2], data_sets[5]], 10, norm_train_restr)
#plt.ylim((0,270))
plt.legend()
plt.savefig(f"{save_path}normed_error_distr_10_uncer_{uncer_cut_str}_error_{error_cut_str}.pdf")
plt.show()

#plt.scatter(data_sets[5].decay_lengths, data_sets[5].errors, c='k', marker='.', s=15)
data_sets[5].scatter_plot_candidates(c='k')
plt.xlim((0,20))
plt.ylim((0,20))
plt.savefig(f"{save_path}scatter_uncer_{uncer_cut_str}_error_{error_cut_str}.pdf")
plt.show()