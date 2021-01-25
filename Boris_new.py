import numpy as np
from analysis.params import ParamRun
from model.profile2D import HOD, Arnaud
from model.power_spectrum import HalomodCorrection
from model.power_spectrum import hm_ang_power_spectrum

fname = "params_dam_wnarrow.yml"
p = ParamRun(fname)
cosmo = p.get_cosmo()
ell = np.arange(6, 2048)

# let's cross-correlate with WIxSC bin 3
for m in p.get("maps"):
    if m["name"] == "wisc3":
        break

g = HOD(nz_file=m["dndz"])
y = Arnaud()

# find appropriate z-range for the N(z)'s
# take the first and last elements of the N(z) func above 0.005
zrange = g.z[g.nzf(g.z) > 0.005].take((0, -1))

# old halo model correction
#hmcorr_HF = HalomodCorrection(cosmo)

######## NEW STUFF ########
# new halo model correction
from scipy.interpolate import interp1d
a_arr = np.linspace(1, 0.5, 16)  # scale factors to sample
# fitting Gaussian parameters:
# a := amplitude // k0 := ref. wavenumber // sf := std
a_bf = np.array([0.17893658, 0.1900067 , 0.201497  , 0.21362104, 0.22650189,
       0.24036806, 0.25538411, 0.27171024, 0.2895082 , 0.30901426,
       0.330462  , 0.35407112, 0.38029703, 0.4094307 , 0.44198723,
       0.47856681])
k0_bf = np.array([0.51832479, 0.50782013, 0.50223392, 0.50036461, 0.501353  ,
       0.50499482, 0.51111962, 0.51970303, 0.5308099 , 0.54475338,
       0.56187687, 0.58258276, 0.60773395, 0.63805158, 0.67477683,
       0.71960561])
sf_bf = np.array([0.31856689, 0.30959041, 0.30381387, 0.30046177, 0.2988959 ,
       0.29884044, 0.30003966, 0.30232376, 0.30553639, 0.30962163,
       0.3144779 , 0.31999348, 0.32607184, 0.33254895, 0.3392506 ,
       0.34598609])
# now, we interpolate these to make them available at any redshift
af = interp1d(a_arr, a_bf, bounds_error=False, fill_value="extrapolate")
k0f = interp1d(a_arr, k0_bf, bounds_error=False, fill_value=1.)
sf = interp1d(a_arr, sf_bf, bounds_error=False, fill_value=1e64)

class HMcorr(object):
    def __init__(self, af, k0f, sf):
        self.af = af
        self.k0f = k0f
        self.sf = sf

    def rk_interp(self, k, a):
        A = af(a)
        k0 = k0f(a)
        s = sf(a)

        k0, s = np.atleast_1d(k0, s)
        k0 = k0[..., None]
        s = s[..., None]

        R = 1 + A*np.exp(-0.5*(np.log10(k/k0)/s)**2)
        return R.squeeze()

hmcorr = HMcorr(af, k0f, sf)
############################


cell_gg = hm_ang_power_spectrum(cosmo, ell, (g,g),
                                zrange=zrange,
                                hm_correction=hmcorr,
                                **m["model"])

cell_gy = hm_ang_power_spectrum(cosmo, ell, (g,y),
                                zrange=zrange,
                                hm_correction=hmcorr,
                                **m["model"])



##### TEST #####
# cell_gg_HF = hm_ang_power_spectrum(cosmo, ell, (g,g),
#                                    zrange=zrange,
#                                    hm_correction=hmcorr_HF,
#                                    **m["model"])
#
# cell_gy_HF = hm_ang_power_spectrum(cosmo, ell, (g,y),
#                                    zrange=zrange,
#                                    hm_correction=hmcorr_HF,
#                                    **m["model"])

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2)
ax[0].set_ylabel("cell")
ax[0].set_xlabel("ell")
ax[1].set_xlabel("ell")
#ax[0].loglog(ell, cell_gg_HF, "k-", lw=3, label="HALOFIT")
ax[0].loglog(ell, cell_gg, "r--", lw=3, label="Gauss")
#ax[1].loglog(ell, cell_gy_HF, "k-", lw=3, label="HALOFIT")
ax[1].loglog(ell, cell_gy, "r--", lw=3, label="Gauss")
ax[0].legend(loc="upper right")
ax[1].legend(loc="upper right")
fig.tight_layout()
fig.savefig("boris_test.pdf")
