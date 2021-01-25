import numpy as np
from analysis.params import ParamRun
from model.profile2D import HOD, Arnaud
from model.power_spectrum import hm_ang_power_spectrum
from model.power_spectrum import HalomodCorrection

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

# halo model correction
hmcorr = HalomodCorrection(cosmo)

cell_gg = hm_ang_power_spectrum(cosmo, ell, (g,g),
                                zrange=zrange,
                                hm_correction=hmcorr,
                                **m["model"])

cell_gy = hm_ang_power_spectrum(cosmo, ell, (g,y),
                                zrange=zrange,
                                hm_correction=hmcorr,
                                **m["model"])
print(cell_gg)
print(cell_gy)
