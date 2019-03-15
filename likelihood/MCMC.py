import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
import pyccl as ccl
import fittingtools as ft



keyword = ""  # data identification keyword (add "_" for readability)

# SURVEY PROPERTIES #
dir1 = "../analysis/data/dndz/"
wisc = ["wisc_b%d" % i for i in range(1, 6)]
surveys = ["2mpz"] + wisc
bins = np.append([1], np.arange(1, 6))
sprops = ft.survey_properties(dir1, surveys, bins)

cosmo = ccl.Cosmology(Omega_c=0.26066676,
                      Omega_b=0.048974682,
                      h=0.6766,
                      sigma8=0.8102,
                      n_s=0.9665)

# PARAMETER : [VALUE, STATUS, CONSTRAINTS]
# (free : 0) ::: (fixed : 1) ::: (coupled : -N)
priors = {"Mmin"       :  [11.99,   -1,   (10, 16)],
          "M0"         :  [11.99,   -1,   (10, 16)],
          "M1"         :  [13.18,   0,   (10, 16)],
          "sigma_lnM"  :  [0.26,    1,   (0.1, 1.0)],
          "alpha"      :  [1.43,    1,   (0.5, 1.5)],
          "fc"         :  [0.54,    1,   (0.1, 1.0)],
          "bg"         :  [1.0,     1,   (0, np.inf)],
          "bmax"       :  [1.0,     1,   (0, np.inf)],
          "b_hydro"    :  [0.45,    0,   (0.1, 0.9)]}


nwalkers, nsteps = 60, 50
def sampler(sur): return ft.MCMC(sur, sprops, cosmo, priors, nwalkers, nsteps)
results = Pool().map(sampler, list(sprops.keys()))