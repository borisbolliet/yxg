import numpy as np
from numpy.linalg import lstsq
from scipy.special import sici
from scipy.special import erf
from scipy.integrate import quad
from scipy.integrate import simps
from scipy.interpolate import interp1d
import scipy.constants as u
from scipy.constants import value as v
import pyccl as ccl

import cosmotools as ct



class Arnaud(object):
    """
    Calculate an Arnaud profile quantity of a halo and its Fourier transform.


    Parameters
    ----------
    rrange : tuple
        Desired physical distance to probe (expressed in units of R_Delta).
        Change only if necessary. For distances too much outside of the
        default range the calculation might become unstable.
    qpoints : int
        Number of integration sampling points.
    """
    def __init__(self, rrange=(1e-3, 10), qpoints=1e2):

        self.rrange = rrange         # range of probed distances [R_Delta]
        self.qpoints = int(qpoints)  # no of sampling points
        self.Delta = 500             # reference overdensity (Arnaud et al.)

        self._fourier_interp = self._integ_interp()


    def kernel(self, cosmo, a, **kwargs):
        """The thermal Sunyaev-Zel'dovich anisotropy window function."""
        prefac = 4.017100792437957e-06  # avoid recomputing every time
        return prefac*a


    def profnorm(self, cosmo, a, squeeze=True, **kwargs):
        """Computes the overall profile normalisation for the angular cross-
        correlation calculation."""
        return np.ones_like(a)


    def norm(self, cosmo, M, a, b, squeeze=True):
        """Computes the normalisation factor of the Arnaud profile.

        .. note:: Normalisation factor is given in units of ``eV/cm^3``. \
        (Arnaud et al., 2009)
        """
        # Input handling
        M, a = np.atleast_1d(M), np.atleast_1d(a)

        aP = 0.12  # Arnaud et al.
        h70 = cosmo["h"]/0.7
        P0 = 6.41 # reference pressure

        K = 1.65*h70**2*P0 * (h70/3e14)**(2/3+aP)  # prefactor

        PM = (M*(1-b))**(2/3+aP)             # mass dependence
        Pz = ccl.h_over_h0(cosmo, a)**(8/3)  # scale factor (z) dependence

        P = K * PM[..., None] * Pz
        return P.squeeze() if squeeze else P


    def form_factor(self, x):
        """Computes the form factor of the Arnaud profile."""
        # Planck collaboration (2013a) best fit
        c500 = 1.81
        alpha = 1.33
        beta = 4.13
        gama = 0.31

        f1 = (c500*x)**(-gama)
        f2 = (1+(c500*x)**alpha)**(-(beta-gama)/alpha)
        return f1*f2


    def _integ_interp(self):
        """Computes the integral of the power spectrum at different points and
        returns an interpolating function connecting these points.
        """
        def integrand(x): return self.form_factor(x)*x

        ## Integration Boundaries ##
        rmin, rmax = self.rrange  # physical distance [R_Delta]
        lgqmin, lgqmax = np.log10(1/rmax), np.log10(1/rmin)  # log10 bounds

        q_arr = np.logspace(lgqmin, lgqmax, self.qpoints)
        f_arr = np.array([quad(integrand,
                               a=1e-4, b=np.inf,     # limits of integration
                               weight="sin", wvar=q  # fourier sine weight
                               )[0] / q for q in q_arr])

        F2 = interp1d(np.log10(q_arr), np.array(f_arr), kind="cubic")

        ## Extrapolation ##
        # Backward Extrapolation
        def F1(x): return f_arr[0]*np.ones_like(x)  # constant value

        # Forward Extrapolation
        # linear fitting
        Q = np.log10(q_arr[q_arr > 1e2])
        F = np.log10(f_arr[q_arr > 1e2])
        A = np.vstack([Q, np.ones(len(Q))]).T
        m, c = lstsq(A, F, rcond=None)[0]

        def F3(x): return 10**(m*x+c)  # logarithmic drop

        def F(x): return np.piecewise(x,
                                     [x < lgqmin,        # backward extrapolation
                                     (lgqmin <= x)*(x <= lgqmax),  # common range
                                     lgqmax < x],       # forward extrapolation
                                     [F1, F2, F3])
        return F


    def fourier_profiles(self, cosmo, k, M, a, squeeze=True, **kwargs):
        """Computes the Fourier transform of the Arnaud profile.

        .. note:: Output units are ``[norm] Mpc^3``
        """
        # Input handling
        M, a, k = np.atleast_1d(M), np.atleast_1d(a), np.atleast_2d(k)

        b = kwargs["b_hydro"]  # hydrostatic bias
        R = ct.R_Delta(cosmo, M, a, self.Delta, squeeze=False) / a  # R_Delta*(1+z)
        R = R[..., None]  # transform axes

        ff = self._fourier_interp(np.log10(k*R))
        nn = self.norm(cosmo, M, a, b)[..., None]

        F = 4*np.pi*R**3 * nn * ff
        return (F.squeeze(), (F**2).squeeze()) if squeeze else (F, F**2)



class NFW(object):
    """Calculate a Navarro-Frenk-White profile quantity of a halo and its
    Fourier transform.
    """
    def __init__(self, kernel=None):

        self.Delta = 500    # reference overdensity (Arnaud et al.)
        self.kernel = kernel  # associated window function


    def profnorm(self, cosmo, a, squeeze=True, **kwargs):
        """Computes the overall profile normalisation for the angular
        cross-correlation calculation."""
        return np.ones_like(a)


    def norm(self, cosmo, M, a, squeeze=True):
        """Computes the normalisation factor of the Navarro-Frenk-White profile.
        """
        # Input handling
        M, a = np.atleast_1d(M), np.atleast_1d(a)
        norm = np.ones((len(M), len(a)))
        return norm.squeeze() if squeeze else norm


    def form_factor(self, cosmo, x, M, a, squeeze=True):
        """Computes the Navarro-Frenk-White profile.

        .. note:: Normalisation factor is given in units of ``M_sun/Mpc^3``.
        """
        # Input handling
        M, a, x = np.atleast_1d(M), np.atleast_1d(a), np.atleast_2d(x)

        rho = ccl.rho_x(cosmo, a, "matter")
        c = ct.concentration_duffy(M, a, is_D500=True, squeeze=False)

        P1 = self.Delta/3 * rho * c**3 / (np.log(1+c)-c/(1+c))
        P2 = 1/(x*c[..., None] * (1+x*c[..., None])**2)
        return (P1[..., None]*P2).squeeze() if squeeze else P1[..., None]*P2


    def fourier_profiles(self, cosmo, k, M, a, squeeze=True, **kwargs):
        """Computes the Fourier transform of the Navarro-Frenk-White profile."""
        # Input handling
        M, a, k = np.atleast_1d(M), np.atleast_1d(a), np.atleast_2d(k)

        c = ct.concentration_duffy(M, a, is_D500=True, squeeze=False)

        R = ct.R_Delta(cosmo, M, a, self.Delta, is_matter=False, squeeze=False)/(c*a)
        x = k*R[..., None]

        c = c[..., None]
        Si1, Ci1 = sici((1+c)*x)
        Si2, Ci2 = sici(x)

        P1 = 1/(np.log(1+c) - c/(1+c))
        P2 = np.sin(x)*(Si1-Si2) + np.cos(x)*(Ci1-Ci2)
        P3 = np.sin(c*x)/((1+c)*x)

        F = P1*(P2-P3)
        return (F.squeeze(), (F**2).squeeze()) if squeeze else (F, F**2)


    def four(self, cosmo, k, M, a, squeeze=True, **kwargs):
        """Computes a modified version of the NFW profile."""
        # Input handling
        M, a, k = np.atleast_1d(M), np.atleast_1d(a), np.atleast_2d(k)

        c = ct.concentration_duffy(M, a, is_D500=True, squeeze=False)

        rs = 1
        rmax = 1

        Si1, Ci1 = sici(k*rs)
        Si2, Ci2 = sici(k*(rs+rmax))

        P1 = k*(rs+rmax) * (np.sin(k*rs)*(Si1-Si2) + np.cos(k*rs)*(Ci1-Ci2))
        P2 = np.sin(k*rmax)
        P3 = k*(rmax+(rs+rmax)*np.log(rs/(rs+rmax)))

        F = (P1+P2)/P3
        return (F.squeeze(), (F**2).squeeze()) if squeeze else (F, F**2)




class HOD(object):
    """Calculates a Halo Occupation Distribution profile quantity of a halo."""
    def __init__(self, nz_file=None):

        self.Delta = 500  # reference overdensity (Arnaud et al.)
        z, nz = np.loadtxt(nz_file, unpack=True)
        self.nzf = interp1d(z, nz, bounds_error=False, fill_value=0)


    def kernel(self, cosmo, a):
        """The galaxy number overdensity window function."""
        unit_norm = 3.3356409519815204e-04  # 1/c
        Hz = ccl.h_over_h0(cosmo, a)*cosmo["h"]
        return Hz*unit_norm * self.nzf(1/a - 1)


    def profnorm(self, cosmo, a, squeeze=True, **kwargs):
        """Computes the overall profile normalisation for the angular cross-
        correlation calculation."""
        # Input handling
        a = np.atleast_1d(a)

        # extract parameters
        Mmin = 10**kwargs["Mmin"]
        M0 = 10**kwargs["M0"]
        M1 = 10**kwargs["M1"]
        sigma_lnM = kwargs["sigma_lnM"]
        alpha = kwargs["alpha"]
        fc = kwargs["fc"]

        logMmin, logMmax = (6, 17) # log of min and max halo mass [Msun]
        mpoints = int(64)         # number of integration points
        M = np.logspace(logMmin, logMmax, mpoints)  # masses sampled

        Dm = self.Delta/ccl.omega_x(cosmo, a, "matter")  # CCL uses delta_matter
        mfunc = [ccl.massfunc(cosmo, M, A1, A2) for A1, A2 in zip(a, Dm)]

        Nc = 0.5 * (1 + erf((np.log10(M/Mmin))/sigma_lnM))  # centrals
        Ns = ((M-M0)*np.heaviside(M-M0, 0) / M1)**alpha     # satellites

        dng = mfunc*Nc*(fc+Ns)  # integrand

        ng = simps(dng, x=np.log10(M))
        return ng.squeeze() if squeeze else ng


    def fourier_profiles(self, cosmo, k, M, a, squeeze=True, **kwargs):
        """Computes the Fourier transform of the Halo Occupation Distribution.
        Default parameter values from Krause & Eifler (2014).
        """
        # Input handling
        M, a, k = np.atleast_1d(M), np.atleast_1d(a), np.atleast_2d(k)

        # extract parameters
        Mmin = 10**kwargs["Mmin"]
        M0 = 10**kwargs["M0"]
        M1 = 10**kwargs["M1"]
        sigma_lnM = kwargs["sigma_lnM"]
        alpha = kwargs["alpha"]
        fc = kwargs["fc"]

        # HOD Model
        Nc = 0.5 * (1 + erf((np.log10(M/Mmin))/sigma_lnM))  # centrals
        Ns = ((M-M0)*np.heaviside(M-M0, 0) / M1)**alpha     # satellites
        Nc, Ns = Nc[..., None, None], Ns[..., None, None]

        H, _ = NFW().fourier_profiles(cosmo, k, M, a)

        F, F2 = Nc*(fc + Ns*H), Nc*(2*fc*Ns*H + (Ns*H)**2)
        return (F.squeeze(), F2.squeeze()) if squeeze else F, F2