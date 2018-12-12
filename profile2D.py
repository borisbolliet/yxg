"""
"""

import numpy as np
from numpy.linalg import lstsq
from scipy.special import sici
from scipy.special import erf
from scipy.integrate import quad
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

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import pyccl as ccl
    >>> cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045,
                              h=0.67, A_s=2.1e-9, n_s=0.96)
    >>> p1 = Arnaud()
    >>> # radial profile is the product of the normalisation and the form factor
    >>> x = np.linspace(1e-3, 2, 100)  # R/R_Delta
    >>> radial_profile = p1.norm(cosmo, M=1e+14, a=0.7) * p1.form_factor(x)
    >>> plt.loglog(x, radial_profile)  # plot profile as a function of radius

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import pyccl as ccl
    >>> cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045,
                              h=0.67, A_s=2.1e-9, n_s=0.96)
    >>> p2 = Arnaud()
    >>> # plot the profile in fourier space
    >>> k = np.logspace(-1, 1, 100)  # wavenumber
    >>> U = p2.fourier_profile(cosmo, k, M=1e+14, a=0.6)
    >>> plt.loglog(k, U)  # plot profile in fourier space
    """
    def __init__(self, rrange=(1e-3, 10), qpoints=1e2):

        self.rrange = rrange         # range of probed distances [R_Delta]
        self.qpoints = int(qpoints)  # no of sampling points
        self.Delta = 500             # reference overdensity (Arnaud et al.)
        self.kernel = kernel.y       # associated window function

        self._fourier_interp = self._integ_interp()


    def norm(self, cosmo, M, a, b=0.4):
        """Computes the normalisation factor of the Arnaud profile.

        .. note:: Normalisation factor is given in units of ``eV/cm^3``. \
        (Arnaud et al., 2009)
        """
        aP = 0.12  # Arnaud et al.
        h70 = cosmo["h"]/0.7
        P0 = 6.41 # reference pressure

        K = 1.65*h70**2*P0 * (h70/3e14)**(2./3.+aP)  # prefactor

        Pz = ccl.h_over_h0(cosmo, a)**(8./3.)  # scale factor (z) dependence
        PM = (M*(1-b))**(2./3.+aP)             # mass dependence
        P = K*Pz*PM
        return P


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
        integrand = lambda x: self.form_factor(x)*x

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
        F1 = lambda x: f_arr[0]*np.ones_like(x)  # constant value

        # Forward Extrapolation
        # linear fitting
        Q = np.log10(q_arr[q_arr > 1e2])
        F = np.log10(f_arr[q_arr > 1e2])
        A = np.vstack([Q, np.ones(len(Q))]).T
        m, c = lstsq(A, F, rcond=None)[0]

        F3 = lambda x: 10**(m*x+c) # logarithmic drop

        F = lambda x: np.piecewise(x,
                                   [x < lgqmin,        # backward extrapolation
                                    (lgqmin <= x)*(x <= lgqmax),  # common range
                                    lgqmax < x],       # forward extrapolation
                                    [F1, F2, F3])

        return F


    def fourier_profile(self, cosmo, k, M, a, b=0.4):
        """Computes the Fourier transform of the Arnaud profile.

        .. note:: Output units are ``[norm] Mpc^3``
        """
        R = ct.R_Delta(cosmo, M, a, self.Delta) / a  # R_Delta*(1+z) [Mpc]
        F = self.norm(cosmo, M, a, b) * self._fourier_interp(np.log10(k*R))
        return 4*np.pi * R**3 * F



class NFW(object):
    """Calculate a Navarro-Frenk-White profile quantity of a halo and its
    Fourier transform.
    """
    def __init__(self):

        self.kernel = kernel.g  # associated window function


    def norm(self, cosmo, M, a, Delta=500):
        """Computes the normalisation factor of the Navarro-Frenk-White profile.

        .. note:: Normalisation factor is given in units of ``M_sun/Mpc^3``.
        """
        rho = ccl.rho_x(cosmo, a, "matter")

        # Halo Concentration Handling
        c = ct.concentration_duffy(M, a, is_D500=(Delta==500))
        if (Delta != 200) and (Delta != 500):
            raise ValueError("Concentration not implemented for Delta=%d." % Delta)

        P = Delta/3 * rho * c**3 / (np.log(1+c)-c/(1+c))
        return P


    def form_factor(self, cosmo, x, M, a, Delta=500):
        """Computes the form factor of the Navarro-Frenk-White profile."""
        c = ccl.halo_concentration(cosmo, M, a, Delta)
        P = 1/(x*c*(1+x*c)**2)
        return P


    def fourier_profile(self, cosmo, k, M, a, Delta=500):
        """Computes the Fourier transform of the Navarro-Frenk-White profile."""
        # Halo Concentration Handling
        c = ct.concentration_duffy(M, a, is_D500=(Delta==500))
        if (Delta != 200) and (Delta != 500):
            raise ValueError("Concentration not implemented for Delta=%d." % Delta)

        x = k*ct.R_Delta(cosmo, M, a, Delta)/c

        Si1, Ci1 = sici((1+c)*x)
        Si2, Ci2 = sici(x)

        P1 = (np.log(1+c) - c/(1+c))**-1
        P2 = np.sin(x)*(Si1-Si2) + np.cos(x)*(Ci1-Ci2)
        P3 = np.sin(c*x)/((1+c)*x)

        F = P1*(P2-P3)
        return F



class HOD(object):
    """Calculate a Halo Occupation Distribution profile quantity of a halo."""
    def __init__(self):

        self.kernel = kernel.g


    def fourier_profile(self, cosmo, k, M, a, Delta=500):
        """Computes the Fourier transform of the Halo Occupation Distribution."""
        # HOD model (Krause & Eifler, 2014)
        Mmin = 10**12.1
        M1 = 10**13.65
        M0 = 10**12.2
        sigma_lnM = 10**0.4
        alpha_sat = 1.0
        fc = 0.25

        # HOD Model
        Nc = 0.5 * (1 + erf((np.log(M/Mmin))/sigma_lnM))
        Ns = np.heaviside(M-M0, 0.5) * ((M-M0)/M1)**alpha_sat

        H = NFW().fourier_profile(cosmo, k, M, a, Delta)

        return Nc * (fc + Ns*H)



class kernel(object):
    """Window function definitions.

    This class contains definitions for all used window functions (kernels)
    for computation of the angular power spectrum. Multiplying the window
    function with its corresponding profile normalisation factor yields units
    of ``1/L``.
    """
    def y(cosmo, a):
        """The thermal Sunyaev-Zel'dovich anisotropy window function."""
#        sigma = v("Thomson cross section")
#        prefac = sigma/(u.m_e*u.c**2)
#        # normalisation
#        J2eV = 1/v("electron volt")
#        cm2m = u.centi
#        m2Mpc = 1/(u.mega*u.parsec)
#        unit_norm = J2eV * cm2m**3 * m2Mpc
#        prefac/=unit_norm
        prefac = 4.017100792437957e-06 # avoid recomputing every time
        return prefac*a


    def g(cosmo, a):
        """The galaxy number overdensity window function."""
        unit_norm = 1/(u.c/u.kilo)  # [s/km]
        Hz = ccl.h_over_h0(cosmo, a)*cosmo["H0"]  # [km/(s Mpc)]

        return Hz*unit_norm * ct.dNdz(a)
