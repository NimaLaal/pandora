import numpy as np
import warnings
import os
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
from functools import lru_cache, partial
try:
    from interpax import interp1d
except ImportError:
    pass
    # warnings.warn("The package `interpax` is needed for spline interpolation of ORF.")

########################################################################################
fref = 1 / (1 * 365.25 * 24 * 60 * 60)

spl_knts_decent = (
    jnp.array([1e-3, 25.0, 49.3, 82.5, 121.8, 150.0, 180.0]) * jnp.pi / 180.0
)

bins_decent = (
    jnp.array([1e-3, 30.0, 50.0, 80.0, 100.0, 120.0, 150.0, 180.0]) * jnp.pi / 180.0
)

psd_low_threshold = os.environ.get("pandora_low_GWB_psd_threshold", 1e-18)

def clip_psd(psd, threshold = psd_low_threshold):
    '''
    Sets the lowest possible value for PSD.
    PSD values below the `threshold` are set 
    to the `threshold`.
    '''
    return jnp.maximum(psd, threshold)
########################################################################################

"""
Library for GWB PSD and ORF Functions.
"""
@jax.jit
def powerlaw(f, df, log10_A, gamma):
    calc= (
        df
        * 10 ** (2 * log10_A)
        / (12 * jnp.pi**2 * f[:, None] ** 3)
        * (f[:, None] / fref) ** (3 - gamma)
    )
    return clip_psd(calc)


##############################################################################################
@jax.jit
def free_spectrum(f, df, *halflog10_rho):
    """
    Free spectral model. PSD  amplitude at each frequency
    is a free parameter. Model is parameterized by
    S(f_i) = \rho_i^2 * T,
    where \rho_i is the free parameter and T is the observation
    length.
    """
    return 10 ** (2 * jnp.array(halflog10_rho))[:, None]


##############################################################################################
@jax.jit
def turnover(f, df, log10_A, gamma, lf0, kappa, beta):
    hcf = (
        10**log10_A
        * (f[:, None] / fref) ** ((3 - gamma) / 2)
        / (1 + (10**lf0 / f) ** kappa) ** beta
    )
    calc = hcf**2 / 12 / jnp.pi**2 / f[:, None]**3 * df
    return clip_psd(calc)


##############################################################################################
@jax.jit
def t_process(f, df, log10_A, gamma, alphas):
    """
    t-process model. PSD  amplitude at each frequency
    is a fuzzy power-law.
    """
    calc= (
        df
        * 10 ** (2 * log10_A)
        / (12 * jnp.pi**2 * f[:, None] ** 3)
        * (f[:, None] / fref) ** (3 - gamma)
    )
    return clip_psd(calc * alphas) 


##############################################################################################
@jax.jit
def turnover_knee(f, df, log10_A, gamma, lfb, lfk, kappa, delta):
    """
    Generic turnover spectrum with a high-frequency knee.
    :param f: sampling frequencies of GWB
    :param A: characteristic strain amplitude at f=1/yr
    :param gamma: negative slope of PSD around f=1/yr (usually 13/3)
    :param lfb: log10 transition frequency at which environment dominates GWs
    :param lfk: log10 knee frequency due to population finiteness
    :param kappa: smoothness of turnover (10/3 for 3-body stellar scattering)
    :param delta: slope at higher frequencies
    """
    hcf = (
        10**log10_A
        * (f[:, None] / fref) ** ((3 - gamma) / 2)
        * (1.0 + (f[:, None] / 10**lfk)) ** delta
        / jnp.sqrt(1 + (10**lfb / f) ** kappa)
    )
    calc = hcf**2 / 12 / jnp.pi**2 / f[:, None]**3 * df
    return clip_psd(calc)


##############################################################################################
@jax.jit
def broken_powerlaw(f, df, log10_A, gamma, delta, log10_fb, kappa):
    """
    Generic broken powerlaw spectrum.
    :param f: sampling frequencies
    :param A: characteristic strain amplitude [set for gamma at f=1/yr]
    :param gamma: negative slope of PSD for f > f_break [set for comparison
        at f=1/yr (default 13/3)]
    :param delta: slope for frequencies < f_break
    :param log10_fb: log10 transition frequency at which slope switches from
        gamma to delta
    :param kappa: smoothness of transition (Default = 0.1)
    """
    hcf = (
        10**log10_A
        * (f[:, None] / fref) ** ((3 - gamma) / 2)
        * (1 + (f[:, None] / 10**log10_fb) ** (1 / kappa)) ** (kappa * (gamma - delta) / 2)
    )
    return hcf**2 / 12 / jnp.pi**2 / f[:, None]**3 * df


##############################################################################################
@jax.jit
def powerlaw_genmodes(f, df, log10_A, gamma, wgts):
    calc = (
        (10**log10_A) ** 2
        / 12.0
        / jnp.pi**2
        * fref ** (gamma - 3)
        * f[:, None] ** (-gamma)
        * wgts**2
    )
    return clip_psd(calc)


##############################################################################################
@jax.jit
def infinitepower(f, df):
    return jnp.full_like(f[:, None], 1e40)


##############################################################################################
@jax.jit
def param_hd_orf(angle, a, b, c):
    """
    Pre-factor parametrized Hellings & Downs spatial correlation function.

    :param: a, b, c:
        coefficients of H&D-like curve [default=1.5,-0.25,0.5].

    Reference: Taylor, Gair, Lentati (2013), https://arxiv.org/abs/1210.6014
    Author: S. R. Taylor (2020)

    """
    omc2 = (1 - jnp.cos(angle)) / 2
    return a * omc2 * jnp.log(omc2) + b * omc2 + c


##############################################################################################
@jax.jit
def spline_orf(angle, b1, b2, b3, b4, b5, b6, b7):
    """
    Agnostic spline-interpolated spatial correlation function. Bin edges are
    placed at edges and across angular separation space.
    Note, the bin edges are at 1e-3, 30.0, 50.0, 80.0, 100.0, 120.0, 
    150.0, and 180.0 degrees. You need to edit this function and the global 
    constant `bins_decent` manually if you want to have different bin edges.

    :param: params
        spline knot amplitudes.

    Reference: Taylor, Gair, Lentati (2013), https://arxiv.org/abs/1210.6014
    Author: S. R. Taylor (2020)

    """
    # spline knots placed at edges, zeros, and minimum of H&D

    omc2_knts = (1 - jnp.cos(spl_knts_decent)) / 2
    omc2 = (1 - jnp.cos(angle)) / 2
    interp1d(xq=omc2, xp=omc2_knts, fp=jnp.array([b1, b2, b3, b4, b5, b6, b7]), method="cubic")


##############################################################################################
@jax.jit
def bin_orf(angle, b1, b2, b3, b4, b5, b6, b7):
    """
    Agnostic binned spatial correlation function. Bin edges are
    placed at edges and across angular separation space.
    Note, the bin edges are at 1e-3, 30.0, 50.0, 80.0, 100.0, 120.0, 
    150.0, and 180.0 degrees. You need to edit this function and the global 
    constant `bins_decent` manually if you want to have different bin edges.

    :params: b1, b2, b3, b4, b5, b6, b7
        inter-pulsar correlation bin amplitudes.

    :ref: 
        Taylor et al. (2017), https://arxiv.org/abs/1606.09180

    Author: S. R. Taylor (2020)

    """
    # TODO: try to cache this. Though, this model is very rarely used!
    # I am not sure if it deserves its own unique treatment!
    chosen_indices = jnp.digitize(angle, bins_decent) - 1
    #chosen_indices are indices between 0 and 6 (inclusive) with 
    #the shape equal to the number of unique pulsar pairs
    return jnp.array([b1, b2, b3, b4, b5, b6, b7])[chosen_indices]


##############################################################################################
@jax.jit
def hd_orf(angle):
    """HD correlation function."""
    return 3/2*( (1/3 + ((1-jnp.cos(angle))/2) * (jnp.log((1-jnp.cos(angle))/2) - 1/6)))


##############################################################################################
@jax.jit
def dipole_orf(angle):
    """Dipole spatial correlation function."""
    return jnp.cos(angle)


##############################################################################################
@jax.jit
def monopole_orf(angle):
    """Monopole spatial correlation function."""
    return jnp.ones_like(angle)


##############################################################################################
@jax.jit
def gw_monopole_orf(angle):
    """
    GW-monopole Correlations. This phenomenological correlation pattern can be
    used in Bayesian runs as the simplest type of correlations.
    Author: N. Laal (2020)
        
    :ref:
        https://iopscience.iop.org/article/10.3847/2041-8213/ac401c
    """
    return 0.5 * jnp.ones_like(angle)


##############################################################################################
@jax.jit
def gw_dipole_orf(angle):
    """
    GW-dipole Correlations.
    
    :ref:
        https://iopscience.iop.org/article/10.3847/2041-8213/ac401c

    Author: N. Laal (2020)
    """
    return 0.5 * jnp.cos(angle)


##############################################################################################
@jax.jit
def st_orf(angle):
    """
    Scalar tensor correlations as induced by the breathing polarization mode of gravity.

    :ref:
        https://iopscience.iop.org/article/10.3847/2041-8213/ac401c
    """
    return 0.125 * (3.0 + jnp.cos(angle))


##############################################################################################
@jax.jit
def gt_orf(angle, tau):
    """
    General Transverse (GT) Correlations. This ORF is used to detect the relative
    significance of all possible correlation patterns induced by the most general
    family of transverse gravitational waves.

    :param: tau
        tau = 1 results in ST correlations while tau = -1 results in HD correlations.

    :ref:
        https://ir.library.oregonstate.edu/concern/graduate_thesis_or_dissertations/4j03d741d?locale=en

    Author: N. Laal (2020)

    """
    k = 1 / 2 * (1 - jnp.cos(angle))
    return 1 / 8 * (3 + jnp.cos(angle)) + (1 - tau) * 3 / 4 * k * jnp.log(k)

##############################################################################################
@jax.jit
def zero_orf(angle):
    """No correlation."""
    return jnp.zeros(angle)
