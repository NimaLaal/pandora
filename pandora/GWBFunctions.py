import numpy as np
import warnings
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr

try:
    from interpax import interp1d
except ImportError:
    warnings.warn("The package `interpax` is needed for spline interpolation of ORF.")

# jax.config.update('jax_platform_name', 'cpu')
# jax.config.update("jax_enable_x64", False)

fref = 1 / (1 * 365.25 * 24 * 60 * 60)
spl_knts_decent = (
    jnp.array([1e-3, 25.0, 49.3, 82.5, 121.8, 150.0, 180.0]) * jnp.pi / 180.0
)
bins_decent = (
    jnp.array([1e-3, 30.0, 50.0, 80.0, 100.0, 120.0, 150.0, 180.0]) * jnp.pi / 180.0
)

psd_lower_threshold = 1e-18 #No PSD value should go lower than this!

"""
Library for GWB PSD and ORF Functions.
"""


def powerlaw(f, df, log10_A, gamma):
    calc= (
        df
        * 10 ** (2 * log10_A)
        / (12 * jnp.pi**2 * f[:, None] ** 3)
        * (f[:, None] / fref) ** (3 - gamma)
    )
    return jnp.maximum(calc, psd_lower_threshold)


##############################################################################################
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
def turnover(f, df, log10_A, gamma, lf0, kappa, beta):
    hcf = (
        10**log10_A
        * (f[:, None] / fref) ** ((3 - gamma) / 2)
        / (1 + (10**lf0 / f) ** kappa) ** beta
    )
    calc = hcf**2 / 12 / jnp.pi**2 / f[:, None]**3 * df
    return jnp.maximum(calc, psd_lower_threshold)


##############################################################################################
def t_process(f, df, log10_A, gamma, alphas):
    """
    t-process model. PSD  amplitude at each frequency
    is a fuzzy power-law.
    """
    return powerlaw(f, df, log10_A=log10_A, gamma=gamma) * alphas


##############################################################################################
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
    return jnp.maximum(calc, psd_lower_threshold)


##############################################################################################
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
    calc = hcf**2 / 12 / jnp.pi**2 / f[:, None]**3 * df
    return jnp.maximum(calc, psd_lower_threshold)


##############################################################################################
def powerlaw_genmodes(f, df, log10_A, gamma, wgts):
    calc = (
        (10**log10_A) ** 2
        / 12.0
        / jnp.pi**2
        * fref ** (gamma - 3)
        * f[:, None] ** (-gamma)
        * wgts**2
    )
    return jnp.maximum(calc, psd_lower_threshold)


##############################################################################################
def infinitepower(f, df):
    return jnp.full_like(f[:, None], 1e40, dtype="d")


##############################################################################################
def param_hd_orf(angle, a, b, c):
    """
    Pre-factor parametrized Hellings & Downs spatial correlation function.

    :param: a, b, c:
        coefficients of H&D-like curve [default=1.5,-0.25,0.5].

    Reference: Taylor, Gair, Lentati (2013), https://arxiv.org/abs/1210.6014
    Author: S. R. Taylor (2020)

    """
    omc2 = (1 - jnp.cos(angle)) / 2
    params = [a, b, c]
    return params[0] * omc2 * jnp.log(omc2) + params[1] * omc2 + params[2]


##############################################################################################
def spline_orf(angle, params):
    """
    Agnostic spline-interpolated spatial correlation function. Spline knots
    are placed at edges, zeros, and minimum of H&D curve. Changing locations
    will require manual intervention to create new function.

    :param: params
        spline knot amplitudes.

    Reference: Taylor, Gair, Lentati (2013), https://arxiv.org/abs/1210.6014
    Author: S. R. Taylor (2020)

    """
    # spline knots placed at edges, zeros, and minimum of H&D

    omc2_knts = (1 - jnp.cos(spl_knts_decent)) / 2
    omc2 = (1 - jnp.cos(angle)) / 2
    # return jnp.interp(x = omc2, xp = omc2_knts, fp = params, left=None, right=None, period=None)
    interp1d(xq=omc2, xp=omc2_knts, fp=params, method="cubic")


##############################################################################################
def bin_orf(angle, params):
    """
    Agnostic binned spatial correlation function. Bin edges are
    placed at edges and across angular separation space. Changing bin
    edges will require manual intervention to create new function.

    :param: params
        inter-pulsar correlation bin amplitudes.

    Author: S. R. Taylor (2020)

    """
    return params[jnp.digitize(angle, bins_decent) - 1]


##############################################################################################
def freq_hd(angle, params):
    """
    Frequency-dependent Hellings & Downs spatial correlation function.
    Implemented as a model that only enforces H&D inter-pulsar correlations
    after a certain number of frequencies in the spectrum. The first set of
    frequencies are uncorrelated.

    :param: params
        params[0] is the number of components in the stochastic process.
        params[1] is the frequency at which to start the H&D inter-pulsar
        correlations (indexing from 0).

    Reference: Taylor et al. (2017), https://arxiv.org/abs/1606.09180
    Author: S. R. Taylor (2020)

    """
    nfreq = params[0]
    orf_ifreq = params[1]
    omc2 = (1 - jnp.cos(angle)) / 2
    hd_coeff = (1.5 * omc2 * jnp.log(omc2) - 0.25 * omc2 + 0.5) * jnp.ones(2 * nfreq)
    return hd_coeff.at[: 2 * orf_ifreq].set(0.0)


##############################################################################################
def hd_orf(angle, *args):
    """HD correlation function."""
    return 3/2*( (1/3 + ((1-jnp.cos(angle))/2) * (jnp.log((1-jnp.cos(angle))/2) - 1/6)))


##############################################################################################
def dipole_orf(angle, *args):
    """Dipole spatial correlation function."""
    return jnp.cos(angle)


##############################################################################################
def monopole_orf(angle, *args):
    """Monopole spatial correlation function."""
    return jnp.ones_like(angle)


##############################################################################################
def gw_monopole_orf(angle, *args):
    """
    GW-monopole Correlations. This phenomenological correlation pattern can be
    used in Bayesian runs as the simplest type of correlations.
    Author: N. Laal (2020)
    """
    return jnp.ones_like(angle) * 0.5


##############################################################################################
def gw_dipole_orf(angle, *args):
    """
    GW-dipole Correlations.
    Author: N. Laal (2020)
    """
    return 1 / 2 * jnp.cos(angle)


##############################################################################################
def st_orf(angle, *args):
    """
    Scalar tensor correlations as induced by the breathing polarization mode of gravity.
    Author: N. Laal (2020)
    """
    return 1 / 8 * (3.0 + jnp.cos(angle))


##############################################################################################
def gt_orf(angle, tau):
    """
    General Transverse (GT) Correlations. This ORF is used to detect the relative
    significance of all possible correlation patterns induced by the most general
    family of transverse gravitational waves.

    :param: tau
        tau = 1 results in ST correlations while tau = -1 results in HD correlations.

    Author: N. Laal (2020)

    """
    k = 1 / 2 * (1 - jnp.cos(angle))
    return 1 / 8 * (3 + jnp.cos(angle)) + (1 - tau) * 3 / 4 * k * jnp.log(k)


##############################################################################################
def legendre_orf(leg_pol_values, params):
    """
    Legendre polynomial spatial correlation function (up to 10th order). Assumes process
    normalization such that autocorrelation signature is 1. A separate function
    is needed to use a "split likelihood" model with this Legendre process
    decoupled from the autocorrelation signature ("zero_diag_legendre_orf").

    :param: params
        Legendre polynomial amplitudes describing the Legendre series approximation
        to the inter-pulsar correlation signature.
        H&D coefficients are a_0=0, a_1=0, a_2=0.3125, a_3=0.0875, ...

    Reference: Gair et al. (2014), https://arxiv.org/abs/1406.4664
    """
    # x = jnp.cos(angle)
    # lmax = params.shape[0]
    # calc = jnp.array([\
    # 1.0,
    # x,
    # (1/2)*(3*x**2 - 1),
    # (1/2)*(5*x**3 - 3*x),
    # (1/8)*(35*x**4 - 30*x**2 + 3),
    # (1/8)*(63*x**5 - 70*x**3 + 15*x),
    # (1/16)*(231*x**6 - 315*x**4 + 105*x**2 - 5),
    # (1/16)*(343*x**7 - 693*x**5 + 315*x**3 - 35*x),
    # (1/128)*(6435*x**8 - 12012*x**6 + 6930*x**4 - 1260*x**2 + 35),
    # (1/128)*(12155*x**9 - 25740*x**7 + 18018*x**5 - 4620*x**3 + 315*x)])
    return jnp.sum(params * leg_pol_values[: params.shape[0]])


##############################################################################################
def zero_orf(angle, *args):
    """No correlation."""
    return jnp.zeros(angle)
