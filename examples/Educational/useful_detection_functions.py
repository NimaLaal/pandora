import numpy as np

yr_to_sec = 86400 * 365.25
ref_f = 1/(yr_to_sec)

def Mmat(toas):
    '''
    Simplest timing design-matrix
    '''
    DesignM = np.ones((len(toas), 3))
    DesignM[:,1] = toas
    DesignM[:,2] = DesignM[:,1]**2
    return DesignM

def Rmat(n_obs, DesignM):
    '''
    The Rmatrix used to do the quadratic fitting
    '''
    I = np.identity(n_obs)
    return I - np.einsum('kn,nm,pm->kp', DesignM, np.linalg.inv(np.einsum('km,kn->nm',DesignM,DesignM)),DesignM)

def Fmat(self, freqs, toas):
    '''
    The 'F-matrix' used to do a discrete Fourier transform

    Author: Nima Laal
    '''
    nmodes = len(freqs)
    N = len(toas)
    F = np.zeros((N, 2 * nmodes))
    F[:, 0::2] = np.sin(2 * np.pi * toas[:, None] * freqs[None, :])
    F[:, 1::2] = np.cos(2 * np.pi * toas[:, None] * freqs[None, :])
    return F


def Dinvmat(Nmat_inv, Mmat):
    '''
    The Inverse of 'D-matrix' which is like the N inverse matrix,
    but it takes into account marginalization over timing model parameters

    Author: Nima Laal
    '''

    MNM_inv = np.linalg.inv(Mmat.T @ Nmat_inv @ Mmat)
    D_inv = Nmat_inv - Nmat_inv @ Mmat @ MNM_inv @ Mmat.T @ Nmat_inv
    return D_inv

def Gmat(Mmat):
    '''
    The G-Matrix used for fitting timing residulas

    Author: Nima Laal
    '''

    U, S, V = np.linalg.svd(Mmat)
    # extract G from U
    N_par = len(V)
    G = U[:, N_par:]
    return G

def ptainvgamma(tau, low = 1e-18, high = 1e-8):
    '''
    Inverse gamma distribution which has an upper and a lower bound for its domain.
    Little bit hard to exaplain the form here!!! I will show the derivation in a seprate document.

    Author: Nima Laal
    '''
    eta = np.random.uniform(0, 1-np.exp((tau/high) - (tau/low)))
    return tau / ((tau/high) - np.log(1-eta))

def b_given_rho(res, phiinv, D_inv, Fmat):
    '''
    Calculate Fourier coefficients form rho values.

    Author: Nima Laal
    '''

    TNT = Fmat.T @ D_inv @ Fmat
    Sigma = TNT + np.diag(phiinv)
    var = np.linalg.inv(Sigma)
    mean = np.array(var @ Fmat.T @ D_inv @ res)
    b = np.random.default_rng().multivariate_normal(mean = mean, cov = var, check_valid = 'raise', method = 'svd')
    return b