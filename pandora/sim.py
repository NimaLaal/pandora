import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jar
import numpy as np
from pandora import models, utils, GWBFunctions
from pandora import LikelihoodCalculator as LC
from enterprise_extensions.model_utils import get_tspan
import pickle, json, os, corner, glob, random, copy, time, inspect
from tqdm import tqdm


class SimGWBFromPhi(object):
    """
    A class to make a simualted PTA data set by performing injections
    on the same frequency bins as thoese that will be used for detection
    runs.
    TODO: Add documentation
    """

    def __init__(self, phi, freqs, toas, white_sigma, seed=None, psrs=None):

        self.phi = phi
        self.Npulsars = phi.shape[-1]
        self.real = phi.shape[0]
        self.nf = phi.shape[1]
        self.toas = toas
        self.freqs = freqs
        self.psrs = psrs
        self.white_sigma = white_sigma
        if seed:
            random.seed(seed)
        self.rngkey = jar.split(
            jar.key(random.randint(10, 100000)), num=self.Npulsars + 1
        )

    def Fmat(self, pidx):
        """
        The 'F-matrix' used to do a discrete Fourier transform
        """
        N = len(self.toas[pidx])
        F = jnp.zeros((N, 2 * self.nf))
        F = F.at[:, 0::2].set(
            jnp.sin(2 * jnp.pi * self.toas[pidx][:, None] * self.freqs[None, :])
        )
        return F.at[:, 1::2].set(
            jnp.cos(2 * jnp.pi * self.toas[pidx][:, None] * self.freqs[None, :])
        )

    def Mmat(self, pidx):
        """
        Simplest timing design-matrix
        """
        DesignM = jnp.ones((len(self.toas[pidx]), 3))
        DesignM = DesignM.at[:, 1].set(self.toas[pidx])
        return DesignM.at[:, 2].set(DesignM[:, 1] ** 2)

    def Rmat(self, pidx):
        """
        The R-matrix used to do the quadratic fitting
        """
        I = jnp.identity(len(self.toas[pidx]))
        Mmat = self.Mmat(pidx)
        calc = jnp.linalg.inv(jnp.einsum("km,kn->nm", Mmat, Mmat))
        return I - jnp.einsum("kn,nm,pm->kp", Mmat, calc, Mmat)

    def get_coeff(self):
        L = jnp.linalg.cholesky(self.phi)
        nu = jar.normal(
            key=self.rngkey[-1], shape=(self.real, 2 * self.nf, self.Npulsars, 1)
        )
        calc = jnp.repeat(L, 2, axis=1) @ nu
        return calc[:, :, None, :, 0]

    def sim(self):
        res = []
        a = self.get_coeff()
        for pidx in tqdm(range(self.Npulsars)):
            F = self.Fmat(pidx)[None]
            R = self.Rmat(pidx)[None]
            w = jar.normal(self.rngkey[pidx], shape=(self.real, len(self.toas[pidx])))
            res.append(
                (R @ (F @ a[..., pidx]))[..., 0] + w * self.white_sigma[pidx][None]
            )
        return res

    def write_to_psrs(self, residual_list, overwrite=False):
        ans = []
        for rr in tqdm(range(self.real)):
            psrs_copy = copy.deepcopy(self.psrs)
            for pidx, psr in enumerate(psrs_copy):

                psr._toas = np.array(self.toas[pidx])
                psr._toaerrs = np.array(self.white_sigma[pidx])

                if overwrite:
                    psr._residuals = np.array(residual_list[pidx][rr])
                else:
                    psr._residuals += np.array(residual_list[pidx][rr])

                psr._designmatrix = np.array(self.Mmat(pidx))
                psr.sort_data()
            ans.append(psrs_copy)
        return ans
