import numpy as np
import scipy.linalg as sl
from tqdm import tqdm
from functools import cached_property, partial
import os, random, warnings, math
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
try:
    from enterprise_extensions import blocks
    from enterprise.signals import signal_base, gp_signals
except ImportError:
    warnings.warn("enterprise and enterprise_extensions are not found. Make sure you know your TNT and TNr tensors!!!!")

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
from jax.dlpack import to_dlpack, from_dlpack

# jax.config.update('jax_platform_name', 'cpu')
# jax.config.update("jax_enable_x64", False)

class CURN(object):
    """
    A class to calculate factorized likelihood based on given IRN + GWB models (no deterministic signal)

    :param run_type_object: a class from `run_types.py`
    :param device_to_run_likelihood_on: the device (cpu, gpu, cuda, METAL) to perform likelihood calculation on.
    :param psrs: an enterprise `psrs` object. Ignored if `TNr` and `TNT` is supplied
    :param TNr: the so-called TNr matrix. It is the product of the basis matrix `F`
    with the inverse of the timing marginalized white noise covaraince matrix D and the timing residulas `r`.
    The naming convension should read FD^-1r but TNr is a more well-known name for this quantity!
    :param TNT: the so-called TNT matrix. It is the product of the basis matrix `F`
    with the inverse of the timing marginalized white noise covaraince matrix D and the `F` transpose matrix.
    The naming convension should read FD^-1F but TNT is a more well-known name for this quantity (it sounds dynamite!)
    :param: noise_dict: the white noise noise dictionary. Ignored if `TNr` and `TNT` is supplied
    :param: backend: the telescope backend. Ignored if `TNr` and `TNT` is supplied
    :param: tnequad: do you want to use the temponest convention?. Ignored if `TNr` and `TNT` is supplied
    :param: inc_ecorr: do you want to use ecorr? Ignored if `TNr` and `TNT` is supplied
    :param: del_pta_after_init: do you want to delete the in-house-made `pta` object? Ignored if `TNr` and `TNT` is supplied
    :param: matrix_stabilization: performing some matrix stabilization on the `TNT` matrix.
    :param: the amount by which the diagonal of the correlation version of TNT is added by. This stabilizes the TNT matrix.
    if `matrix_stabilization` is set to False, this has no efect.
    Author:
    Nima Laal (02/12/2025)
    """

    def __init__(
        self,
        run_type_object,
        device_to_run_likelihood_on,
        psrs,
        TNr=jnp.array([False]),
        TNT=jnp.array([False]),
        noise_dict=None,
        backend="none",
        tnequad=False,
        inc_ecorr=False,
        del_pta_after_init=True,
        matrix_stabilization=True,
        delta = 1e-6,
    ):
        assert jnp.any(TNr.any() and TNT.any()) or any(psrs), (
            "Either supply a `psrs` object or provide `TNr` and `TNT` arrays."
        )
        self.delta = delta
        self.device = device_to_run_likelihood_on
        self.run_type_object = run_type_object
        self.Tspan = run_type_object.Tspan
        self.noise_dict = noise_dict
        self.Npulsars = run_type_object.Npulsars
        self.renorm_const = run_type_object.renorm_const
        self.crn_bins = run_type_object.crn_bins
        self.int_bins = run_type_object.int_bins
        self.diag_idx = jnp.arange(0, self.Npulsars, 1, int)
        assert self.crn_bins <= self.int_bins
        self.kmax = 2 * self.int_bins
        self.k_idx = jnp.arange(0, self.kmax)
        self.total_dim = self.Npulsars * self.kmax
        self._eye = jnp.repeat(np.eye(self.Npulsars)[None], self.int_bins, axis=0)
        self.lower_prior_lim_all = self.jax_to_numpy(
            self.run_type_object.lower_prior_lim_all
        )
        self.num_IR_params = self.run_type_object.num_IR_params

        self.upper_prior_lim_all = self.jax_to_numpy(
            self.run_type_object.upper_prior_lim_all
        )
        self.num_gwb_params = len(
            self.run_type_object.lower_prior_lim_all[self.num_IR_params :]
        )

        if not TNr.any() and not TNT.any():
            tm = gp_signals.MarginalizingTimingModel(use_svd=True)
            wn = blocks.white_noise_block(
                vary=False,
                inc_ecorr=inc_ecorr,
                gp_ecorr=False,
                select=backend,
                tnequad=tnequad,
            )
            if self.num_IR_params:
                rn = blocks.red_noise_block(
                    psd="powerlaw",
                    prior="log-uniform",
                    Tspan=self.Tspan,
                    components=self.int_bins,
                    gamma_val=None,
                )
            gwb = blocks.common_red_noise_block(
                psd="spectrum",
                prior="log-uniform",
                Tspan=self.Tspan,
                components=self.crn_bins,
                gamma_val=13 / 3,
                name="gw",
                orf="crn",
            )
            if self.num_IR_params:
                s = tm + wn + rn + gwb
            else:
                s = tm + wn + gwb

            self.pta = signal_base.PTA(
                [s(p) for p in psrs], signal_base.LogLikelihoodDenseCholesky
            )
            self.pta.set_default_params(self.noise_dict)

            self._TNr = jnp.array(self.pta.get_TNr(params={}))[..., None] / jnp.sqrt(
                self.renorm_const
            )
            self._TNT = jnp.array(
                jnp.array(self.pta.get_TNT(params={})) / self.renorm_const
            )
            if del_pta_after_init:
                del self.pta
        else:
            self._TNr = TNr / jnp.sqrt(self.renorm_const)
            self._TNT = TNT / self.renorm_const

        ##############Make TNT More Stable:
        if matrix_stabilization:
            print(f"The delta is {self.delta}")
            print(
                f"The max condition number of the TNT matrix before stabilizing is: {np.format_float_scientific(np.linalg.cond(self._TNT).max())}"
            )
            self._TNT =  self.regularize(self._TNT)
            print(
                f"The max condition number of the TNT matrix after stabilizing is: {np.format_float_scientific(np.linalg.cond(self._TNT).max())}"
            )

    ##################################################################

    """
    Some convenient functions for moving between JAX and Numpy
    depending on the device (dlpack cannot copy between CPU and GPU.)
    dlpack simply gives you a `view` of the array.
    """

    def jax_to_numpy(self, jax_array):
        if self.device == "cpu":
            return np.from_dlpack(jax_array)
        else:
            return np.array(jax_array)

    def numpy_to_jax(self, numpy_array):
        if self.device == "cpu":
            return jax.dlpack.from_dlpack(numpy_array)
        else:
            return jnp.array(numpy_array)

    def lnliklihood_wrapper_numpy(self, xs):
        xs_jax = self.numpy_to_jax(xs)
        return self.jax_to_numpy(self.get_lnliklihood(xs_jax))

    def lnliklihood_LU_wrapper_numpy(self, xs):
        xs_jax = self.numpy_to_jax(xs)
        return self.jax_to_numpy(self.get_lnliklihood_LU(xs_jax))

    ##################################################################

    def regularize(self, herm_mat):
        '''
        Regularizes a rank-deficient real symmetric matrix into a full-rank matrix
        param: `herm_mat`: the matrix to perform regularization on. The shape must be
                            (n_freq by n_pulsar by n_pulsar)
        param: `return_fac`: wheher to return the square-root factorization
        '''
        sqr_diags = np.sqrt(herm_mat.diagonal(axis1 = 1, axis2 = 2))[..., None]
        D = sqr_diags @ sqr_diags.transpose(0, 2, 1)
        corr = herm_mat/D
        return corr.at[:, self.diag_idx, self.diag_idx].add(self.delta) * D

    @partial(jax.jit, static_argnums=(0,))
    def get_lnliklihood(self, xs):
        '''
        A function to return natural log of the CURN likelihood

        param: `xs`: powerlaw model parameters
        '''
        phi_diagonal, psd_common = self.run_type_object.get_phi_mat_CURN(xs)
        logdet_phis = 2 * jnp.sum(jnp.log(phi_diagonal), axis = 0)
        phiinv = jnp.repeat(1/phi_diagonal.T, 2, axis = 1)

        Sigmas = self._TNT.at[:, self.k_idx, self.k_idx].add(phiinv)
        cfs = jsp.linalg.cho_factor(Sigmas, lower=False)
        expvals = jsp.linalg.cho_solve(cfs, self._TNr)
        logdet_sigmas = jnp.sum(2 * jnp.log(cfs[0].diagonal(axis1 = 1, axis2 = 2)), axis = 1)

        return 0.5 * jnp.sum((self._TNr.transpose((0, 2, 1)) @ expvals)[:, 0, 0] - logdet_sigmas - logdet_phis)

    @partial(jax.jit, static_argnums=(0,))
    def get_lnliklihood_from_phi_diag(self, phi_diagonal):
        '''
        A function to return natural log of the CURN likelihood

        param: `xs`: powerlaw model parameters
        '''
        logdet_phis = 2 * jnp.sum(jnp.log(phi_diagonal), axis = 0)
        phiinv = jnp.repeat(1/phi_diagonal.T, 2, axis = 1)

        Sigmas = self._TNT.at[:, self.k_idx, self.k_idx].add(phiinv)
        cfs = jsp.linalg.cho_factor(Sigmas, lower=False)
        expvals = jsp.linalg.cho_solve(cfs, self._TNr)
        logdet_sigmas = jnp.sum(2 * jnp.log(cfs[0].diagonal(axis1 = 1, axis2 = 2)), axis = 1)

        return 0.5 * jnp.sum((self._TNr.transpose((0, 2, 1)) @ expvals)[:, 0, 0] - logdet_sigmas - logdet_phis)

    def get_lnprior(self, xs):
        """
        A function to return natural log uniform-prior

        :param: xs: flattened array of model paraemters (`xs`)

        :return: either -infinity or a constant based on the predefined limits of the uniform-prior.
        """
        return self.run_type_object.get_lnprior(xs=xs)

    def get_lnprior_numpy(self, xs):
        """
        A function to return natural log prior (uniform)

        param: `xs`: powerlaw model parameters
        """
        state = np.logical_and(
            xs > self.lower_prior_lim_all, xs < self.upper_prior_lim_all
        ).all()

        if state:
            return -8.01
        else:
            return -np.inf

    def make_initial_guess_numpy(self, seed=None):
        """
        Generates an initial guess using uniform random values within
        specified limits.

        :param seed: random number generator `seed`

        :return: an array of random numbers generated using
        numpy. The shape of the array is determined by the number of elements in
        `self.upper_prior_lim_all`.
        """
        if seed:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        return rng.uniform(self.lower_prior_lim_all, self.upper_prior_lim_all)

    def sample(
        self,
        x0,
        niter,
        savedir,
        resume=True,
        seed=None,
        sample_enterprise=False,
    ):
        """
        A function to perform the sampling using PTMCMC

        :param the initial guess `x0` of all model parameters
        :param niter: the number of sampling iterations
        :param savedir: the directory to save the chains
        :param resume: do you want to resume from a saved chain?
        :param seed: rng seed!
        :param sample_enterprise: do you want to sample the internal pta object?
        :param LU_decomp: do you want to use the PLU decomposed likelihood?
        """
        if not np.any(x0):
            x0 = self.make_initial_guess_numpy(seed)
        ndim = len(x0)
        cov = np.diag(np.ones(ndim) * 0.01**2)
        groups = [list(np.arange(0, ndim))]
        nonIR_idxs = np.array(range(self.num_IR_params, x0.shape[0]))
        [groups.append(nonIR_idxs) for ii in range(2)]

        if not sample_enterprise:
            sampler = ptmcmc(
                ndim,
                self.lnliklihood_wrapper_numpy,
                self.get_lnprior_numpy,
                cov,
                groups=groups,
                outDir=savedir,
                resume=resume,
            )
        else:
            sampler = ptmcmc(
                ndim,
                self.pta.get_lnlikelihood,
                self.get_lnprior_numpy,
                cov,
                groups=groups,
                outDir=savedir,
                resume=resume,
            )

        sampler.addProposalToCycle(self.draw_from_prior, 10)
        if self.num_IR_params:
            sampler.addProposalToCycle(self.draw_from_red_prior, 10)
        sampler.addProposalToCycle(self.draw_from_nonIR_prior, 10)
        # TO DO: add proposal for orf parameters in case they exist in the model.

        sampler.sample(
            x0,
            niter,
            SCAMweight=30,
            AMweight=15,
            DEweight=50,
        )

    def draw_from_prior(self, x, iter, beta):
        """Prior draw.

        The function signature is specific to PTMCMCSampler.
        """

        q = x.copy()
        lqxy = 0

        # randomly choose parameter
        param_idx = random.randint(0, x.shape[0] - 1)
        q[param_idx] = self.make_initial_guess_numpy()[param_idx]
        return q, float(lqxy)

    def draw_from_red_prior(self, x, iter, beta):
        """Prior draw.

        The function signature is specific to PTMCMCSampler.
        """

        q = x.copy()
        lqxy = 0

        # randomly choose parameter
        param_idx = random.randint(0, self.num_IR_params - 1)
        q[param_idx] = self.make_initial_guess_numpy()[param_idx]
        return q, float(lqxy)

    def draw_from_nonIR_prior(self, x, iter, beta):
        """Prior draw.

        The function signature is specific to PTMCMCSampler.
        """

        q = x.copy()
        lqxy = 0

        # randomly choose parameter
        param_idx = random.randint(self.num_IR_params, x.shape[0] - 1)
        q[param_idx] = self.make_initial_guess_numpy()[param_idx]
        return q, float(lqxy)

class MultiPulsarModel(object):
    """
    A class to calculate full likelihood based on given IRN + GWB models (no deterministic signal)

    :param run_type_object: a class from `run_types.py`
    :param device_to_run_likelihood_on: the device (cpu, gpu, cuda, METAL) to perform likelihood calculation on.
    :param psrs: an enterprise `psrs` object. Ignored if `TNr` and `TNT` is supplied
    :param TNr: the so-called TNr matrix. It is the product of the basis matrix `F`
    with the inverse of the timing marginalized white noise covaraince matrix D and the timing residulas `r`.
    The naming convension should read FD^-1r but TNr is a more well-known name for this quantity!
    :param TNT: the so-called TNT matrix. It is the product of the basis matrix `F`
    with the inverse of the timing marginalized white noise covaraince matrix D and the `F` transpose matrix.
    The naming convension should read FD^-1F but TNT is a more well-known name for this quantity (it sounds dynamite!)
    :param: noise_dict: the white noise noise dictionary. Ignored if `TNr` and `TNT` is supplied
    :param: backend: the telescope backend. Ignored if `TNr` and `TNT` is supplied
    :param: tnequad: do you want to use the temponest convention?. Ignored if `TNr` and `TNT` is supplied
    :param: inc_ecorr: do you want to use ecorr? Ignored if `TNr` and `TNT` is supplied
    :param: del_pta_after_init: do you want to delete the in-house-made `pta` object? Ignored if `TNr` and `TNT` is supplied
    :param: matrix_stabilization: performing some matrix stabilization on the `TNT` matrix.
    :param: the amount by which the diagonal of the correlation version of TNT is added by. This stabilizes the TNT matrix.
    if `matrix_stabilization` is set to False, this has no efect.
    Author:
    Nima Laal (02/12/2025)
    """

    def __init__(
        self,
        run_type_object,
        device_to_run_likelihood_on,
        psrs,
        TNr=jnp.array([False]),
        TNT=jnp.array([False]),
        noise_dict=None,
        backend="none",
        tnequad=False,
        inc_ecorr=False,
        del_pta_after_init=True,
        matrix_stabilization=True,
        delta = 1e-6,
    ):
        assert jnp.any(TNr.any() and TNT.any()) or any(psrs), (
            "Either supply a `psrs` object or provide `TNr` and `TNT` arrays."
        )

        self.delta = delta
        self.device = device_to_run_likelihood_on
        self.run_type_object = run_type_object
        self.Tspan = run_type_object.Tspan
        self.noise_dict = noise_dict
        self.Npulsars = run_type_object.Npulsars
        self.renorm_const = run_type_object.renorm_const
        self.crn_bins = run_type_object.crn_bins
        self.int_bins = run_type_object.int_bins
        assert self.crn_bins <= self.int_bins
        self.kmax = 2 * self.int_bins
        self.k_idx = jnp.arange(0, self.kmax)
        self._eye = jnp.repeat(np.eye(self.Npulsars)[None], self.int_bins, axis=0)
        self.lower_prior_lim_all = self.jax_to_numpy(
            self.run_type_object.lower_prior_lim_all
        )
        self.upper_prior_lim_all = self.jax_to_numpy(
            self.run_type_object.upper_prior_lim_all
        )
        self.num_IR_params = run_type_object.num_IR_params

        self.num_gwb_params = len(
            self.run_type_object.lower_prior_lim_all[self.num_IR_params:]
        )

        if not TNr.any() and not TNT.any():
            tm = gp_signals.MarginalizingTimingModel(use_svd=True)
            wn = blocks.white_noise_block(
                vary=False,
                inc_ecorr=inc_ecorr,
                gp_ecorr=False,
                select=backend,
                tnequad=tnequad,
            )
            if self.num_IR_params:
                rn = blocks.red_noise_block(
                    psd="powerlaw",
                    prior="log-uniform",
                    Tspan=self.Tspan,
                    components=self.int_bins,
                    gamma_val=None,
                )
            gwb = blocks.common_red_noise_block(
                psd="spectrum",
                prior="log-uniform",
                Tspan=self.Tspan,
                components=self.crn_bins,
                gamma_val=13 / 3,
                name="gw",
                orf="hd",
            )
            if self.num_IR_params:
                s = tm + wn + rn + gwb
            else:
                s = tm + wn + gwb

            self.pta = signal_base.PTA(
                [s(p) for p in psrs], signal_base.LogLikelihoodDenseCholesky
            )
            self.pta.set_default_params(self.noise_dict)

            self._TNr = jnp.concatenate(self.pta.get_TNr(params={})) / jnp.sqrt(
                self.renorm_const
            )
            self._TNT = jnp.array(
                sl.block_diag(*self.pta.get_TNT(params={})) / self.renorm_const
            )
            if del_pta_after_init:
                del self.pta
        else:
            self._TNr = TNr / jnp.sqrt(self.renorm_const)
            self._TNT = TNT / self.renorm_const

        ##############Make TNT More Stable:
        if matrix_stabilization:
            print(f"The delta is {self.delta}")
            print(
                f"Condition number of the TNT matrix before stabilizing is: {np.format_float_scientific(np.linalg.cond(self._TNT))}"
            )
            D = jnp.outer(
                jnp.sqrt(self._TNT.diagonal()), jnp.sqrt(self._TNT.diagonal())
            )
            corr = self._TNT / D
            corr = corr + self.delta * jnp.eye(self._TNT.shape[0])
            self._TNT = D * corr / (1 + self.delta)
            # evals, evecs = jnp.linalg.eigh(corr)
            # corr = jnp.dot(evecs * jnp.maximum(evals, self.delta), evecs.T)
            # self._TNT = D * corr
            print(
                f"Condition number of the TNT matrix after stabilizing is: {np.format_float_scientific(np.linalg.cond(self._TNT))}"
            )

    ##################################################################
    """
    Some convenient functions for moving between JAX and Numpy
    depending on the device (dlpack cannot copy between CPU and GPU.)
    dlpack simply gives you a `view` of the array.
    """

    def jax_to_numpy(self, jax_array):
        if self.device == "cpu":
            return np.from_dlpack(jax_array)
        else:
            return np.array(jax_array)

    def numpy_to_jax(self, numpy_array):
        if self.device == "cpu":
            return jax.dlpack.from_dlpack(numpy_array)
        else:
            return jnp.array(numpy_array)

    def lnliklihood_wrapper_numpy(self, xs):
        xs_jax = self.numpy_to_jax(xs)
        return self.jax_to_numpy(self.get_lnliklihood(xs_jax))

    def lnliklihood_LU_wrapper_numpy(self, xs):
        xs_jax = self.numpy_to_jax(xs)
        return self.jax_to_numpy(self.get_lnliklihood_LU(xs_jax))

    ##################################################################

    @partial(jax.jit, static_argnums=(0,))
    def to_ent_phiinv(self, phiinv):
        """
        Changes the format of the phiinv matrix from (2*n_freq, n_pulsar, n_pulsar) to (2*n_freq * n_pulsar by 2*n_freq * n_pulsar)
        by adding zeros to the cross-frequency terms.

        :param: `phiinv`: the phiinv matrix with the shape (2*n_freq, n_pulsar, n_pulsar).

        :return: the phiinv matrix with the shape (2*n_freq * n_pulsar by 2*n_freq * n_pulsar).
        """
        phiinv_ent = jnp.zeros((self.Npulsars, self.kmax, self.Npulsars, self.kmax))
        phiinv_ent = phiinv_ent.at[:, self.k_idx, :, self.k_idx].add(phiinv)
        return phiinv_ent.reshape(
            (self.Npulsars * self.kmax, self.Npulsars * self.kmax)
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_mean(self, phiinv):
        """
        Estimates the mean of the Fourier coefficients as well as the log-determinant of the `Sigma` matrix.

        :param: `phiinv`: the phiinv matrix of the shape (2*n_freq, n_pulsar, n_pulsar).
        package does this automatically.

        :return: the mean of the Fourier coefficients as well as the log-determinant of the `Sigma` matrix.
        """
        cf = jsp.linalg.cho_factor(self._TNT + self.to_ent_phiinv(phiinv), lower=False)
        return jsp.linalg.cho_solve(cf, self._TNr), 2 * jnp.log(cf[0].diagonal()).sum()

    @partial(jax.jit, static_argnums=(0,))
    def get_mean_LU(self, phiinv):
        """
        *****VERY experimental*****. Estimates the mean of the Fourier coefficients as well as the log-determinant of the `Sigma` matrix.
        This function uses a `PLU` decomposition.

        :param: `phiinv`: the phiinv matrix of the shape (2*n_freq, n_pulsar, n_pulsar).
        package does this automatically.

        :return: the mean of the Fourier coefficients as well as the log-determinant of the `Sigma` matrix.
        """
        cf = jsp.linalg.lu_factor(self._TNT + self.to_ent_phiinv(phiinv))
        return jsp.linalg.lu_solve(cf, self._TNr), jnp.log(
            jnp.abs(cf[0].diagonal())
        ).sum()

    @partial(jax.jit, static_argnums=(0,))
    def get_lnliklihood(self, xs):
        """
        Calculates the log-likelihood of a multi-pulsar noise-modeling (no deterministic signal)

        :param: xs: flattened array of model paraemters (`xs`)

        :return: returns the natural-log-likelihood
        """
        phi = self.run_type_object.get_phi_mat(xs)
        cp = jsp.linalg.cho_factor(phi, lower=True)
        logdet_phi = 4 * jnp.sum(
            jnp.log(cp[0].diagonal(axis1=1, axis2=2))
        )  # Note the use of `4` instead of `2`.
        # It is needed as the phi-matrix is shaped
        # (n_freq, n_pulsar, n_pulsar) instead of
        # (2*n_freq, n_pulsar, n_pulsar).
        phiinv_dense = jnp.repeat(jsp.linalg.cho_solve(cp, self._eye), 2, axis=0)
        expval, logdet_sigma = self.get_mean(phiinv_dense)
        return 0.5 * (jnp.dot(self._TNr, expval) - logdet_sigma - logdet_phi)

    @partial(jax.jit, static_argnums=(0,))
    def get_lnliklihood_LU(self, xs):
        """
        *****VERY experimental*****. Calculates the log-likelihood of a multi-pulsar noise-modeling (no deterministic signal)
        This function uses a `PLU` decomposition to deal with the coefficients.

        :param: xs: flattened array of model paraemters (`xs`)

        :return: returns the natural-log-likelihood
        """
        phi = self.run_type_object.get_phi_mat(xs)
        cp = jsp.linalg.cho_factor(phi, lower=True)
        logdet_phi = 4 * jnp.sum(
            jnp.log(cp[0].diagonal(axis1=1, axis2=2))
        )  # Note the use of `4` instead of `2`.
        # It is needed as the phi-matrix is shaped
        # (n_freq, n_pulsar, n_pulsar) instead of
        # (2*n_freq, n_pulsar, n_pulsar).
        phiinv_dense = jnp.repeat(jsp.linalg.cho_solve(cp, self._eye), 2, axis=0)
        expval, logdet_sigma = self.get_mean_LU(phiinv_dense)
        return 0.5 * (jnp.dot(self._TNr, expval) - logdet_sigma - logdet_phi)

    @partial(jax.jit, static_argnums=(0,))
    def get_lnliklihood_using_phi(self, phi):
        """
        Calculates the log-likelihood of a multi-pulsar noise-modeling (no deterministic signal)

        :param: phi: a phi-matrix

        :return: returns the natural-log-likelihood
        """
        cp = jsp.linalg.cho_factor(phi, lower=True)
        logdet_phi = 4 * jnp.sum(
            jnp.log(cp[0].diagonal(axis1=1, axis2=2))
        )  # Note the use of `4` instead of `2`.
        # It is needed as the phi-matrix is shaped
        # (n_freq, n_pulsar, n_pulsar) instead of
        # (2*n_freq, n_pulsar, n_pulsar).
        phiinv_dense = jnp.repeat(jsp.linalg.cho_solve(cp, self._eye), 2, axis=0)
        expval, logdet_sigma = self.get_mean(phiinv_dense)
        return 0.5 * (jnp.dot(self._TNr, expval) - logdet_sigma - logdet_phi)

    def get_lnprior(self, xs):
        """
        A function to return natural log uniform-prior

        :param: xs: flattened array of model paraemters (`xs`)

        :return: either -infinity or a constant based on the predefined limits of the uniform-prior.
        """
        return self.run_type_object.get_lnprior(xs=xs)

    def get_lnprior_numpy(self, xs):
        """
        A function to return natural log prior (uniform)

        param: `xs`: powerlaw model parameters
        """
        state = np.logical_and(
            xs > self.lower_prior_lim_all, xs < self.upper_prior_lim_all
        ).all()

        if state:
            return -8.01
        else:
            return -np.inf

    def make_initial_guess_numpy(self, seed=None):
        """
        Generates an initial guess using uniform random values within
        specified limits.

        :param seed: random number generator `seed`

        :return: an array of random numbers generated using
        numpy. The shape of the array is determined by the number of elements in
        `self.upper_prior_lim_all`.
        """
        if seed:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        return rng.uniform(self.lower_prior_lim_all, self.upper_prior_lim_all)

    def sample(
        self,
        x0,
        niter,
        savedir,
        resume=True,
        seed=None,
        sample_enterprise=False,
        LU_decomp=False,
    ):
        """
        A function to perform the sampling using PTMCMC

        :param the initial guess `x0` of all model parameters
        :param niter: the number of sampling iterations
        :param savedir: the directory to save the chains
        :param resume: do you want to resume from a saved chain?
        :param seed: rng seed!
        :param sample_enterprise: do you want to sample the internal pta object?
        :param LU_decomp: do you want to use the PLU decomposed likelihood?
        """
        if not np.any(x0):
            x0 = self.make_initial_guess_numpy(seed)
        ndim = len(x0)
        cov = np.diag(np.ones(ndim) * 0.01**2)
        groups = [list(np.arange(0, ndim))]
        nonIR_idxs = np.array(range(self.num_IR_params, x0.shape[0]))
        [groups.append(nonIR_idxs) for ii in range(2)]

        if not sample_enterprise:
            if not LU_decomp:
                sampler = ptmcmc(
                    ndim,
                    self.lnliklihood_wrapper_numpy,
                    self.get_lnprior_numpy,
                    cov,
                    groups=groups,
                    outDir=savedir,
                    resume=resume,
                )
            else:
                print("***************Using LU DECOMP*****************")
                sampler = ptmcmc(
                    ndim,
                    self.lnliklihood_LU_wrapper_numpy,
                    self.get_lnprior_numpy,
                    cov,
                    groups=groups,
                    outDir=savedir,
                    resume=resume,
                )
        else:
            sampler = ptmcmc(
                ndim,
                self.pta.get_lnlikelihood,
                self.get_lnprior_numpy,
                cov,
                groups=groups,
                outDir=savedir,
                resume=resume,
            )

        sampler.addProposalToCycle(self.draw_from_prior, 10)
        if self.num_IR_params:
            sampler.addProposalToCycle(self.draw_from_red_prior, 10)
        sampler.addProposalToCycle(self.draw_from_nonIR_prior, 10)
        # TO DO: add proposal for orf parameters in case they exist in the model.

        sampler.sample(
            x0,
            niter,
            SCAMweight=30,
            AMweight=15,
            DEweight=50,
        )

    def draw_from_prior(self, x, iter, beta):
        """Prior draw.

        The function signature is specific to PTMCMCSampler.
        """

        q = x.copy()
        lqxy = 0

        # randomly choose parameter
        param_idx = random.randint(0, x.shape[0] - 1)
        q[param_idx] = self.make_initial_guess_numpy()[param_idx]
        return q, float(lqxy)

    def draw_from_red_prior(self, x, iter, beta):
        """Prior draw.

        The function signature is specific to PTMCMCSampler.
        """

        q = x.copy()
        lqxy = 0

        # randomly choose parameter
        param_idx = random.randint(0, self.num_IR_params - 1)
        q[param_idx] = self.make_initial_guess_numpy()[param_idx]
        return q, float(lqxy)

    def draw_from_nonIR_prior(self, x, iter, beta):
        """Prior draw.

        The function signature is specific to PTMCMCSampler.
        """

        q = x.copy()
        lqxy = 0

        # randomly choose parameter
        param_idx = random.randint(self.num_IR_params, x.shape[0] - 1)
        q[param_idx] = self.make_initial_guess_numpy()[param_idx]
        return q, float(lqxy)

class AstroInferenceModel(object):
    """
    A class to calculate likelihood based on a given IRN + GWB model (no deterministic signal) as well as as a normalizing flow astroemulator

    :param nf_dist: the normalizing flow object
    :param num_astro_params: the number of astro parameters
    :param astr_prior_lower_lim: the lower bound for the astro parameters
    :param astr_prior_upper_lim: the upper bound for the astro parameters
    :param astro_additional_prior_func: if you want non-uniform prior on the astro parameters, supply a numpy-compatible function to calculate
                                        the natural-log of the prior given the astro-parameters.
    :param run_type_object: a class from `run_types.py`
    :param psrs: an enterprise `psrs` object. Ignored if `TNr` and `TNT` is supplied
    :param TNr: the so-called TNr matrix. It is the product of the basis matrix `F`
    with the inverse of the timing marginalized white noise covaraince matrix D and the timing residulas `r`.
    The naming convension should read FD^-1r but TNr is a more well-known name for this quantity!
    :param TNT: the so-called TNT matrix. It is the product of the basis matrix `F`
    with the inverse of the timing marginalized white noise covaraince matrix D and the `F` transpose matrix.
    The naming convension should read FD^-1F but TNT is a more well-known name for this quantity (it sounds dynamite!)
    :param: noise_dict: the white noise noise dictionary. Ignored if `TNr` and `TNT` is supplied
    :param: backend: the telescope backend. Ignored if `TNr` and `TNT` is supplied
    :param: tnequad: do you want to use the temponest convention?. Ignored if `TNr` and `TNT` is supplied
    :param: inc_ecorr: do you want to use ecorr? Ignored if `TNr` and `TNT` is supplied
    :param: del_pta_after_init: do you want to delete the in-house-made `pta` object? Ignored if `TNr` and `TNT` is supplied
    :param: matrix_stabilization: performing some matrix stabilization on the `TNT` matrix.
    :param: the amount by which the diagonal of the correlation version of TNT is added by. This stabilizes the TNT matrix.
    if `matrix_stabilization` is set to False, this has no efect.
    Author:
    Nima Laal (02/12/2025)
    """

    def __init__(
        self,
        nf_dist,
        num_astro_params,
        astr_prior_lower_lim,
        astr_prior_upper_lim,
        astro_additional_prior_func,
        run_type_object,
        psrs,
        astro_param_fixed_values = np.array([False]), 
        astro_param_fixed_indices =np.array([False]), 
        fixed_spectrum = np.array([False]),
        TNr=jnp.array([False]),
        TNT=jnp.array([False]),
        noise_dict=None,
        backend="none",
        tnequad=False,
        inc_ecorr=False,
        del_pta_after_init=True,
        matrix_stabilization=True,
        delta = 1e-6,
    ):
        #assert jnp.any(TNr.any() and TNT.any()) or any(psrs), (
         #   "Either supply a `psrs` object or provide `TNr` and `TNT` arrays."
        #)
        self.delta = delta
        self.nf_dist = nf_dist
        self.run_type_object = run_type_object
        self.Npulsars = run_type_object.Npulsars
        self.astro_additional_prior_func = astro_additional_prior_func
        self.num_astro_params = num_astro_params
        self.num_IR_params = self.run_type_object.num_IR_params 

        self.num_gwb_params = len(
            self.run_type_object.lower_prior_lim_all[self.num_IR_params:]
        )
        self.Tspan = run_type_object.Tspan
        self.noise_dict = noise_dict
        self.renorm_const = run_type_object.renorm_const
        self.log_offset = 0.5 * np.log10(self.renorm_const)
        self.crn_bins = run_type_object.crn_bins
        self.int_bins = run_type_object.int_bins
        assert self.crn_bins <= self.int_bins
        self.kmax = 2 * self.int_bins
        self.k_idx = jnp.arange(0, self.kmax)
        self._eye = jnp.repeat(np.eye(self.Npulsars)[None], self.int_bins, axis=0)

        self.lower_prior_lim_all = np.concatenate(
            (
                self.run_type_object.lower_prior_lim_all,
                astr_prior_lower_lim,
            )
        )
        self.upper_prior_lim_all = np.concatenate(
            (
                run_type_object.upper_prior_lim_all,
                astr_prior_upper_lim,
            )
        )
        
        if fixed_spectrum.any():
            print('A fixed gwb spectrum interms of 0.5log10rho is chosen. Use `get_lnliklihood_fixed_spectrum` for sampling.')
            if fixed_spectrum.ndim == 1:
                self.half_common_log10_rho = fixed_spectrum[None]
            elif fixed_spectrum.ndim == 2:
                self.half_common_log10_rho = fixed_spectrum
            else:
                raise ValueError('The shape of the spectrum must be (1, n_gwb_freq)')

        self.astro_container = np.zeros(self.num_astro_params)
        if astro_param_fixed_values.any():
            assert np.all(astro_param_fixed_indices >= 0), 'No negative indices!'
            self.astro_container[astro_param_fixed_indices] = astro_param_fixed_values
            self.astro_param_varied_indices = np.array([_ for _ in range(self.num_astro_params) if not _ in astro_param_fixed_indices])
        else:
            self.astro_param_varied_indices = np.array([_ for _ in range(self.num_astro_params)])
        self.num_varied_astro_params = len(self.astro_param_varied_indices)
        assert len(self.make_initial_guess()) == len(run_type_object.upper_prior_lim_all) + self.num_varied_astro_params, \
        'You have chosen to fix some astro parameters. Make sure your prior also reflects this choice!'

        if not fixed_spectrum.any():
            if not TNr.any() and not TNT.any():
                tm = gp_signals.MarginalizingTimingModel(use_svd=True)
                wn = blocks.white_noise_block(
                    vary=False,
                    inc_ecorr=inc_ecorr,
                    gp_ecorr=False,
                    select=backend,
                    tnequad=tnequad,
                )
                if self.num_IR_params:
                    rn = blocks.red_noise_block(
                        psd="powerlaw",
                        prior="log-uniform",
                        Tspan=self.Tspan,
                        components=self.int_bins,
                        gamma_val=None,
                    )
                
                gwb = blocks.common_red_noise_block(
                    psd="powerlaw",
                    prior="log-uniform",
                    Tspan=self.Tspan,
                    components=self.crn_bins,
                    gamma_val=13 / 3,
                    name="gw",
                    orf="hd",
                )
                if self.num_IR_params:
                    s = tm + wn + rn + gwb
                else:
                    s = tm + wn + gwb

                self.pta = signal_base.PTA(
                    [s(p) for p in psrs], signal_base.LogLikelihoodDenseCholesky
                )
                self.pta.set_default_params(self.noise_dict)

                self._TNr = jnp.concatenate(self.pta.get_TNr(params={})) / jnp.sqrt(
                    self.renorm_const
                )
                self._TNT = jnp.array(
                    sl.block_diag(*self.pta.get_TNT(params={})) / self.renorm_const
                )
                if del_pta_after_init:
                    del self.pta
            else:
                self._TNr = TNr / jnp.sqrt(self.renorm_const)
                self._TNT = TNT / self.renorm_const

            ##############Make TNT More Stable:
            if matrix_stabilization:
                print(f"The delta is {self.delta}")
                print(
                    f"Condition number of the TNT matrix before stabilizing is: {np.format_float_scientific(np.linalg.cond(self._TNT))}"
                )
                D = jnp.outer(
                    jnp.sqrt(self._TNT.diagonal()), jnp.sqrt(self._TNT.diagonal())
                )
                corr = self._TNT / D
                corr = corr + self.delta * jnp.eye(self._TNT.shape[0])
                self._TNT = D * corr / (1 + self.delta)
                # evals, evecs = jnp.linalg.eigh(corr)
                # corr = jnp.dot(evecs * jnp.maximum(evals, self.delta), evecs.T)
                # self._TNT = D * corr
                print(
                    f"Condition number of the TNT matrix after stabilizing is: {np.format_float_scientific(np.linalg.cond(self._TNT))}"
                )

    @partial(jax.jit, static_argnums=(0,))
    def to_ent_phiinv(self, phiinv):
        """
        Changes the format of the phiinv matrix from (2*n_freq, n_pulsar, n_pulsar) to (2*n_freq * n_pulsar by 2*n_freq * n_pulsar)
        by adding zeros to the cross-frequency terms.

        :param: `phiinv`: the phiinv matrix with the shape (2*n_freq, n_pulsar, n_pulsar).

        :return: the phiinv matrix with the shape (2*n_freq * n_pulsar by 2*n_freq * n_pulsar).
        """
        phiinv_ent = jnp.zeros((self.Npulsars, self.kmax, self.Npulsars, self.kmax))
        phiinv_ent = phiinv_ent.at[:, self.k_idx, :, self.k_idx].add(phiinv)
        return phiinv_ent.reshape(
            (self.Npulsars * self.kmax, self.Npulsars * self.kmax)
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_mean(self, phiinv):
        """
        Estimates the mean of the Fourier coefficients as well as the log-determinant of the `Sigma` matrix.

        :param: `phiinv`: the phiinv matrix of the shape (2*n_freq, n_pulsar, n_pulsar).
        package does this automatically.

        :return: the mean of the Fourier coefficients as well as the log-determinant of the `Sigma` matrix.
        """
        cf = jsp.linalg.cho_factor(self._TNT + self.to_ent_phiinv(phiinv), lower=False)
        return jsp.linalg.cho_solve(cf, self._TNr), 2 * jnp.log(cf[0].diagonal()).sum()

    @partial(jax.jit, static_argnums=(0,))
    def get_lnliklihood_non_astro(self, xs):
        """
        Calculates the log-likelihood of a multi-pulsar noise-modeling (no deterministic signal)

        :param: xs: flattened array of model paraemters (`xs`)

        :return: returns the natural-log-likelihood
        """
        phi, psd_common = self.run_type_object.get_phi_mat_and_common_psd(xs)
        cp = jsp.linalg.cho_factor(phi, lower=True)
        logdet_phi = 4 * jnp.sum(
            jnp.log(cp[0].diagonal(axis1=1, axis2=2))
        )  # Note the use of `4` instead of `2`.
        # It is needed as the phi-matrix is shaped
        # (n_freq, n_pulsar, n_pulsar) instead of
        # (2*n_freq, n_pulsar, n_pulsar).
        phiinv_dense = jnp.repeat(jsp.linalg.cho_solve(cp, self._eye), 2, axis=0)
        expval, logdet_sigma = self.get_mean(phiinv_dense)
        return 0.5 * (
            jnp.dot(self._TNr, expval) - logdet_sigma - logdet_phi
        ), psd_common

    def get_lnliklihood(self, xs):
        """
        Calculates the log-likelihood of a multi-pulsar noise-modeling (no deterministic signal)
        as well as the probability coming from the NF object
        NOTE: the answer is written on CPU's memory. This is needed to use PTMCMC

        :param: xs: flattened array of model paraemters (`xs`)
        """
        astro_params = self.astro_container.copy()
        astro_params[self.astro_param_varied_indices] = xs[-self.num_varied_astro_params :]
        xs_non_astro = jnp.array(xs[: -self.num_varied_astro_params])
        lik0, psd_common = self.get_lnliklihood_non_astro(xs_non_astro)
        half_common_log10_rho = (
            np.array(0.5 * jnp.log10(psd_common.T)) - self.log_offset
        )
        lik1 = self.nf_dist.log_prob(half_common_log10_rho, astro_params)
        return np.array(lik0) + lik1 + self.astro_additional_prior_func(xs[-self.num_varied_astro_params :])

    def get_lnprior(self, xs):
        """
        A function to return natural log prior (uniform)

        param: `xs`: powerlaw model parameters
        """
        state = np.logical_and(
            xs > self.lower_prior_lim_all, xs < self.upper_prior_lim_all
        ).all()

        if state:
            return -8.01
        else:
            return -np.inf

    ########################################################################################
    '''Fixed Spectrum'''

    def get_lnliklihood_fixed_spectrum(self, xs):
        astro_params = self.astro_container.copy()
        astro_params[self.astro_param_varied_indices] = xs[-self.num_varied_astro_params :]
        lik1 = self.nf_dist.log_prob(self.half_common_log10_rho, astro_params)
        return lik1 + self.astro_additional_prior_func(xs[-self.num_varied_astro_params :])

    def get_lnprior_fixed_spectrum(self, xs):
        """
        A function to return natural log prior (uniform)

        param: `xs`: powerlaw model parameters
        """
        state = np.logical_and(
            xs > self.lower_prior_lim_all[-self.num_varied_astro_params:], xs < self.upper_prior_lim_all[-self.num_varied_astro_params:]
        ).all()

        if state:
            return -8.01
        else:
            return -np.inf
    ########################################################################################

    def make_initial_guess(self, seed=None):
        """
        Generates an initial guess using uniform random values within
        specified limits.

        :param seed: random number generator `seed`

        :return: an array of random numbers generated using
        numpy. The shape of the array is determined by the number of elements in
        `self.upper_prior_lim_all`.
        """
        if seed:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        return rng.uniform(self.lower_prior_lim_all, self.upper_prior_lim_all)

    def sample_fixed_spectrum(self, x0, niter, savedir, resume=True):
        """
        A function to perform the sampling using PTMCMC

        :param the initial guess `x0` of all model parameters
        :param niter: the number of sampling iterations
        :param savedir: the directory to save the chains
        :param resume: do you want to resume from a saved chain?
        """
        if not x0.any():
            x0 = self.make_initial_guess()[-self.num_varied_astro_params:]
            print(f'First lnlikelihood = {self.get_lnliklihood_fixed_spectrum(x0)}')
            print(f'First lnlprior = {self.get_lnprior_fixed_spectrum(x0)}')
        ndim = len(x0)
        print(f'The dimensionality of the parameter space is {ndim}')
        cov = np.diag(np.ones(ndim) * 0.01**2)
        groups = [list(np.arange(0, ndim))]

        sampler = ptmcmc(
            ndim,
            self.get_lnliklihood_fixed_spectrum,
            self.get_lnprior_fixed_spectrum,
            cov,
            groups=groups,
            outDir=savedir,
            resume=resume,
        )

        sampler.addProposalToCycle(self.draw_from_astro_prior, 10)

        sampler.sample(
            x0,
            niter,
            SCAMweight=30,
            AMweight=15,
            DEweight=50,
        )

    def sample(self, x0, niter, savedir, resume=True):
        """
        A function to perform the sampling using PTMCMC

        :param the initial guess `x0` of all model parameters
        :param niter: the number of sampling iterations
        :param savedir: the directory to save the chains
        :param resume: do you want to resume from a saved chain?
        """
        if not x0.any():
            x0 = self.make_initial_guess()
        ndim = len(x0)
        cov = np.diag(np.ones(ndim) * 0.01**2)
        groups = [list(np.arange(0, ndim))]
        nonIR_idxs = np.array(range(self.num_IR_params, x0.shape[0]))
        [groups.append(nonIR_idxs) for ii in range(2)]

        sampler = ptmcmc(
            ndim,
            self.get_lnliklihood,
            self.get_lnprior,
            cov,
            groups=groups,
            outDir=savedir,
            resume=resume,
        )

        sampler.addProposalToCycle(self.draw_from_prior, 10)
        if self.num_IR_params:
            sampler.addProposalToCycle(self.draw_from_red_prior, 10)
        sampler.addProposalToCycle(self.draw_from_nonIR_prior, 10)
        sampler.addProposalToCycle(self.draw_from_astro_prior, 10)

        sampler.sample(
            x0,
            niter,
            SCAMweight=30,
            AMweight=15,
            DEweight=50,
        )

    def draw_from_prior(self, x, iter, beta):
        """Prior draw.

        The function signature is specific to PTMCMCSampler.
        """

        q = x.copy()
        lqxy = 0

        # randomly choose parameter
        param_idx = random.randint(0, x.shape[0] - 1)
        q[param_idx] = self.make_initial_guess()[param_idx]
        return q, float(lqxy)

    def draw_from_red_prior(self, x, iter, beta):
        """Prior draw.

        The function signature is specific to PTMCMCSampler.
        """

        q = x.copy()
        lqxy = 0

        # randomly choose parameter
        param_idx = random.randint(0, self.num_IR_params - 1)
        q[param_idx] = self.make_initial_guess()[param_idx]
        return q, float(lqxy)

    def draw_from_nonIR_prior(self, x, iter, beta):
        """Prior draw.

        The function signature is specific to PTMCMCSampler.
        """

        q = x.copy()
        lqxy = 0

        # randomly choose parameter
        param_idx = random.randint(self.num_IR_params, x.shape[0] - 1)
        q[param_idx] = self.make_initial_guess()[param_idx]
        return q, float(lqxy)

    def draw_from_astro_prior(self, x, iter, beta):
        """Prior draw.

        The function signature is specific to PTMCMCSampler.
        """

        q = x.copy()
        lqxy = 0

        # randomly choose parameter
        param_idx = random.randint(
            
            -self.num_varied_astro_params, -1
        )
        q[param_idx] = self.make_initial_guess()[param_idx]
        return q, float(lqxy)

class TwoModelHyperModel(object):
    """
    A class to perform a product-sapce sampling for two competing models.
    Both models MUST have the same number of pulsars.

    param model1: the first model
    param model2: the second model

    Author:
    Nima Laal (02/12/2025)
    """

    def __init__(self, model1, model2, log_weights = [0., 0.], device="cuda"):
        self.model1 = model1
        self.model2 = model2
        self.device = device
        self.log_weights = log_weights

        assert self.model1.Npulsars == self.model2.Npulsars
        self.Npulsars = self.model1.Npulsars
        self.num_IR_params = self.model1.num_IR_params

        self.combine_idxs
        self.lower_prior_lim_all = jnp.concatenate(
            (
                model1.run_type_object.lower_prior_lim_all,
                model2.run_type_object.lower_prior_lim_all[self.model2.num_IR_params :],
                jnp.array([0.0]),
            )
        )
        self.upper_prior_lim_all = jnp.concatenate(
            (
                model1.run_type_object.upper_prior_lim_all,
                model2.run_type_object.upper_prior_lim_all[self.model2.num_IR_params :],
                jnp.array([1.0]),
            )
        )

    ##################################################################
    """
    Some convenient functions for moving between JAX and Numpy
    depending on the device (dlpack cannot copy between CPU and GPU.)
    dlpack simply gives you a `view` of the array.
    """

    def jax_to_numpy(self, jax_array):
        if self.device == "cpu":
            return np.from_dlpack(jax_array)
        else:
            return np.array(jax_array)

    def numpy_to_jax(self, numpy_array):
        if self.device == "cpu":
            return jax.dlpack.from_dlpack(numpy_array)
        else:
            return jnp.array(numpy_array)

    def lnliklihood_wrapper_numpy(self, xs):
        xs_jax = self.numpy_to_jax(xs)
        return self.jax_to_numpy(self.get_lnliklihood(xs_jax))

    def lnprior_wrapper_numpy(self, xs):
        xs_jax = self.numpy_to_jax(xs)
        return self.jax_to_numpy(self.get_lnprior(xs_jax))

    ##################################################################

    def x_to_x1_indices(self):
        """
        Returns an array of indices for different parameters in model 1.
        The indicies are later used to parse a flat array of all model paraemters
        and to find only the ones that belong to model 1.
        The order is `IR + GWB params (all) of model 1`
        """
        IRN_idxs = jnp.array(range(0, self.num_IR_params), dtype=int)
        gwb_idxs = jnp.array(
            range(
                self.model1.num_IR_params,
                self.model1.num_IR_params + self.model1.num_gwb_params,
            ),
            dtype=int,
        )
        hm_idxs = jnp.concatenate((IRN_idxs, gwb_idxs))
        return hm_idxs

    def x_to_x2_indices(self):
        """
        Returns an array of indices for different parameters in model 2.
        The indicies are later used to parse a flat array of all model paraemters
        and to find only the ones that belong to model 2.
        The order is `IR + skip over GWB params of model 1 + GWB params of model 2`
        """
        IRN_idxs = jnp.array(range(0, self.model1.num_IR_params), dtype=int)
        gwb_idxs = jnp.array(
            range(
                self.model1.num_IR_params + self.model1.num_gwb_params,
                self.model1.num_IR_params
                + self.model1.num_gwb_params
                + self.model2.num_gwb_params,
            ),
            dtype=int,
        )
        hm_idxs = jnp.concatenate((IRN_idxs, gwb_idxs))
        return hm_idxs

    @cached_property
    def combine_idxs(self):
        """
        Returns a tuple containing the indices obtained from calling
        `x_to_x1_indices` and `x_to_x2_indices` methods. The output is
        cached.
        """
        return (self.x_to_x1_indices(), self.x_to_x2_indices())

    def make_initial_guess_jax(self, key):
        """
        Generates an initial guess using uniform random values within
        specified limits.

        :param key: RNG key used for generating random numbers in a.
        It is commonly used in libraries like JAX for generating random numbers

        :return: an array of random numbers generated using
        JAX's `random.uniform` function. The shape of the array is determined by the number of elements in
        `self.upper_prior_lim_all`. The random numbers are generated within the range specified by `minval =
        self.lower_prior_lim_all` and `maxval = self.upper_prior_lim_all`.
        """
        return jr.uniform(key, minval = self.lower_prior_lim_all, 
                               maxval = self.upper_prior_lim_all,
                               shape = self.upper_prior_lim_all.shape)


    def get_lnliklihood_pure_jax(self, xs):
        """
        Calculates the log-likelihood of a two-model HM run

        :param: xs: flattened array of model paraemters (`xs`)
        """
        nmodel = jnp.rint(xs[-1]).astype(bool)
        idxs = self.combine_idxs[nmodel.astype(int)]
        return jax.lax.cond(
            nmodel,
            self.model2.get_lnliklihood,
            self.model1.get_lnliklihood,
            xs[idxs],
        )

    def get_lnliklihood(self, xs):
        """
        Calculates the log-likelihood of a two-model HM run

        :param: xs: flattened array of model paraemters (`xs`)
        """
        nmodel = round(xs[-1])
        idxs = self.combine_idxs[nmodel]
        if nmodel:
            return self.model2.get_lnliklihood(xs[idxs]) + self.log_weights[1]
        else:
            return self.model1.get_lnliklihood(xs[idxs]) + self.log_weights[0]

    def make_initial_guess(self, seed=None):
        """
        Generates an initial guess using uniform random values within
        specified limits.

        :param seed: random number generator `seed`

        :return: an array of random numbers generated using
        numpy. The shape of the array is determined by the number of elements in
        `self.upper_prior_lim_all`.
        """
        if seed:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        return rng.uniform(self.lower_prior_lim_all, self.upper_prior_lim_all)

    @partial(jax.jit, static_argnums=(0,))
    def get_lnprior(self, xs):
        """
        A function to return natural log prior (uniform)

        param: `xs`: powerlaw model parameters
        """
        state = jnp.logical_and(
            xs > self.lower_prior_lim_all, xs < self.upper_prior_lim_all
        ).all()
        return jax.lax.cond(state, self.spit_neg_number, self.spit_neg_infinity)


    def spit_neg_infinity(self):
        return -jnp.inf

    def spit_neg_number(self):
        return -8.01

    def sample(self, x0, niter, savedir, resume=True):
        """
        A function to perform the sampling using PTMCMC

        :param x0: the initial guess of all model parameters (flat numpy array)
        :param niter: the number of sampling iterations
        :param savedir: the directory to save the chains
        :param resume: do you want to resume from a saved chain?
        """
        if not x0.any():
            x0 = self.make_initial_guess()
        ndim = len(x0)
        cov = np.diag(np.ones(ndim) * 0.01**2)
        groups = [list(np.arange(0, ndim))]
        nonIR_idxs = np.array(range(self.model1.num_IR_params, x0.shape[0]))
        [groups.append(nonIR_idxs) for ii in range(2)]

        sampler = ptmcmc(
            ndim,
            self.lnliklihood_wrapper_numpy,
            self.lnprior_wrapper_numpy,
            cov,
            groups=groups,
            outDir=savedir,
            resume=resume,
        )

        sampler.addProposalToCycle(self.draw_from_prior, 10)
        sampler.addProposalToCycle(self.draw_from_red_prior, 10)
        sampler.addProposalToCycle(self.draw_from_gwb1_prior, 10)
        sampler.addProposalToCycle(self.draw_from_gwb2_prior, 10)
        sampler.addProposalToCycle(self.draw_from_nmodel, 10)

        sampler.sample(
            x0,
            niter,
            SCAMweight=30,
            AMweight=15,
            DEweight=50,
        )

    def draw_from_prior(self, x, iter, beta):
        """Prior draw.

        The function signature is specific to PTMCMCSampler.
        """

        q = x.copy()
        lqxy = 0

        # randomly choose parameter
        param_idx = random.randint(0, x.shape[0] - 1)
        q[param_idx] = self.make_initial_guess()[param_idx]
        return q, float(lqxy)

    def draw_from_red_prior(self, x, iter, beta):
        """Prior draw.

        The function signature is specific to PTMCMCSampler.
        """

        q = x.copy()
        lqxy = 0

        # randomly choose parameter
        param_idx = random.randint(0, self.model1.num_IR_params)
        q[param_idx] = self.make_initial_guess()[param_idx]
        return q, float(lqxy)

    def draw_from_gwb1_prior(self, x, iter, beta):
        """Prior draw.

        The function signature is specific to PTMCMCSampler.
        """

        q = x.copy()
        lqxy = 0

        # randomly choose parameter
        st = self.model1.num_IR_params
        param_idx = random.randint(st, st + self.model1.num_gwb_params - 1)
        q[param_idx] = self.make_initial_guess()[param_idx]
        return q, float(lqxy)

    def draw_from_gwb2_prior(self, x, iter, beta):
        """Prior draw.

        The function signature is specific to PTMCMCSampler.
        """

        q = x.copy()
        lqxy = 0

        # randomly choose parameter
        st = self.model1.num_IR_params + self.model1.num_gwb_params
        param_idx = random.randint(st, st + self.model2.num_gwb_params - 1)
        q[param_idx] = self.make_initial_guess()[param_idx]
        return q, float(lqxy)

    def draw_from_nmodel(self, x, iter, beta):
        """Prior draw.

        The function signature is specific to PTMCMCSampler.
        """

        q = x.copy()
        lqxy = 0
        q[-1] = random.uniform(a=0, b=1)
        return q, float(lqxy)

class TwoAstroModelHyperModel(object):
    """
    A class to perform a product-sapce sampling for two competing models.

    param model1: the first model
    param model2: the second model

    Author:
    Nima Laal (02/12/2025)
    """

    def __init__(self, model1, model2, device="cuda"):
        self.model1 = model1
        self.model2 = model2
        self.device = device

        assert self.model1.Npulsars == self.model2.Npulsars
        self.Npulsars = self.model1.Npulsars

        self.combine_idxs
        self.lower_prior_lim_all = np.concatenate(
            (
                model1.lower_prior_lim_all,
                model2.lower_prior_lim_all[self.model2.num_IR_params :],
                np.array([0.0]),
            )
        )
        self.upper_prior_lim_all = np.concatenate(
            (
                model1.upper_prior_lim_all,
                model2.upper_prior_lim_all[self.model2.num_IR_params :],
                np.array([1.0]),
            )
        )

    def x_to_x1_indices(self):
        """
        Returns an array of indices for different parameters in model 1.
        The indicies are later used to parse a flat array of all model paraemters
        and to find only the ones that belong to model 1.
        The order is `IR + GWB params (all) of model 1 + astro params of model`
        """
        IRN_idxs = np.array(range(0, self.model1.num_IR_params), dtype=int)
        nonIR_idxs = np.array(
            range(
                self.model1.num_IR_params,
                self.model1.num_IR_params
                + self.model1.num_gwb_params
                + self.model1.num_astro_params,
            ),
            dtype=int,
        )
        hm_idxs = np.concatenate((IRN_idxs, nonIR_idxs))
        return hm_idxs

    def x_to_x2_indices(self):
        """
        Returns an array of indices for different parameters in model 2.
        The indicies are later used to parse a flat array of all model paraemters
        and to find only the ones that belong to model 2.
        The order is `IR + skip over GWB params of model 1 + skip over astro params of model 1
        + GWB params of model 2` + astro params of model 2
        """
        IRN_idxs = np.array(range(0, self.model1.num_IR_params), dtype=int)
        nonIR_idxs = np.array(
            range(
                self.model1.num_IR_params
                + self.model1.num_gwb_params
                + self.model1.num_astro_params,
                self.model1.num_IR_params
                + self.model1.num_gwb_params
                + self.model1.num_astro_params
                + self.model2.num_gwb_params
                + self.model2.num_astro_params,
            ),
            dtype=int,
        )
        hm_idxs = np.concatenate((IRN_idxs, nonIR_idxs))
        return hm_idxs

    @cached_property
    def combine_idxs(self):
        """
        Returns a tuple containing the indices obtained from calling
        `x_to_x1_indices` and `x_to_x2_indices` methods. The output is
        cached.
        """
        return (self.x_to_x1_indices(), self.x_to_x2_indices())

    def make_initial_guess(self, seed=None):
        """
        Generates an initial guess using uniform random values within
        specified limits.

        :param seed: random number generator `seed`

        :return: an array of random numbers generated using
        numpy. The shape of the array is determined by the number of elements in
        `self.upper_prior_lim_all`.
        """
        if seed:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        return rng.uniform(self.lower_prior_lim_all, self.upper_prior_lim_all)

    def get_lnliklihood(self, xs):
        """
        Calculates the log-likelihood of a two-model HM (for astro inference!).
        NOTE: the answer is written on CPU's memory. This is needed to use PTMCMC

        :param: xs: flattened array of model paraemters (`xs`)
        """
        nmodel = np.rint(xs[-1]).astype(bool)
        if nmodel:
            return self.model2.get_lnliklihood(xs[self.combine_idxs[1]])
        else:
            return self.model1.get_lnliklihood(xs[self.combine_idxs[0]])

    def get_lnprior(self, xs):
        """
        A function to return natural log prior (uniform)

        param: `xs`: powerlaw model parameters
        """
        state = np.logical_and(
            xs > self.lower_prior_lim_all, xs < self.upper_prior_lim_all
        ).all()
        if state:
            return 8.01
        else:
            return -np.inf

    def sample(self, x0, niter, savedir, resume=True):
        """
        A function to perform the sampling using PTMCMC

        :param x0: the inital guess of all model parameters
        :param niter: the number of sampling iterations
        :param savedir: the directory to save the chains
        :param resume: do you want to resume from a saved chain?
        :param seed: rng seed!
        """
        ndim = len(x0)
        cov = np.diag(np.ones(ndim) * 0.01**2)
        groups = [list(np.arange(0, ndim))]
        nonIR_idxs = np.array(range(self.model1.num_IR_params, x0.shape[0]))
        [groups.append(nonIR_idxs) for ii in range(2)]

        sampler = ptmcmc(
            ndim,
            self.get_lnliklihood,
            self.get_lnprior,
            cov,
            groups=groups,
            outDir=savedir,
            resume=resume,
        )

        sampler.addProposalToCycle(self.draw_from_prior, 10)
        sampler.addProposalToCycle(self.draw_from_red_prior, 10)
        sampler.addProposalToCycle(self.draw_from_gwb1_prior, 10)
        sampler.addProposalToCycle(self.draw_from_gwb2_prior, 10)
        sampler.addProposalToCycle(self.draw_from_nmodel, 10)
        sampler.addProposalToCycle(self.draw_from_astro1_prior, 10)
        sampler.addProposalToCycle(self.draw_from_astro2_prior, 10)

        sampler.sample(
            x0,
            niter,
            SCAMweight=30,
            AMweight=15,
            DEweight=50,
        )

    def draw_from_prior(self, x, iter, beta):
        """Prior draw.

        The function signature is specific to PTMCMCSampler.
        """

        q = x.copy()
        lqxy = 0

        # randomly choose parameter
        param_idx = random.randint(0, x.shape[0] - 1)
        q[param_idx] = self.make_initial_guess()[param_idx]
        return q, float(lqxy)

    def draw_from_red_prior(self, x, iter, beta):
        """Prior draw.

        The function signature is specific to PTMCMCSampler.
        """

        q = x.copy()
        lqxy = 0

        # randomly choose parameter
        param_idx = random.randint(0, self.model1.num_IR_params)
        q[param_idx] = self.make_initial_guess()[param_idx]
        return q, float(lqxy)

    def draw_from_gwb1_prior(self, x, iter, beta):
        """Prior draw.

        The function signature is specific to PTMCMCSampler.
        """

        q = x.copy()
        lqxy = 0

        # randomly choose parameter
        st = self.model1.num_IR_params
        param_idx = random.randint(st, st + self.model1.num_gwb_params - 1)
        q[param_idx] = self.make_initial_guess()[param_idx]
        return q, float(lqxy)

    def draw_from_gwb2_prior(self, x, iter, beta):
        """Prior draw.

        The function signature is specific to PTMCMCSampler.
        """

        q = x.copy()
        lqxy = 0

        # randomly choose parameter
        st = (
            self.model1.num_IR_params
            + self.model1.num_gwb_params
            + self.model1.num_astro_params
        )
        param_idx = random.randint(st, st + self.model2.num_gwb_params - 1)
        q[param_idx] = self.make_initial_guess()[param_idx]
        return q, float(lqxy)

    def draw_from_astro1_prior(self, x, iter, beta):
        """Prior draw.

        The function signature is specific to PTMCMCSampler.
        """

        q = x.copy()
        lqxy = 0

        self.model1.num_astro_params
        # randomly choose parameter
        st = self.model1.num_IR_params + self.model1.num_gwb_params
        param_idx = random.randint(st, st + self.model1.num_astro_params - 1)
        q[param_idx] = self.make_initial_guess()[param_idx]
        return q, float(lqxy)

    def draw_from_astro2_prior(self, x, iter, beta):
        """Prior draw.

        The function signature is specific to PTMCMCSampler.
        """

        q = x.copy()
        lqxy = 0

        self.model1.num_astro_params
        # randomly choose parameter
        st = (
            self.model1.num_IR_params
            + self.model1.num_gwb_params
            + self.model1.num_astro_params
            + self.model2.num_gwb_params
        )
        param_idx = random.randint(st, st + self.model2.num_astro_params - 1)
        q[param_idx] = self.make_initial_guess()[param_idx]
        return q, float(lqxy)

    def draw_from_nmodel(self, x, iter, beta):
        """Prior draw.

        The function signature is specific to PTMCMCSampler.
        """

        q = x.copy()
        lqxy = 0
        q[-1] = random.uniform(a=0, b=1)
        return q, float(lqxy)

class KDE(object):
    """
    A class to calculate likelihood based on given kernel density estimates. EXPERIMENTAL

    :param run_type_object: a class from `run_types.py`
    :param device_to_run_likelihood_on: the device (cpu, gpu, cuda, METAL) to perform likelihood calculation on.
    :param psrs: an enterprise `psrs` object. Ignored if `TNr` and `TNT` is supplied
    :param TNr: the so-called TNr matrix. It is the product of the basis matrix `F`
    with the inverse of the timing marginalized white noise covaraince matrix D and the timing residulas `r`.
    The naming convension should read FD^-1r but TNr is a more well-known name for this quantity!
    :param TNT: the so-called TNT matrix. It is the product of the basis matrix `F`
    with the inverse of the timing marginalized white noise covaraince matrix D and the `F` transpose matrix.
    The naming convension should read FD^-1F but TNT is a more well-known name for this quantity (it sounds dynamite!)
    :param: noise_dict: the white noise noise dictionary. Ignored if `TNr` and `TNT` is supplied
    :param: backend: the telescope backend. Ignored if `TNr` and `TNT` is supplied
    :param: tnequad: do you want to use the temponest convention?. Ignored if `TNr` and `TNT` is supplied
    :param: inc_ecorr: do you want to use ecorr? Ignored if `TNr` and `TNT` is supplied
    :param: del_pta_after_init: do you want to delete the in-house-made `pta` object? Ignored if `TNr` and `TNT` is supplied
    :param: matrix_stabilization: performing some matrix stabilization on the `TNT` matrix.
    :param: the amount by which the diagonal of the correlation version of TNT is added by. This stabilizes the TNT matrix.
    if `matrix_stabilization` is set to False, this has no efect.
    Author:
    Nima Laal (02/12/2025)
    """

    def __init__(
        self,
        grid,
        den,
        run_type_object,
        device_to_run_likelihood_on,
    ):
        self.device = device_to_run_likelihood_on
        if self.device == 'cpu':
            self.search_method = 'scan'
        else:    
            self.search_method = 'scan_unrolled'

        self.grid = jnp.array(grid)
        self.den = jnp.array(den)
        self.run_type_object = run_type_object
        self.Tspan = run_type_object.Tspan
        self.Npulsars = run_type_object.Npulsars
        self.renorm_const = run_type_object.renorm_const
        self.crn_bins = run_type_object.crn_bins
        self.int_bins = run_type_object.int_bins
        self.diag_idx = jnp.arange(0, self.Npulsars, 1, int)
        assert self.crn_bins <= self.int_bins
        self.kmax = 2 * self.int_bins
        self.k_idx = jnp.arange(0, self.kmax)
        self.total_dim = self.Npulsars * self.kmax
        self._eye = jnp.repeat(np.eye(self.Npulsars)[None], self.int_bins, axis=0)
        self.lower_prior_lim_all = self.jax_to_numpy(
            self.run_type_object.lower_prior_lim_all
        )
        self.num_IR_params = self.run_type_object.num_IR_params

        self.upper_prior_lim_all = self.jax_to_numpy(
            self.run_type_object.upper_prior_lim_all
        )
        self.num_gwb_params = len(
            self.run_type_object.lower_prior_lim_all[self.num_IR_params :]
        )
    ##################################################################
    #indices needed to search inside `den` in a vectorized way 
        self.u = np.ones((self.crn_bins, self.Npulsars), dtype = int)
        self.s = np.ones((self.crn_bins, self.Npulsars), dtype = int)
        for pidx in range(self.Npulsars):
            self.u[:, pidx] = range(self.crn_bins)
        for fidx in range(self.crn_bins):
            self.s[fidx, :] = range(self.Npulsars)
        self.u = jnp.array(self.u)
        self.s = jnp.array(self.s)
    ##################################################################

    """
    Some convenient functions for moving between JAX and Numpy
    depending on the device (dlpack cannot copy between CPU and GPU.)
    dlpack simply gives you a `view` of the array.
    """

    def jax_to_numpy(self, jax_array):
        if self.device == "cpu":
            return np.from_dlpack(jax_array)
        else:
            return np.array(jax_array)

    def numpy_to_jax(self, numpy_array):
        if self.device == "cpu":
            return jax.dlpack.from_dlpack(numpy_array)
        else:
            return jnp.array(numpy_array)

    def lnliklihood_wrapper_numpy(self, xs):
        xs_jax = self.numpy_to_jax(xs)
        return self.jax_to_numpy(self.get_lnliklihood(xs_jax))

    def lnliklihood_LU_wrapper_numpy(self, xs):
        xs_jax = self.numpy_to_jax(xs)
        return self.jax_to_numpy(self.get_lnliklihood_LU(xs_jax))

    ##################################################################

    @partial(jax.jit, static_argnums=(0,))
    def get_lnliklihood(self, xs):
        '''
        A function to return natural log of the CURN likelihood

        param: `xs`: powerlaw model parameters
        '''
        phi_diagonal, psd_common = self.run_type_object.get_phi_mat_CURN(xs)
        idxs = jnp.searchsorted(self.grid, 0.5 * jnp.log10(phi_diagonal), method = self.search_method) - 1
        return self.den[self.s, self.u, idxs].sum()

    def get_lnprior(self, xs):
        """
        A function to return natural log uniform-prior

        :param: xs: flattened array of model paraemters (`xs`)

        :return: either -infinity or a constant based on the predefined limits of the uniform-prior.
        """
        return self.run_type_object.get_lnprior(xs=xs)

    def get_lnprior_numpy(self, xs):
        """
        A function to return natural log prior (uniform)

        param: `xs`: powerlaw model parameters
        """
        state = np.logical_and(
            xs > self.lower_prior_lim_all, xs < self.upper_prior_lim_all
        ).all()

        if state:
            return -8.01
        else:
            return -np.inf

    def make_initial_guess_numpy(self, seed=None):
        """
        Generates an initial guess using uniform random values within
        specified limits.

        :param seed: random number generator `seed`

        :return: an array of random numbers generated using
        numpy. The shape of the array is determined by the number of elements in
        `self.upper_prior_lim_all`.
        """
        if seed:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        return rng.uniform(self.lower_prior_lim_all, self.upper_prior_lim_all)

    def sample(
        self,
        x0,
        niter,
        savedir,
        resume=True,
        seed=None,
        sample_enterprise=False,
    ):
        """
        A function to perform the sampling using PTMCMC

        :param the initial guess `x0` of all model parameters
        :param niter: the number of sampling iterations
        :param savedir: the directory to save the chains
        :param resume: do you want to resume from a saved chain?
        :param seed: rng seed!
        :param sample_enterprise: do you want to sample the internal pta object?
        :param LU_decomp: do you want to use the PLU decomposed likelihood?
        """
        if not np.any(x0):
            x0 = self.make_initial_guess_numpy(seed)
        ndim = len(x0)
        cov = np.diag(np.ones(ndim) * 0.01**2)
        groups = [list(np.arange(0, ndim))]
        nonIR_idxs = np.array(range(self.num_IR_params, x0.shape[0]))
        [groups.append(nonIR_idxs) for ii in range(2)]

        if not sample_enterprise:
            sampler = ptmcmc(
                ndim,
                self.lnliklihood_wrapper_numpy,
                self.get_lnprior_numpy,
                cov,
                groups=groups,
                outDir=savedir,
                resume=resume,
            )
        else:
            sampler = ptmcmc(
                ndim,
                self.pta.get_lnlikelihood,
                self.get_lnprior_numpy,
                cov,
                groups=groups,
                outDir=savedir,
                resume=resume,
            )

        sampler.addProposalToCycle(self.draw_from_prior, 10)
        if self.num_IR_params:
            sampler.addProposalToCycle(self.draw_from_red_prior, 10)
        sampler.addProposalToCycle(self.draw_from_nonIR_prior, 10)
        # TO DO: add proposal for orf parameters in case they exist in the model.

        sampler.sample(
            x0,
            niter,
            SCAMweight=30,
            AMweight=15,
            DEweight=50,
        )

    def draw_from_prior(self, x, iter, beta):
        """Prior draw.

        The function signature is specific to PTMCMCSampler.
        """

        q = x.copy()
        lqxy = 0

        # randomly choose parameter
        param_idx = random.randint(0, x.shape[0] - 1)
        q[param_idx] = self.make_initial_guess_numpy()[param_idx]
        return q, float(lqxy)

    def draw_from_red_prior(self, x, iter, beta):
        """Prior draw.

        The function signature is specific to PTMCMCSampler.
        """

        q = x.copy()
        lqxy = 0

        # randomly choose parameter
        param_idx = random.randint(0, self.num_IR_params - 1)
        q[param_idx] = self.make_initial_guess_numpy()[param_idx]
        return q, float(lqxy)

    def draw_from_nonIR_prior(self, x, iter, beta):
        """Prior draw.

        The function signature is specific to PTMCMCSampler.
        """

        q = x.copy()
        lqxy = 0

        # randomly choose parameter
        param_idx = random.randint(self.num_IR_params, x.shape[0] - 1)
        q[param_idx] = self.make_initial_guess_numpy()[param_idx]
        return q, float(lqxy)