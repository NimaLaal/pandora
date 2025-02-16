import numpy as np
import scipy.linalg as sl
from tqdm import tqdm
from functools import cached_property, partial
import os, random
from enterprise_extensions import blocks
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
from enterprise.signals import signal_base, gp_signals

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
from jax.dlpack import to_dlpack, from_dlpack

# jax.config.update('jax_platform_name', 'cpu')
# jax.config.update("jax_enable_x64", False)


class MultiPulsarModel(object):
    """
    A class to calculate likelihood based on given IRN + GWB models (no deterministic signal)

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
    :param: matrix_stabilization: performing some matrix stabilization on the `TNT` matrix. It is highly recommended!

    Author:
    Nima Laal (02/12/2025)
    """

    def __init__(
        self,
        run_type_object,
        psrs,
        TNr=jnp.array([False]),
        TNT=jnp.array([False]),
        noise_dict=None,
        backend="none",
        tnequad=False,
        inc_ecorr=False,
        del_pta_after_init=True,
        matrix_stabilization=True,
    ):
        assert jnp.any(TNr.any() and TNT.any()) or any(psrs), (
            "Either supply a `psrs` object or provide `TNr` and `TNT` arrays."
        )

        self.run_type_object = run_type_object
        self.Tspan = run_type_object.Tspan
        self.noise_dict = noise_dict
        self.Npulsars = run_type_object.Npulsars
        self.renorm_const = run_type_object.renorm_const
        self.crn_bins = run_type_object.crn_bins
        self.int_bins = run_type_object.int_bins
        assert self.crn_bins <= self.int_bins
        self.kmax = 2 * self.int_bins
        self.diag_idx = jnp.arange(0, self.Npulsars)
        self.diag_idx_large = np.arange(0, self.Npulsars * self.kmax)
        self.k_idx = jnp.arange(0, self.kmax)
        self.c_idx = jnp.arange(0, self.crn_bins)
        self._eye = jnp.repeat(np.eye(self.Npulsars)[None], self.int_bins, axis=0)

        if not TNr and not TNT:
            tm = gp_signals.MarginalizingTimingModel(use_svd=True)
            wn = blocks.white_noise_block(
                vary=False,
                inc_ecorr=inc_ecorr,
                gp_ecorr=False,
                select=backend,
                tnequad=tnequad,
            )
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
            s = tm + wn + rn + gwb

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
            print(
                f"Condition number of the TNT matrix before stabilizing is: {np.format_float_scientific(np.linalg.cond(self._TNT))}"
            )
            D = jnp.outer(
                jnp.sqrt(self._TNT.diagonal()), jnp.sqrt(self._TNT.diagonal())
            )
            corr = self._TNT / D
            corr = corr + 1e-3 * jnp.eye(self._TNT.shape[0])
            self._TNT = D * corr / (1 + 1e-3)
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

    def sample(self, niter, savedir, resume=True, rngkey=jr.key(100)):
        """
        A function to perform the sampling using PTMCMC

        :param niter: the number of sampling iterations
        :param savedir: the directory to save the chains
        :param resume: do you want to resume from a saved chain?
        :param resume: annoying JAX rng key!
        """

        x0 = self.make_initial_guess(rngkey)
        ndim = len(x0)
        cov = np.diag(np.ones(ndim) * 0.01**2)
        groups = [list(np.arange(0, ndim))]
        important_idxs = np.array(
            range(
                2 * self.Npulsars, 2 * self.Npulsars + len(self.upper_prior_lim_astro)
            )
        )
        print(important_idxs)
        [groups.append(important_idxs) for ii in range(2)]

        sampler = ptmcmc(
            ndim,
            self.get_lnliklihood,
            self.get_lnprior,
            cov,
            groups=groups,
            outDir=savedir,
            resume=resume,
        )

        sampler.addProposalToCycle(self.draw_from_prior, 15)

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
        param_idx = random.randint(0, len(self.upper_prior_lim_all) - 1)
        q[param_idx] = self.make_initial_guess()[param_idx]
        return q, float(lqxy)


class TwoModelHyperModel(object):
    """
    A class to perform a product-sapce sampling for two competing models.

    param model1: the first model
    param model2: the second model

    Author:
    Nima Laal (02/12/2025)
    """

    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2
        assert self.model1.Npulsars == self.model2.Npulsars
        self.combine_idxs
        self.lower_prior_lim_all = jnp.concatenate(
            (
                model1.run_type_object.lower_prior_lim_all,
                model2.run_type_object.lower_prior_lim_all[2 * model2.Npulsars :],
                jnp.array([0.0]),
            )
        )
        self.upper_prior_lim_all = jnp.concatenate(
            (
                model1.run_type_object.upper_prior_lim_all,
                model2.run_type_object.upper_prior_lim_all[2 * model2.Npulsars :],
                jnp.array([1.0]),
            )
        )

    def x_to_x1_indices(self):
        IRN_idxs = jnp.array(range(0, 2 * self.model1.Npulsars), dtype=int)
        a = self.model1.run_type_object.gwb_helper_dictionary["varied_gwb_psd_params"]
        gwb_idxs = jnp.array(
            range(2 * self.model1.Npulsars, 2 * self.model1.Npulsars + len(a)),
            dtype=int,
        )
        hm_idxs = jnp.concatenate((IRN_idxs, gwb_idxs))
        return hm_idxs

    def x_to_x2_indices(self):
        IRN_idxs = jnp.array(range(0, 2 * self.model2.Npulsars), dtype=int)
        a = self.model1.run_type_object.gwb_helper_dictionary["varied_gwb_psd_params"]
        b = self.model2.run_type_object.gwb_helper_dictionary["varied_gwb_psd_params"]
        gwb_idxs = jnp.array(
            range(
                2 * self.model2.Npulsars + len(a),
                2 * self.model2.Npulsars + len(a) + len(b),
            ),
            dtype=int,
        )
        hm_idxs = jnp.concatenate((IRN_idxs, gwb_idxs))
        return hm_idxs

    @cached_property
    def combine_idxs(self):
        x_to_x1_indices = self.x_to_x1_indices()
        x_to_x2_indices = self.x_to_x2_indices()
        len1 = x_to_x1_indices.shape[0]
        len2 = x_to_x2_indices.shape[0]

        if len1 > len2:
            # you have to pad the second array
            self.model1_idxs = x_to_x1_indices
            self.model2_idxs = jnp.pad(
                x_to_x2_indices, pad_width=(0, len1 - len2), constant_values=9999
            )
        elif len2 > len1:
            # you have to pad the first array
            self.model1_idxs = jnp.pad(
                x_to_x1_indices, pad_width=(0, len2 - len1), constant_values=9999
            )
            self.model2_idxs = x_to_x2_indices
        elif len1 == len2:
            self.model1_idxs = x_to_x1_indices
            self.model2_idxs = x_to_x2_indices

    def make_initial_guess(self, key):
        x0_0 = self.model1.run_type_object.make_initial_guess(key)
        x0_1 = self.model2.run_type_object.make_initial_guess(key)[
            2 * self.model2.Npulsars :
        ]
        nmodel = jr.uniform(key, shape=(1,), minval=0, maxval=1)
        return jnp.concatenate((x0_0, x0_1, nmodel))

    def get_lnliklihood(self, xs):
        nmodel = jnp.rint(xs[-1]).astype(bool)
        idxs = jax.lax.cond(nmodel, self.wrapper_model2_idxs, self.wrapper_model1_idxs)
        idxs_with_undone_padding = idxs[jnp.nonzero(idxs < 9999)]
        return jax.lax.cond(
            nmodel,
            self.model2.get_lnliklihood,
            self.model1.get_lnliklihood,
            xs[idxs_with_undone_padding],
        )

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

    def wrapper_model1_idxs(self):
        return self.model1_idxs

    def wrapper_model2_idxs(self):
        return self.model2_idxs

    def spit_neg_infinity(self):
        return -jnp.inf

    def spit_neg_number(self):
        return -8.01

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
    :param: matrix_stabilization: performing some matrix stabilization on the `TNT` matrix. It is highly recommended!

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
        TNr=jnp.array([False]),
        TNT=jnp.array([False]),
        noise_dict=None,
        backend="none",
        tnequad=False,
        inc_ecorr=False,
        del_pta_after_init=True,
        matrix_stabilization=True,
    ):
        assert jnp.any(TNr.any() and TNT.any()) or any(psrs), (
            "Either supply a `psrs` object or provide `TNr` and `TNT` arrays."
        )
        self.nf_dist = nf_dist
        self.astro_additional_prior_func = astro_additional_prior_func
        self.num_astro_params = num_astro_params
        self.run_type_object = run_type_object
        self.Tspan = run_type_object.Tspan
        self.noise_dict = noise_dict
        self.Npulsars = run_type_object.Npulsars
        self.renorm_const = run_type_object.renorm_const
        self.log_offset = 0.5 * np.log10(self.renorm_const)
        self.crn_bins = run_type_object.crn_bins
        self.int_bins = run_type_object.int_bins
        assert self.crn_bins <= self.int_bins
        self.kmax = 2 * self.int_bins
        self.diag_idx = jnp.arange(0, self.Npulsars)
        self.diag_idx_large = np.arange(0, self.Npulsars * self.kmax)
        self.k_idx = jnp.arange(0, self.kmax)
        self.c_idx = jnp.arange(0, self.crn_bins)
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

        if not TNr and not TNT:
            tm = gp_signals.MarginalizingTimingModel(use_svd=True)
            wn = blocks.white_noise_block(
                vary=False,
                inc_ecorr=inc_ecorr,
                gp_ecorr=False,
                select=backend,
                tnequad=tnequad,
            )
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
            s = tm + wn + rn + gwb

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
            print(
                f"Condition number of the TNT matrix before stabilizing is: {np.format_float_scientific(np.linalg.cond(self._TNT))}"
            )
            D = jnp.outer(
                jnp.sqrt(self._TNT.diagonal()), jnp.sqrt(self._TNT.diagonal())
            )
            corr = self._TNT / D
            corr = corr + 1e-3 * jnp.eye(self._TNT.shape[0])
            self._TNT = D * corr / (1 + 1e-3)
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
        return 0.5 * (jnp.dot(self._TNr, expval) - logdet_sigma - logdet_phi), psd_common

    def get_lnliklihood(self, xs):
        astro_params = xs[-self.num_astro_params:]
        xs_non_astro = jnp.array(xs[:-self.num_astro_params])
        lik0, psd_common = self.get_lnliklihood_non_astro(xs_non_astro)
        try:
            lik1 = self.nf_dist.log_prob((np.array(jnp.log10(psd_common)).T - self.log_offset), astro_params)
            return np.array(lik0) + lik1 + self.astro_additional_prior_func(astro_params)
        except ValueError:
            return -np.inf

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

    def make_initial_guess(self, seed = None):
        if seed:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        return rng.uniform(self.lower_prior_lim_all, self.upper_prior_lim_all)

    def sample(self, x0, niter, savedir, resume=True):
        """
        A function to perform the sampling using PTMCMC

        :param niter: the number of sampling iterations
        :param savedir: the directory to save the chains
        :param resume: do you want to resume from a saved chain?
        :param resume: annoying JAX rng key!
        """

        ndim = len(x0)
        cov = np.diag(np.ones(ndim) * 0.01**2)
        groups = [list(np.arange(0, ndim))]
        nonIR_idxs = np.array(
            range(
                2 * self.Npulsars, x0.shape[0]
            )
        )
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

        sampler.addProposalToCycle(self.draw_from_prior, 15)

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
        param_idx = random.randint(0, len(self.upper_prior_lim_all) - 1)
        q[param_idx] = self.make_initial_guess()[param_idx]
        return q, float(lqxy)