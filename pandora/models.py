import numpy as np
from functools import cached_property, partial
import warnings, inspect, jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
import torch

# jax.config.update('jax_platform_name', 'cpu')
# jax.config.update("jax_enable_x64", False)

fref = 1 / (1 * 365.25 * 24 * 60 * 60)


class UniformPrior(object):
    """
    A class to take care of prior and the phi-matrix construction based on uniform/log-uniform priors.

    :param gwb_psd_func: a PSD function from the `GWBFunctions` class
    :param orf_func: an orf function from the `GWBFunctions` class
    :param crn_bins: number of frequency-bins for the GWB
    :param int_bins: number of frequency-bins for the non-GWB (IRN) red noise
    :param `f_common`: an array of frequency-bins for the common process
    :param `f_intrin`: an array of frequency-bins for the IRN process
    :param df: the diffence between consecutive frequency-bins. It is usually 1/Tspan
    :param psr_pos: an array of pulsar-pair sky positions in cartesian-coordinates (every other coordinate system is pretentious and hence not supported!)
    :param Tspan: the baseline (time-span) of the PTA
    :param Npulsars: number of pulsars in the PTA
    :param gwb_helper_dictionary: the helper dictionary from `utils.py` script
    :param gamma_min: the lowest allowed value of the spectral-index of IRN
    :param gamma_max: the highest allowed value of the spectral-index of IRN
    :param log10A_min: the lowest allowed value of log10 of the amplitude of IRN
    :param log10A_max: the highest allowed value of log10 of the amplitude of IRN
    :param renorm_const: the factor by which the units are going to change. Set it to `1` for no unit change, or let it be `1e9` for better performance.

    Author:
    Nima Laal (02/12/2025)
    """

    def __init__(
        self,
        gwb_psd_func,
        orf_func,
        crn_bins,
        int_bins,
        f_common,
        f_intrin,
        df,
        psr_pos,
        Tspan,
        Npulsars,
        gwb_helper_dictionary,
        gamma_min=0,
        gamma_max=7,
        log10A_min=-18 + 0.5 * jnp.log10(1e9),
        log10A_max=-11 + 0.5 * jnp.log10(1e9),
        renorm_const=1e9,
    ):
        self.gwb_helper_dictionary = gwb_helper_dictionary
        self.Npulsars = Npulsars
        self.psr_pos = psr_pos
        self.gwb_psd_func = gwb_psd_func
        self.orf_func = orf_func
        self.diag_idx = jnp.arange(0, self.Npulsars)
        self.num_IR_params = 2 * Npulsars
        self.renorm_const = renorm_const

        self.crn_bins = crn_bins
        self.int_bins = int_bins
        self.f_common = f_common
        self.f_intrin = f_intrin
        self.df = df
        self.Tspan = Tspan

        self.logrenorm_offset = 0.5 * jnp.log10(renorm_const)
        self.ppair_idx = jnp.arange(0, int(self.Npulsars * (self.Npulsars - 1) * 0.5))

        self.log10A_min = log10A_min + self.logrenorm_offset
        self.log10A_max = log10A_max + self.logrenorm_offset

        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

        self.upper_prior_lim_all = jnp.zeros(2 * self.Npulsars)
        self.upper_prior_lim_all = self.upper_prior_lim_all.at[0::2].set(gamma_max)
        self.upper_prior_lim_all = self.upper_prior_lim_all.at[1::2].set(
            self.log10A_max
        )

        self.lower_prior_lim_all = jnp.zeros(2 * self.Npulsars)
        self.lower_prior_lim_all = self.lower_prior_lim_all.at[0::2].set(gamma_min)
        self.lower_prior_lim_all = self.lower_prior_lim_all.at[1::2].set(
            self.log10A_min
        )

        self.upper_prior_lim_all = jnp.append(
            self.upper_prior_lim_all, gwb_helper_dictionary["gwb_psd_param_upper_lim"]
        )
        self.lower_prior_lim_all = jnp.append(
            self.lower_prior_lim_all, gwb_helper_dictionary["gwb_psd_param_lower_lim"]
        )

        psd_func_sigs = np.array(
            [str(_) for _ in inspect.signature(gwb_psd_func).parameters if not 'args' in str(_)][2:]
        )
        orf_func_signs = np.array(
            [str(_) for _ in inspect.signature(orf_func).parameters if not 'args' in str(_)][1:]
        )

        if 'halflog10_rho' in psd_func_sigs:
            self.param_value_container = jnp.zeros(self.crn_bins)
            self.gwb_psd_varied_param_indxs = jnp.array(
                    [_ for _ in range(crn_bins)], dtype=int
                )
        else:
            self.param_value_container = jnp.zeros(len(psd_func_sigs))
            if "fixed_gwb_psd_param_indices" in gwb_helper_dictionary:
                fixed_gwb_psd_param_indxs = gwb_helper_dictionary[
                    "fixed_gwb_psd_param_indices"
                ]
                fixed_gwb_psd_param_values = gwb_helper_dictionary[
                    "fixed_gwb_psd_param_values"
                ]
                self.param_value_container = self.param_value_container.at[
                    fixed_gwb_psd_param_indxs
                ].set(fixed_gwb_psd_param_values)
                self.gwb_psd_varied_param_indxs = jnp.array(
                    [
                        _
                        for _ in range(len(psd_func_sigs))
                        if not _ in fixed_gwb_psd_param_indxs
                    ],
                    dtype=int,
                )
            else:
                self.gwb_psd_varied_param_indxs = jnp.array(
                    [_ for _ in range(len(psd_func_sigs))], dtype=int
                )

        self.gwb_psd_params_end_idx = self.num_IR_params + len(
            self.gwb_psd_varied_param_indxs
        )

        if not 'halflog10_rho' in psd_func_sigs:
            assertion_psd_msg = f"""Your ordering of GWB PSD params is wrong! Check the `gwb_psd_func` signature.
            The signature demands {psd_func_sigs[self.gwb_psd_varied_param_indxs]}. You supplied {gwb_helper_dictionary["ordered_gwb_psd_model_params"][self.gwb_psd_varied_param_indxs]}."""
            assert np.all(
                gwb_helper_dictionary["ordered_gwb_psd_model_params"][
                    self.gwb_psd_varied_param_indxs
                ]
                == psd_func_sigs[self.gwb_psd_varied_param_indxs]
            ), assertion_psd_msg

        self.get_cross_info
        if "ordered_orf_model_params" in gwb_helper_dictionary:
            assertion_orf_msg = f"""Your ordering of ORF params is wrong! Check the `orf_func` signature
            The signature demands {orf_func_signs}. You supplied {gwb_helper_dictionary["ordered_orf_model_params"]}."""
            assert (
                gwb_helper_dictionary["ordered_orf_model_params"] == orf_func_signs
            ), assertion_orf_msg
            self.orf_fixed = False
        else:
            self.orf_val = orf_func(self.xi)
            self.orf_fixed = True

        if renorm_const != 1:
            warnings.warn(
                "You have chosen to change units. Make sure your amplitude priors reflect that!"
            )

    def jax_to_numpy_CPU(self, jax_CPU_array):
        """
        Converts a JAX CPU array to a NumPy array.

        :param: jax_CPU_array: a JAX array that is stored on the CPU.

        :return: a NumPy array converted from a JAX CPU array using the `np.from_dlpack()` function.
        """
        return np.from_dlpack(jax_CPU_array)

    @cached_property
    def get_cross_info(self):
        """
        Calculates the angular separation as well as their indices (between pairs of pulsars) based on
        their positions and stores them in the memory.
        """

        I, J = np.tril_indices(self.Npulsars)
        a = np.zeros(len(self.ppair_idx), dtype=int)
        b = np.zeros(len(self.ppair_idx), dtype=int)
        ct = 0
        for i, j in zip(I, J):
            if not i == j:
                a[ct] = i
                b[ct] = j
                ct += 1
        self.xi = jnp.array(
            [np.arccos(np.dot(self.psr_pos[I], self.psr_pos[J])) for I, J in zip(a, b)]
        )
        self.I = jnp.array(a)
        self.J = jnp.array(b)

    @partial(jax.jit, static_argnums=(0,))
    def make_initial_guess(self, key):
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
        return jr.uniform(
            key,
            shape=(self.upper_prior_lim_all.shape[0],),
            minval=self.lower_prior_lim_all,
            maxval=self.upper_prior_lim_all,
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_phi_diag(self, log10amp, gamma, gwb_psd_params):
        """
        Calculates the diagonal of the phi-matrix based on input parameters `log10amp`, `gamma`, and
        `gwb_psd_params`.

        :param log10amp: the amplitude of the IRN
        :param gamma: the spectral index of IRN
        :param gwb_psd_params: the psd parameters of the GWB

        :return:
        1. diagonal of the phi-matrix
        2. the common process PSD
        """
        psd_common = self.gwb_psd_func(
            self.f_common,
            self.df,
            *self.param_value_container.at[self.gwb_psd_varied_param_indxs].set(
                gwb_psd_params
            ),
        )
        phi_diag = self.powerlaw(log10amp, gamma)
        return phi_diag.at[: self.crn_bins].add(psd_common), psd_common

    @partial(jax.jit, static_argnums=(0,))
    def get_phi_mat(self, xs):
        """
        Constructs the phi-matrix based on the flattened array of model paraemters (`xs`)

        :param xs: flattened array of model paraemters (`xs`)

        :return: the phi-matrix` with dimensions `(n_f,n_p, n_p)`.
        """
        log10amp, gamma, gwb_psd_params = (
            xs[1 : self.num_IR_params : 2],
            xs[0 : self.num_IR_params : 2],
            xs[self.num_IR_params : self.gwb_psd_params_end_idx],
        )
        phi_diag, psd_common = self.get_phi_diag(log10amp, gamma, gwb_psd_params)
        phi = jnp.zeros((self.int_bins, self.Npulsars, self.Npulsars))
        phi = phi.at[:, self.diag_idx, self.diag_idx].set(phi_diag)
        if self.orf_fixed:
            return phi.at[: self.crn_bins, self.I, self.J].set(
                self.orf_val * psd_common
            )
        else:
            return phi.at[: self.crn_bins, self.I, self.J].set(
                self.orf_func(self.xi, *xs[self.gwb_psd_params_end_idx :]) * psd_common
            )

    @partial(jax.jit, static_argnums=(0,))
    def get_phi_mat_and_common_psd(self, xs):
        """
        Constructs the phi-matrix based on the flattened array of model paraemters (`xs`)

        :param xs: flattened array of model paraemters (`xs`)

        :return: the phi-matrix` with dimensions `(n_f,n_p, n_p)`.
        """
        log10amp, gamma, gwb_psd_params = (
            xs[1 : self.num_IR_params : 2],
            xs[0 : self.num_IR_params : 2],
            xs[self.num_IR_params : self.gwb_psd_params_end_idx],
        )
        phi_diag, psd_common = self.get_phi_diag(log10amp, gamma, gwb_psd_params)
        phi = jnp.zeros((self.int_bins, self.Npulsars, self.Npulsars))
        phi = phi.at[:, self.diag_idx, self.diag_idx].set(phi_diag)
        if self.orf_fixed:
            return phi.at[: self.crn_bins, self.I, self.J].set(
                self.orf_val * psd_common
            ), psd_common
        else:
            return phi.at[: self.crn_bins, self.I, self.J].set(
                self.orf_func(self.xi, *xs[self.gwb_psd_params_end_idx :]) * psd_common
            ), psd_common
        
    @partial(jax.jit, static_argnums=(0,))
    def get_lnprior(self, xs):
        """
        A function to return natural log uniform-prior

        :param: xs: flattened array of model paraemters (`xs`)

        :return: either -infinity or a constant based on the predefined limits of the uniform-prior.
        """
        state = jnp.logical_and(
            xs > self.lower_prior_lim_all, xs < self.upper_prior_lim_all
        ).all()
        return jax.lax.cond(state, self.spit_neg_number, self.spit_neg_infinity)

    @partial(jax.jit, static_argnums=(0,))
    def powerlaw(self, log10_A, gamma):
        """
        Calculates a powerlaw expression given amplitude and spectral-index

        :param log10_A: the logarithm (base 10) of the amplitude.
        :param gamma: the spectral-index of the powerlaw

        :return: a powerlaw expression on predefined frequencies.
        """
        return (
            10 ** (2 * log10_A)
            / (12 * jnp.pi**2 * self.f_intrin[:, None] ** 3 * self.Tspan)
            * (self.f_intrin[:, None] / fref) ** (3 - gamma)
        )

    def spit_neg_infinity(self):
        """
        returns negative infinity using the JAX NumPy library.

        :return: negative infinity.
        """
        return -jnp.inf

    def spit_neg_number(self):
        """
        returns the negative value -8.01 (an arbitrary constant used by the uniform prior).

        :return: returnins the value -8.01.
        """
        return -8.01

    