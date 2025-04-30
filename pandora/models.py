import numpy as np
from functools import cached_property, partial
import warnings, inspect, jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr

fref = 1 / (1 * 365.25 * 24 * 60 * 60)

class UniformPrior(object):
    """
    A class to take care of prior and the phi-matrix construction based 
    on log-uniform priors.
    NOTE: this class is for achromatic IRN + GWB

    :param gwb_psd_func: 
        a PSD function from the `GWBFunctions` class

    :param orf_func: 
        an orf function from the `GWBFunctions` class

    :param crn_bins: 
        number of frequency-bins for the GWB

    :param int_bins: 
        number of frequency-bins for the non-GWB (IRN) red noise

    :param `f_common`: 
        an array of frequency-bins for the common process in Hz

    :param `f_intrin`: 
        an array of frequency-bins for the IRN process in Hz

    :param df: 
        the diffence between consecutive frequency-bins. 
        It is usually 1/Tspan.

    :param psr_pos: 
        an array of pulsar-pair sky positions in cartesian-coordinates 
        (every other coordinate system is pretentious and hence not supported!)

    :param Tspan: 
        the baseline (time-span) of the PTA in seconds

    :param Npulsars: 
        number of pulsars in the PTA

    :param gwb_helper_dictionary: 
        the helper dictionary from `utils.py` script

    :param: first_crn_bin_index: 
        the first frequency-bin index (starting from zero) where the common proccess
        begins.

    :param gamma_min: 
        the lowest allowed value of the spectral-index of IRN

    :param gamma_max: 
        the highest allowed value of the spectral-index of IRN

    :param log10A_min: 
        the lowest allowed value of log10 of the amplitude of IRN

    :param log10A_max: 
        the highest allowed value of log10 of the amplitude of IRN

    :param renorm_const: 
        the factor by which the units are going to change. 
        Set it to `1` for no unit change.

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
        first_crn_bin_index = 0,
        gamma_min=1.,
        gamma_max=7.,
        log10A_min=-18.,
        log10A_max=-11.,
        renorm_const=1.,
        ):

        # Cache some useful constants/arrays
        self.gwb_helper_dictionary = gwb_helper_dictionary
        self.Npulsars = Npulsars
        self.psr_pos = psr_pos
        self.gwb_psd_func = gwb_psd_func
        self.orf_func = orf_func
        self.diag_idx = jnp.arange(0, self.Npulsars)
        self.ppair_number = int(self.Npulsars * (self.Npulsars - 1) * 0.5)
        self.num_IR_params = 2 * Npulsars
        self.renorm_const = renorm_const

        # Cache frequency related indices
        self.df = df
        self.Tspan = Tspan
        self.crn_bins = crn_bins
        self.int_bins = int_bins
        self.dm_bins = None
        self.f_common = f_common
        self.f_intrin = f_intrin
        self.f_dm = None
        self.GWB_fidxs = jnp.array(range(first_crn_bin_index, self.crn_bins + first_crn_bin_index))
        self.nonGWB_fidxs = jnp.array([ii for ii in range(self.int_bins) if not ii in self.GWB_fidxs])

        # Do we need to Cholesky invert everything?
        if self.nonGWB_fidxs.any():
            # Answer: no, if there are bins with only non-GWB noise
            # Mix Cholesky inverse with diagonal inverse
            self.separate_inversion_strat = True
        else:
            # Answer: yes, if there are bins with GWB noise
            # Stick to Cholesky inverse!
            self.separate_inversion_strat = False

        # Cache an identity matrix
        # This will be used to invert a triangular matrix
        self._eye = jnp.repeat(np.eye(self.Npulsars)[None], self.crn_bins, axis=0)
        
        # Below, calculates the angular separation as well as their indices (between pairs of pulsars) based on
        # their positions and stores them in the memory.
        I, J = np.tril_indices(self.Npulsars)
        a = np.zeros(self.ppair_number, dtype=int)
        b = np.zeros(self.ppair_number, dtype=int)
        ct = 0
        # I know there is a better way to do this, but this is the most readable way!
        for i, j in zip(I, J):
            if not i == j:
                a[ct] = i
                b[ct] = j
                ct += 1
        # `xi` is the angular separation
        self.xi = jnp.array(
            [np.arccos(np.dot(self.psr_pos[I], self.psr_pos[J])) for I, J in zip(a, b)]
        )
        # I and J are the cross-correlation indices
        # KGW is the GWB frequency indices
        # KIR is the non-GWB frequency indices 
        # (expanded up to `Npulsars` only because of the way it will be used!)
        # DIR is the diagonal indices for the IRN process
        # These are expanded forms of the indices needed
        # for multi-dimensional array indexing
        self.I = jnp.repeat(a[None, :], self.crn_bins, axis = 0)
        self.J = jnp.repeat(b[None, :], self.crn_bins, axis = 0)
        self.KGW = jnp.repeat(self.GWB_fidxs[:, None], len(a), axis = 1)
        if self.nonGWB_fidxs.any():
            self.DIR = jnp.repeat(self.diag_idx[None, :], len(self.nonGWB_fidxs), axis = 0)
        self.KIR = jnp.repeat(self.nonGWB_fidxs[:, None], self.Npulsars, axis = 1)

        # Cache prior related things...
        self.logrenorm_offset = 0.5 * jnp.log10(renorm_const)
        self.log10A_min = log10A_min + self.logrenorm_offset
        self.log10A_max = log10A_max + self.logrenorm_offset
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

        self.upper_prior_lim_all = jnp.zeros(self.num_IR_params)
        self.upper_prior_lim_all = self.upper_prior_lim_all.at[0::2].set(self.gamma_max)
        self.upper_prior_lim_all = self.upper_prior_lim_all.at[1::2].set(self.log10A_max)

        self.lower_prior_lim_all = jnp.zeros(self.num_IR_params)
        self.lower_prior_lim_all = self.lower_prior_lim_all.at[0::2].set(self.gamma_min)
        self.lower_prior_lim_all = self.lower_prior_lim_all.at[1::2].set(self.log10A_min)
        
        self.upper_prior_lim_all = jnp.append(
            self.upper_prior_lim_all, gwb_helper_dictionary["gwb_psd_param_upper_lim"]
        )
        self.lower_prior_lim_all = jnp.append(
            self.lower_prior_lim_all, gwb_helper_dictionary["gwb_psd_param_lower_lim"]
        )
        # Below, the array container/indices for the GWB PSD and ORF are made
        # and chaced. The chaced values help construct the red noise covaraince
        # matrix in a general way.
        psd_func_sigs = np.array(
            [
                str(_)
                for _ in inspect.signature(gwb_psd_func).parameters
                if not "args" in str(_)
            ][2:] #the first 2 are skipped because their are `f``, and `df``
        )
        orf_func_signs = np.array(
            [
                str(_)
                for _ in inspect.signature(orf_func).parameters
                if not "args" in str(_)
            ][1:] #the first one is skipped because it is `angle`
        )

        if "halflog10_rho" in psd_func_sigs:
            # The function signature of the free-spectral model
            #is different from others. Hence, it needs its own logic.
            self.param_value_container = jnp.zeros(self.crn_bins)
            self.gwb_psd_varied_param_indxs = jnp.array(
                [_ for _ in range(crn_bins)], dtype=int
            )
        else:
            # A few checks need to be done in case PSD params have parameters
            # with fixed values
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
        # This is the index where the ORF paraemters begin and PSD params end
        self.gwb_psd_params_end_idx = self.num_IR_params + len(
            self.gwb_psd_varied_param_indxs
        )

        # Some safeguards to protect the user from mis-specifying the models in the right order
        if not "halflog10_rho" in psd_func_sigs:
            assertion_psd_msg = f"""Your ordering of GWB PSD params is wrong! Check the `gwb_psd_func` signature.
            The signature demands {psd_func_sigs[self.gwb_psd_varied_param_indxs]}.
            You supplied {gwb_helper_dictionary["ordered_gwb_psd_model_params"][self.gwb_psd_varied_param_indxs]}."""
            assert np.all(
                gwb_helper_dictionary["ordered_gwb_psd_model_params"][
                    self.gwb_psd_varied_param_indxs
                ]
                == psd_func_sigs[self.gwb_psd_varied_param_indxs]
            ), assertion_psd_msg

        # Some safeguards to protect the user from mis-specifying the models in the right order for ORF
        if "ordered_orf_model_params" in gwb_helper_dictionary:
            assertion_orf_msg = f"""Your ordering of ORF params is wrong! Check the `orf_func` signature
            The signature demands {orf_func_signs}. You supplied {gwb_helper_dictionary["ordered_orf_model_params"]}."""
            assert (
                np.all(gwb_helper_dictionary["ordered_orf_model_params"] == orf_func_signs)
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
        # No need to use multi-dimensional indexing as the array is 2D
        # and the slices are 1D
        return phi_diag.at[self.GWB_fidxs].add(psd_common), psd_common

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
            # Need to use multi-dimensional indexing as the array is 3D
            # and the slices are 2D
            return phi.at[self.KGW, self.I, self.J].set(
                self.orf_val * psd_common
            )
        else:
            # Need to use multi-dimensional indexing as the array is 3D
            # and the slices are 2D
            return phi.at[self.KGW, self.I, self.J].set(
                self.orf_func(self.xi, *xs[self.gwb_psd_params_end_idx:]) * psd_common
            )

    @partial(jax.jit, static_argnums=(0,))
    def get_phi_mat_CURN(self, xs):
        """
        Constructs the phi-matrix based on the flattened array of model paraemters (`xs`)
        Since this is for a CURN run, correlations are ignored even if supplied!

        :param xs: flattened array of model paraemters (`xs`)

        :return: the phi-matrix` with dimensions `(n_f,n_p, n_p)`.
        """
        log10amp, gamma, gwb_psd_params = (
            xs[1 : self.num_IR_params : 2],
            xs[0 : self.num_IR_params : 2],
            xs[self.num_IR_params : self.gwb_psd_params_end_idx],
        )
        return self.get_phi_diag(log10amp, gamma, gwb_psd_params)

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
            # Need to use multi-dimensional indexing as the array is 3D
            # and the slices are 2D
            return phi.at[self.KGW, self.I, self.J].set(
                self.orf_val * psd_common
            ), psd_common
        else:
            # Need to use multi-dimensional indexing as the array is 3D
            # and the slices are 2D
            return phi.at[self.KGW, self.I, self.J].set(
                self.orf_func(self.xi, *xs[self.gwb_psd_params_end_idx:]) * psd_common
            ), psd_common

    @partial(jax.jit, static_argnums=(0,))
    def get_phi_mat_inv(self, phi):
        """
        Constructs the phiinv-matrix based on a given phi-matrix
        The correlated bins are treated differently from the 
        uncorrelated bins as the uncorrelated bins do not need
        a Cholesky decomposition.

        :param phi: red noise covaraince matrix

        :return: the phiinv-matrix` with dimensions `(2n_f,n_p, n_p)`
                as well as the log-determinat of phi
        """
        phiinv = jnp.zeros((self.int_bins, self.Npulsars, self.Npulsars))

        #correlated part needs Cholesky decomposition
        cp = jsp.linalg.cho_factor(phi[self.GWB_fidxs], lower=True)
        phiinv = phiinv.at[self.GWB_fidxs].set(jsp.linalg.cho_solve(cp, self._eye))
        logdet_phi = 2 * jnp.sum(jnp.log(cp[0].diagonal(axis1=-2, axis2=-1)))

        #uncorrelated part needs a simple inversion
        if self.separate_inversion_strat:
            diags = phi[self.nonGWB_fidxs].diagonal(axis1 = -2, axis2 = -1)
            return jnp.repeat(phiinv.at[self.KIR, self.DIR, self.DIR].set(1/diags), 2, axis = 0), \
                2 * (logdet_phi + jnp.sum(jnp.log(diags)))
        else:
            return jnp.repeat(phiinv, 2, axis = 0), 2 * logdet_phi

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
        This is for IRN only not GWB!

        :param log10_A: the logarithm (base 10) of the amplitude (IRN).
        :param gamma: the spectral-index of the powerlaw (IRN).

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

class UniformPriorDM(object):
    """
    A class to take care of prior and the phi-matrix construction based 
    on uniform/log-uniform priors.
    NOTE: this class is for achromatic IRN + DM + GWB

    :param gwb_psd_func: 
        a PSD function from the `GWBFunctions` class

    :param orf_func: 
        an orf function from the `GWBFunctions` class

    :param crn_bins: 
        number of frequency-bins for the GWB

    :param int_bins: 
        number of frequency-bins for the non-GWB (IRN) red noise

    :param dm_bins: 
        number of frequency-bins for the dm

    :param `f_common`: 
        an array of frequency-bins for the common process in Hz

    :param `f_intrin`: 
        an array of frequency-bins for the IRN process in Hz

    :param `f_dm`: 
        an array of frequency-bins for the dm process in Hz

    :param df: 
        the diffence between consecutive frequency-bins. 
        It is usually 1/Tspan.

    :param psr_pos: 
        an array of pulsar-pair sky positions in cartesian-coordinates 
        (every other coordinate system is pretentious and hence not supported!)

    :param Tspan: 
        the baseline (time-span) of the PTA in seconds

    :param Npulsars: 
        number of pulsars in the PTA

    :param gwb_helper_dictionary: 
        the helper dictionary from `utils.py` script

    :param: first_crn_bin_index: 
        the first frequency-bin index (starting from zero) where the common proccess
        begins.

    :param: first_irn_bin_index: 
        the first frequency-bin index (starting from zero) where the IRN proccess
        begins.

    :param: first_dm_bin_index: 
        the first frequency-bin index (starting from zero) where the dm proccess
        begins. Note, dm has its own basis. So, it will not interfere with GWB!

    :param gamma_min: 
        the lowest allowed value of the spectral-index of IRN

    :param gamma_max: 
        the highest allowed value of the spectral-index of IRN

    :param log10A_min: 
        the lowest allowed value of log10 of the amplitude of IRN

    :param log10A_max: 
        the highest allowed value of log10 of the amplitude of IRN

    :param gamma_dm_min: 
        the lowest allowed value of the spectral-index of dm

    :param gamma_dm_max: 
        the highest allowed value of the spectral-index of dm

    :param log10A_dm_min: 
        the lowest allowed value of log10 of the amplitude of dm

    :param log10A_dm_max: 
        the highest allowed value of log10 of the amplitude of dm

    :param renorm_const: 
        the factor by which the units are going to change. 
        Set it to `1` for no unit change.

    Author:
    Nima Laal (02/12/2025)
    """

    def __init__(
        self,
        gwb_psd_func,
        orf_func,
        crn_bins,
        int_bins,
        dm_bins,
        f_common,
        f_intrin,
        f_dm,
        df,
        psr_pos,
        Tspan,
        Npulsars,
        gwb_helper_dictionary,
        first_crn_bin_index,
        first_irn_bin_index,
        first_dm_bin_index,
        gamma_min=1.,
        gamma_max=7.,
        log10A_min=-18.,
        log10A_max=-11.,
        gamma_dm_min=1.,
        gamma_dm_max=7.,
        log10A_dm_min=-18.,
        log10A_dm_max=-11.,
        renorm_const=1.,
        ):

        # Cache some useful constants/arrays
        self.gwb_helper_dictionary = gwb_helper_dictionary
        self.Npulsars = Npulsars
        self.psr_pos = psr_pos
        self.gwb_psd_func = gwb_psd_func
        self.orf_func = orf_func
        self.diag_idx = jnp.arange(0, self.Npulsars)
        self.ppair_number = int(self.Npulsars * (self.Npulsars - 1) * 0.5)
        self.num_IR_params = 4 * Npulsars
        self.renorm_const = renorm_const

        # Cache frequency related indices
        self.df = df
        self.Tspan = Tspan
        self.crn_bins = crn_bins
        self.int_bins = int_bins
        self.dm_bins = dm_bins
        self.f_common = f_common
        self.f_intrin = f_intrin
        self.f_dm = f_dm

        self.first_irn_bin_index = first_irn_bin_index
        self.last_irn_bin_index = first_irn_bin_index + self.int_bins

        self.first_crn_bin_index = first_crn_bin_index
        self.last_crn_bin_index = first_crn_bin_index + self.crn_bins

        self.first_dm_bin_index = first_dm_bin_index
        self.last_dm_bin_index = first_dm_bin_index + self.dm_bins

        self.DM_fidxs = jnp.array([ii for ii in range(self.first_dm_bin_index, self.last_dm_bin_index)])
        self.GWB_fidxs = jnp.array(range(self.first_crn_bin_index, self.last_crn_bin_index))
        self.nonGWB_fidxs = jnp.array([ii for ii in range(self.first_irn_bin_index, self.last_irn_bin_index) if not ii in self.GWB_fidxs])

        # Cache an identity matrix
        self._eye = jnp.repeat(np.eye(self.Npulsars)[None], self.crn_bins, axis=0)

        # Do we need to Cholesky invert everything?
        if self.nonGWB_fidxs.any():
            # Answer: no, if there are bins with only non-GWB noise
            # Mix Cholesky inverse with diagonal inverse
            self.separate_inversion_strat = True
        else:
            # Answer: yes, if there are bins with GWB noise
            # Stick to Cholesky inverse!
            self.separate_inversion_strat = False

        # Below, calculates the angular separation as well as their indices (between pairs of pulsars) based on
        # their positions and stores them in the GPU/CPU memory.
        I, J = np.tril_indices(self.Npulsars)
        a = np.zeros(self.ppair_number, dtype=int)
        b = np.zeros(self.ppair_number, dtype=int)
        ct = 0
        # I know there is a better way to do this, but this is the most readable way!
        for i, j in zip(I, J):
            if not i == j:
                a[ct] = i
                b[ct] = j
                ct += 1
        # `xi` is the angular separation
        self.xi = jnp.array(
            [np.arccos(np.dot(self.psr_pos[I], self.psr_pos[J])) for I, J in zip(a, b)]
        )
        # I and J are the cross-correlation indices
        # KGW is the GWB frequency indices
        # KIR is the non-GWB frequency indices 
        # (expanded up to `Npulsars` only because of the way it will be used!)
        # DIR is the diagonal indices for the IRN process
        # These are expanded forms of the indices needed
        # for multi-dimensional array indexing
        self.I = jnp.repeat(a[None, :], self.crn_bins, axis = 0)
        self.J = jnp.repeat(b[None, :], self.crn_bins, axis = 0)
        self.KGW = jnp.repeat(self.GWB_fidxs[:, None], len(a), axis = 1)
        if self.nonGWB_fidxs.any():
            self.DIR = jnp.repeat(self.diag_idx[None, :], len(self.nonGWB_fidxs), axis = 0)
        self.KIR = jnp.repeat(self.nonGWB_fidxs[:, None], self.Npulsars, axis = 1)

        self.DIRDM = jnp.repeat(self.diag_idx[None, :], len(self.DM_fidxs), axis = 0)
        self.KDM = jnp.repeat(self.DM_fidxs[:, None], self.Npulsars, axis = 1)
        
        # Cache prior related things...
        self.logrenorm_offset = 0.5 * jnp.log10(renorm_const)
        self.log10A_min = log10A_min + self.logrenorm_offset
        self.log10A_max = log10A_max + self.logrenorm_offset
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.log10A_dm_min = log10A_dm_min + self.logrenorm_offset
        self.log10A_dm_max = log10A_dm_max + self.logrenorm_offset
        self.gamma_dm_min = gamma_dm_min
        self.gamma_dm_max = gamma_dm_max

        self.upper_prior_lim_all = jnp.zeros(self.num_IR_params)
        self.upper_prior_lim_all = self.upper_prior_lim_all.at[0::4].set(self.gamma_dm_max)
        self.upper_prior_lim_all = self.upper_prior_lim_all.at[1::4].set(self.log10A_dm_max)
        self.upper_prior_lim_all = self.upper_prior_lim_all.at[2::4].set(self.gamma_max)
        self.upper_prior_lim_all = self.upper_prior_lim_all.at[3::4].set(self.log10A_max)

        self.lower_prior_lim_all = jnp.zeros(self.num_IR_params)
        self.lower_prior_lim_all = self.lower_prior_lim_all.at[0::4].set(self.gamma_dm_min)
        self.lower_prior_lim_all = self.lower_prior_lim_all.at[1::4].set(self.log10A_dm_min)
        self.lower_prior_lim_all = self.lower_prior_lim_all.at[2::4].set(self.gamma_min)
        self.lower_prior_lim_all = self.lower_prior_lim_all.at[3::4].set(self.log10A_min)
        
        self.upper_prior_lim_all = jnp.append(
            self.upper_prior_lim_all, gwb_helper_dictionary["gwb_psd_param_upper_lim"]
        )
        self.lower_prior_lim_all = jnp.append(
            self.lower_prior_lim_all, gwb_helper_dictionary["gwb_psd_param_lower_lim"]
        )
        # Below, the array container/indices for the GWB PSD and ORF are made
        # and chaced. The chaced values help construct the red noise covaraince
        # matrix in a general way.
        psd_func_sigs = np.array(
            [
                str(_)
                for _ in inspect.signature(gwb_psd_func).parameters
                if not "args" in str(_)
            ][2:] #the first 2 are skipped because their are `f``, and `df``
        )
        orf_func_signs = np.array(
            [
                str(_)
                for _ in inspect.signature(orf_func).parameters
                if not "args" in str(_)
            ][1:] #the first one is skipped because it is `angle`
        )

        if "halflog10_rho" in psd_func_sigs:
            # The function signature of the free-spectral model
            #is different from others. Hence, it needs its own logic.
            self.param_value_container = jnp.zeros(self.crn_bins)
            self.gwb_psd_varied_param_indxs = jnp.array(
                [_ for _ in range(crn_bins)], dtype=int
            )
        else:
            # A few checks need to be done in case PSD params have parameters
            # with fixed values
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
        # This is the index where the ORF paraemters begin and PSD params end
        self.gwb_psd_params_end_idx = self.num_IR_params + len(
            self.gwb_psd_varied_param_indxs
        )

        # Some safeguards to protect the user from mis-specifying the models in the right order
        if not "halflog10_rho" in psd_func_sigs:
            assertion_psd_msg = f"""Your ordering of GWB PSD params is wrong! Check the `gwb_psd_func` signature.
            The signature demands {psd_func_sigs[self.gwb_psd_varied_param_indxs]}.
            You supplied {gwb_helper_dictionary["ordered_gwb_psd_model_params"][self.gwb_psd_varied_param_indxs]}."""
            assert np.all(
                gwb_helper_dictionary["ordered_gwb_psd_model_params"][
                    self.gwb_psd_varied_param_indxs
                ]
                == psd_func_sigs[self.gwb_psd_varied_param_indxs]
            ), assertion_psd_msg

        # Some safeguards to protect the user from mis-specifying the models in the right order for ORF
        if "ordered_orf_model_params" in gwb_helper_dictionary:
            assertion_orf_msg = f"""Your ordering of ORF params is wrong! Check the `orf_func` signature
            The signature demands {orf_func_signs}. You supplied {gwb_helper_dictionary["ordered_orf_model_params"]}."""
            assert (
                np.all(gwb_helper_dictionary["ordered_orf_model_params"] == orf_func_signs)
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
    def get_phi_diag(self, log10ampIRN, gammaIRN, 
                           log10ampDM, gammaDM,
                           gwb_psd_params):
        """
        Calculates the diagonal of the phi-matrix based on input parameters `log10amp`, `gamma`, and
        `gwb_psd_params`.

        :param log10ampIRN: the amplitude of the IRN
        :param gammaIRN: the spectral index of the IRN
        :param log10ampDM: the amplitude of the DM
        :param gammaDM: the spectral index of the DM
        :param gwb_psd_params: the psd parameters of the GWB

        :return:
        1. diagonal of the phi-matrix (IRN + GWB)
        2. diagonal of the phi-matrix (DM)
        3. the common process PSD
        """
        psd_common = self.gwb_psd_func(
            self.f_common,
            self.df,
            *self.param_value_container.at[self.gwb_psd_varied_param_indxs].set(
                gwb_psd_params
            ),
        )
        phi_irn = self.powerlaw(self.f_intrin, log10ampIRN, gammaIRN)
        phi_dm = self.powerlaw(self.f_dm, log10ampDM, gammaDM)
        return phi_dm, phi_irn, psd_common

    @partial(jax.jit, static_argnums=(0,))
    def get_phi_mat(self, xs):
        """
        Constructs the phi-matrix based on the flattened array of model paraemters (`xs`)

        :param xs: flattened array of model paraemters (`xs`)

        :return: the `phi-matrix` with dimensions `(n_f,n_p, n_p)`.
        """
        log10ampDM, gammaDM, log10ampIRN, gammaIRN, gwb_psd_params = (
            xs[1 : self.num_IR_params : 4],
            xs[0 : self.num_IR_params : 4],
            xs[3 : self.num_IR_params : 4],
            xs[2 : self.num_IR_params : 4],
            xs[self.num_IR_params : self.gwb_psd_params_end_idx],
        )
        phi_dm, phi_irn, psd_common = self.get_phi_diag(log10ampIRN, gammaIRN, 
                                                log10ampDM, gammaDM,
                                                gwb_psd_params)

        phi = jnp.zeros((self.dm_bins + self.int_bins, self.Npulsars, self.Npulsars))
        phi = phi.at[self.first_irn_bin_index:self.last_irn_bin_index, self.diag_idx, self.diag_idx].set(phi_irn)
        phi = phi.at[self.first_dm_bin_index:self.last_dm_bin_index, self.diag_idx, self.diag_idx].set(phi_dm)
        phi = phi.at[self.first_crn_bin_index:self.last_crn_bin_index, self.diag_idx, self.diag_idx].set(psd_common)

        if self.orf_fixed:
            return phi.at[self.KGW, self.I, self.J].set(
                self.orf_val * psd_common
            )
        else:
            return phi.at[self.KGW, self.I, self.J].set(
                self.orf_func(self.xi, *xs[self.gwb_psd_params_end_idx:]) * psd_common
            )

    @partial(jax.jit, static_argnums=(0,))
    def get_phi_mat_CURN(self, xs):
        """
        Constructs the phi-matrix based on the flattened array of model paraemters (`xs`)
        Since this is for a CURN run, correlations are ignored even if supplied!

        :param xs: flattened array of model paraemters (`xs`)

        :return: the phi-matrix` with dimensions `(n_f,n_p, n_p)`.
        """
        log10ampDM, gammaDM, log10ampIRN, gammaIRN, gwb_psd_params = (
            xs[1 : self.num_IR_params : 4],
            xs[0 : self.num_IR_params : 4],
            xs[3 : self.num_IR_params : 4],
            xs[2 : self.num_IR_params : 4],
            xs[self.num_IR_params : self.gwb_psd_params_end_idx],
        )
        phi_dm, phi_irn, psd_common = self.get_phi_diag(log10ampIRN, gammaIRN, 
                                                log10ampDM, gammaDM,
                                                gwb_psd_params)

        phi = jnp.zeros((self.dm_bins + self.int_bins, self.Npulsars))
        phi = phi.at[self.first_irn_bin_index:self.last_irn_bin_index].set(phi_irn)
        phi = phi.at[self.first_dm_bin_index:self.last_dm_bin_index].set(phi_dm)
        phi = phi.at[self.first_crn_bin_index:self.last_crn_bin_index].set(psd_common)
        return phi

    @partial(jax.jit, static_argnums=(0,))
    def get_phi_mat_and_common_psd(self, xs):
        """
        Constructs the phi-matrix based on the flattened array of model paraemters (`xs`)

        :param xs: flattened array of model paraemters (`xs`)

        :return: the phi-matrix` with dimensions `(n_f,n_p, n_p)`.
        """
        log10ampDM, gammaDM, log10ampIRN, gammaIRN, gwb_psd_params = (
            xs[1 : self.num_IR_params : 4],
            xs[0 : self.num_IR_params : 4],
            xs[3 : self.num_IR_params : 4],
            xs[2 : self.num_IR_params : 4],
            xs[self.num_IR_params : self.gwb_psd_params_end_idx],
        )
        phi_dm, phi_irn, psd_common = self.get_phi_diag(log10ampIRN, gammaIRN, 
                                                log10ampDM, gammaDM,
                                                gwb_psd_params)

        phi = jnp.zeros((self.dm_bins + self.int_bins, self.Npulsars, self.Npulsars))
        phi = phi.at[self.first_irn_bin_index:self.last_irn_bin_index, self.diag_idx, self.diag_idx].set(phi_irn)
        phi = phi.at[self.first_dm_bin_index:self.last_dm_bin_index, self.diag_idx, self.diag_idx].set(phi_dm)
        phi = phi.at[self.first_crn_bin_index:self.last_crn_bin_index, self.diag_idx, self.diag_idx].set(psd_common)

        if self.orf_fixed:
            return phi.at[self.KGW, self.I, self.J].set(
                self.orf_val * psd_common
            ), psd_common
        else:
            return phi.at[self.KGW, self.I, self.J].set(
                self.orf_func(self.xi, *xs[self.gwb_psd_params_end_idx:]) * psd_common
            ), psd_common

    @partial(jax.jit, static_argnums=(0,))
    def get_phi_mat_inv(self, phi):
        """
        Constructs the phiinv-matrix based on a given phi-matrix
        The correlated bins are treated differently from the 
        uncorrelated bins as the uncorrelated bins do not need
        a Cholesky decomposition.

        :param phi: red noise covaraince matrix

        :return: the phiinv-matrix` with dimensions `(2n_f,n_p, n_p)`
                as well as the log-determinat of phi
        """
        phiinv = jnp.zeros((self.dm_bins + self.int_bins, self.Npulsars, self.Npulsars))

        #correlated part needs Cholesky decomposition
        cp = jsp.linalg.cho_factor(phi[self.GWB_fidxs], lower=True)
        phiinv = phiinv.at[self.GWB_fidxs].set(jsp.linalg.cho_solve(cp, self._eye))
        logdet_phi = 2 * jnp.sum(jnp.log(cp[0].diagonal(axis1=-2, axis2=-1)))
        
        #DM PSD
        diags_dm = phi[self.DM_fidxs].diagonal(axis1 = -2, axis2 = -1)

        #uncorrelated part needs a simple inversion
        if self.separate_inversion_strat:
            diags = phi[self.nonGWB_fidxs].diagonal(axis1 = -2, axis2 = -1)

            phiinv = phiinv.at[self.KIR, self.DIR, self.DIR].set(1/diags)
            phiinv = phiinv.at[self.KDM, self.DIRDM, self.DIRDM].set(1/diags_dm)

            return jnp.repeat(phiinv, 2, axis = 0), \
                2 * (logdet_phi + jnp.sum(jnp.log(diags)) + jnp.sum(jnp.log(diags_dm)))
        else:
            return jnp.repeat(phiinv, 2, axis = 0), \
                2 * (logdet_phi + jnp.sum(jnp.log(diags_dm)))

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
    def powerlaw(self, freqs, log10_A, gamma):
        """
        Calculates a powerlaw expression given amplitude and spectral-index
        This is for IRN and DM only, not GWB!

        :param log10_A: the logarithm (base 10) of the amplitude (IRN or DM).
        :param gamma: the spectral-index of the powerlaw (IRN or DM).

        :return: a powerlaw expression on predefined frequencies.
        """
        return (
            10 ** (2 * log10_A)
            / (12 * jnp.pi**2 * freqs[:, None] ** 3 * self.Tspan)
            * (freqs[:, None] / fref) ** (3 - gamma)
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

class UniformPriorGwbOnly(object):
    """
    A class to take care of prior and the phi-matrix construction based 
    on uniform/log-uniform priors. Only GWB is modeled.
    NOTE: this class is used for very specific purposes where there is no IRN.
    You should not be using this class if you want to model IRN.
    This class is a stripped down version of the `UniformPrior`
    class. For more documentation details, refer to `UniformPrior`.

    :param gwb_psd_func: 
        a PSD function from the `GWBFunctions` class

    :param orf_func: 
        an orf function from the `GWBFunctions` class

    :param crn_bins: 
        number of frequency-bins for the GWB

    :param `f_common`: 
        an array of frequency-bins for the common process in Hz

    :param df: 
        the diffence between consecutive frequency-bins. 
        It is usually 1/Tspan.

    :param psr_pos: 
        an array of pulsar-pair sky positions in cartesian-coordinates 
        (every other coordinate system is pretentious and hence not supported!)

    :param Tspan: 
        the baseline (time-span) of the PTA in seconds

    :param Npulsars: 
        number of pulsars in the PTA

    :param gwb_helper_dictionary: 
        the helper dictionary from `utils.py` script
        
    :param renorm_const: 
        the factor by which the units are going to change. 
        Set it to `1` for no unit change.

    Author:
    Nima Laal (02/12/2025)
    """

    def __init__(
        self,
        gwb_psd_func,
        orf_func,
        crn_bins,
        f_common,
        df,
        psr_pos,
        Tspan,
        Npulsars,
        gwb_helper_dictionary,
        renorm_const=1.,
    ):
        self.gwb_helper_dictionary = gwb_helper_dictionary
        self.Npulsars = Npulsars
        self.psr_pos = psr_pos
        self.gwb_psd_func = gwb_psd_func
        self.orf_func = orf_func
        self.diag_idx = jnp.arange(0, self.Npulsars)
        self.renorm_const = renorm_const

        self.crn_bins = crn_bins
        self.int_bins = crn_bins
        self.dm_bins = None
        self.f_common = f_common
        self.df = df
        self.Tspan = Tspan

        self.logrenorm_offset = 0.5 * jnp.log10(renorm_const)
        self.ppair_number = int(self.Npulsars * (self.Npulsars - 1) * 0.5)

        self.upper_prior_lim_all = jnp.array(gwb_helper_dictionary["gwb_psd_param_upper_lim"])
        self.lower_prior_lim_all = jnp.array(gwb_helper_dictionary["gwb_psd_param_lower_lim"])
        self.num_IR_params = 0

        psd_func_sigs = np.array(
            [
                str(_)
                for _ in inspect.signature(gwb_psd_func).parameters
                if not "args" in str(_)
            ][2:]
        )
        orf_func_signs = np.array(
            [
                str(_)
                for _ in inspect.signature(orf_func).parameters
                if not "args" in str(_)
            ][1:]
        )

        if "halflog10_rho" in psd_func_sigs:
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

        self.gwb_psd_params_end_idx = len(
            self.gwb_psd_varied_param_indxs
        )

        if not "halflog10_rho" in psd_func_sigs:
            assertion_psd_msg = f"""Your ordering of GWB PSD params is wrong! Check the `gwb_psd_func` signature.
            The signature demands {psd_func_sigs[self.gwb_psd_varied_param_indxs]}. You supplied {gwb_helper_dictionary["ordered_gwb_psd_model_params"][self.gwb_psd_varied_param_indxs]}."""
            assert np.all(
                gwb_helper_dictionary["ordered_gwb_psd_model_params"][
                    self.gwb_psd_varied_param_indxs
                ]
                == psd_func_sigs[self.gwb_psd_varied_param_indxs]
            ), assertion_psd_msg

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

        # Below, calculates the angular separation as well as their indices (between pairs of pulsars) based on
        # their positions and stores them in the GPU/CPU memory.
        I, J = np.tril_indices(self.Npulsars)
        a = np.zeros(self.ppair_number, dtype=int)
        b = np.zeros(self.ppair_number, dtype=int)
        ct = 0
        # I know there is a better way to do this, but this is the most readable way!
        for i, j in zip(I, J):
            if not i == j:
                a[ct] = i
                b[ct] = j
                ct += 1
        # `xi` is the angular separation
        self.xi = jnp.array(
            [np.arccos(np.dot(self.psr_pos[I], self.psr_pos[J])) for I, J in zip(a, b)]
        )
        # I and J are the cross-correlation indices
        self.I = jnp.array(a)
        self.J = jnp.array(b)

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
    def get_phi_diag(self, gwb_psd_params):
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
        return psd_common, psd_common 

    @partial(jax.jit, static_argnums=(0,))
    def get_phi_mat(self, xs):
        """
        Constructs the phi-matrix based on the flattened array of model paraemters (`xs`)

        :param xs: flattened array of model paraemters (`xs`)

        :return: the phi-matrix` with dimensions `(n_f,n_p, n_p)`.
        """
        phi_diag, psd_common = self.get_phi_diag(xs[: self.gwb_psd_params_end_idx])
        phi = jnp.zeros((self.crn_bins, self.Npulsars, self.Npulsars))
        phi = phi.at[:, self.diag_idx, self.diag_idx].set(phi_diag)
        if self.orf_fixed:
            return phi.at[: self.crn_bins, self.I, self.J].set(
                self.orf_val * psd_common
            )
        else:
            return phi.at[:, self.I, self.J].set(
                self.orf_func(self.xi, *xs[self.gwb_psd_params_end_idx :]) * psd_common
            )

    @partial(jax.jit, static_argnums=(0,))
    def get_phi_mat_CURN(self, xs):
        """
        Constructs the phi-matrix based on the flattened array of model paraemters (`xs`)

        :param xs: flattened array of model paraemters (`xs`)

        :return: the phi-matrix` with dimensions `(n_f,n_p, n_p)`.
        """
        return self.get_phi_diag(xs[: self.gwb_psd_params_end_idx])


    @partial(jax.jit, static_argnums=(0,))
    def get_phi_mat_and_common_psd(self, xs):
        """
        Constructs the phi-matrix based on the flattened array of model paraemters (`xs`)

        :param xs: flattened array of model paraemters (`xs`)

        :return: the phi-matrix` with dimensions `(n_f,n_p, n_p)`.
        """
        phi_diag, psd_common = self.get_phi_diag(xs[: self.gwb_psd_params_end_idx])
        phi = jnp.zeros((self.crn_bins, self.Npulsars, self.Npulsars))
        phi = phi.at[:, self.diag_idx, self.diag_idx].set(phi_diag)
        if self.orf_fixed:
            return phi.at[: self.crn_bins, self.I, self.J].set(
                self.orf_val * psd_common
            ), psd_common
        else:
            return phi.at[:, self.I, self.J].set(
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