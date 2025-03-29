import numpy as np
from pandora import GWBFunctions
import inspect
import jax.numpy as jnp
import jax

# jax.config.update('jax_platform_name', 'cpu')
# jax.config.update("jax_enable_x64", False)
##############################################################################################


def param_order_help(
    lower_bound_array,
    upper_bound_array,
    list_of_orf_params=[None],
    list_of_psd_params=["log10_A", "gamma"],
    fixed_gwb_psd_params=[None],
    fixed_gwb_psd_param_values=[None],
):
    """
    A utility function that helps organize and structure parameters
    related to gravitational wave background (GWB) PSD and ORF models.

    :param lower_bound_array: the lower bound on the model params (PSD + ORF) as a JAX array
    :param upper_bound_array: the upper bound on the model params (PSD + ORF) as a JAX array
    :list_of_orf_params: a list containing the name of the ORF model parameters. The ordering
      ***MUST*** match those in `GWBFunctions.py`
    :list_of_psd_params: a list containing the name of the PSD model parameters. The ordering
      ***MUST*** match those in `GWBFunctions.py`
    :fixed_gwb_psd_params: a list containing the GWB PSD parameters that you want to be fixed
    :fixed_gwb_psd_param_values: a JAX array containing the values of fixed GWB PSD params.

    """
    x = {}
    x.update({"ordered_gwb_psd_model_params": list_of_psd_params})
    if any(fixed_gwb_psd_params):
        fixed_gwb_psd_param_indxs = [
            list(list_of_psd_params).index(_) for _ in fixed_gwb_psd_params
        ]
        x.update({"fixed_gwb_psd_params": fixed_gwb_psd_params})
    if any(list_of_orf_params):
        x.update({"ordered_orf_model_params": list_of_orf_params})
    x.update(
        {
            "varied_gwb_psd_params": [
                *[_ for _ in list_of_psd_params if _ not in fixed_gwb_psd_params],
                *list_of_orf_params,
            ]
        }
    )
    x.update({"gwb_psd_param_lower_lim": lower_bound_array})
    x.update({"gwb_psd_param_upper_lim": upper_bound_array})
    if any(fixed_gwb_psd_params):
        x.update({"fixed_gwb_psd_params": fixed_gwb_psd_params})
        x.update({"fixed_gwb_psd_param_indices": jnp.array(fixed_gwb_psd_param_indxs)})
        x.update({"fixed_gwb_psd_param_values": fixed_gwb_psd_param_values})
    return x


##############################################################################################


def fixed_gamma_hd_pl(renorm_const, lower_amp=-18.0, upper_amp=-11.0):
    """
    A lazy way to get the right `param_order_help` dictionary for a fixed gamma HD model
    """
    logamp_offset = logamp_offset = 0.5 * jnp.log10(renorm_const)
    chosen_psd_model = GWBFunctions.powerlaw
    chosen_orf_model = GWBFunctions.hd_orf
    chosen_psd_model_params = np.array(
        [
            str(_)
            for _ in inspect.signature(chosen_psd_model).parameters
            if not "args" in str(_)
        ][2:]
    )
    return (
        chosen_psd_model,
        chosen_orf_model,
        param_order_help(
            list_of_psd_params=chosen_psd_model_params,
            lower_bound_array=jnp.array([lower_amp + logamp_offset]),
            upper_bound_array=jnp.array([upper_amp + logamp_offset]),
            fixed_gwb_psd_params=["gamma"],
            fixed_gwb_psd_param_values=jnp.array([13 / 3]),
            list_of_orf_params=[],
        ),
    )

##############################################################################################


def broken_pl(renorm_const, lower_amp=-18.0, upper_amp=-11.0, 
                                         lower_gamma = 0., upper_gamma = 7.,
                                         lower_log10_fb = -8.7, upper_log10_fb = -7.
                            ):
    """
    A lazy way to get the right `param_order_help` dictionary for a fixed gamma HD model
    """
    logamp_offset = logamp_offset = 0.5 * jnp.log10(renorm_const)
    chosen_psd_model = GWBFunctions.broken_powerlaw
    chosen_orf_model = GWBFunctions.hd_orf
    chosen_psd_model_params = np.array(
        [
            str(_)
            for _ in inspect.signature(chosen_psd_model).parameters
            if not "args" in str(_)
        ][2:]
    )
    return (
        chosen_psd_model,
        chosen_orf_model,
        param_order_help(
            list_of_psd_params=chosen_psd_model_params,
            lower_bound_array=jnp.array([lower_amp + logamp_offset, lower_gamma, lower_log10_fb]),
            upper_bound_array=jnp.array([upper_amp + logamp_offset, upper_gamma, upper_log10_fb]),
            fixed_gwb_psd_params=["delta", "kappa"],
            fixed_gwb_psd_param_values=jnp.array([0., 0.1]),
            list_of_orf_params=[],
        ),
    )


##############################################################################################

def varied_gamma_hd_pl(
    renorm_const, lower_amp=-18.0, upper_amp=-11.0, lower_gamma=0.0, upper_gamma=7.0
):
    """
    A lazy way to get the right `param_order_help` dictionary for a varied gamma HD model
    """
    logamp_offset = 0.5 * jnp.log10(renorm_const)
    chosen_psd_model = GWBFunctions.powerlaw
    chosen_orf_model = GWBFunctions.hd_orf
    chosen_psd_model_params = np.array(
        [
            str(_)
            for _ in inspect.signature(chosen_psd_model).parameters
            if not "args" in str(_)
        ][2:]
    )
    return (
        chosen_psd_model,
        chosen_orf_model,
        param_order_help(
            list_of_psd_params=chosen_psd_model_params,
            lower_bound_array=jnp.array([lower_amp + logamp_offset, lower_gamma]),
            upper_bound_array=jnp.array([upper_amp + logamp_offset, upper_gamma]),
            fixed_gwb_psd_param_values=[],
            list_of_orf_params=[],
        ),
    )


##############################################################################################


def hd_spectrum(renorm_const, crn_bins, lower_halflog10_rho=-9, upper_halflog10_rho=-1):
    """
    A lazy way to get the right `param_order_help` dictionary for a free-spectral HD model
    """
    logamp_offset = 0.5 * jnp.log10(renorm_const)
    chosen_psd_model = GWBFunctions.free_spectrum
    chosen_orf_model = GWBFunctions.hd_orf
    chosen_psd_model_params = np.array(
        [
            str(_)
            for _ in inspect.signature(chosen_psd_model).parameters
            if not "args" in str(_)
        ][2:]
    )
    return (
        chosen_psd_model,
        chosen_orf_model,
        param_order_help(
            list_of_psd_params=chosen_psd_model_params,
            lower_bound_array=jnp.ones(crn_bins)
            * (lower_halflog10_rho + logamp_offset),
            upper_bound_array=jnp.ones(crn_bins)
            * (upper_halflog10_rho + logamp_offset),
            fixed_gwb_psd_param_values=[],
            list_of_orf_params=[],
        ),
    )


##############################################################################################


def varied_gamma_gt_pl(
    renorm_const, lower_amp=-18.0, upper_amp=-11.0, lower_gamma=0.0, upper_gamma=7.0
):
    """
    A lazy way to get the right `param_order_help` dictionary for a varied gamma GT model
    """
    logamp_offset = 0.5 * jnp.log10(renorm_const)
    chosen_psd_model = GWBFunctions.powerlaw
    chosen_psd_model_params = np.array(
        [
            str(_)
            for _ in inspect.signature(chosen_psd_model).parameters
            if not "args" in str(_)
        ][2:]
    )

    chosen_orf_model = GWBFunctions.gt_orf
    chosen_orf_model_params = np.array(
        [
            str(_)
            for _ in inspect.signature(chosen_orf_model).parameters
            if not "args" in str(_)
        ][1:]
    )

    return (
        chosen_psd_model,
        chosen_orf_model,
        param_order_help(
            list_of_psd_params=chosen_psd_model_params,
            lower_bound_array=jnp.array([lower_amp + logamp_offset, lower_gamma, -1.5]),
            upper_bound_array=jnp.array([upper_amp + logamp_offset, upper_gamma, 1.5]),
            fixed_gwb_psd_param_values=[],
            list_of_orf_params=chosen_orf_model_params,
        ),
    )


##############################################################################################


def varied_gamma_bin_orf_pl(
    renorm_const, lower_amp=-18.0, upper_amp=-11.0, lower_gamma=0.0, upper_gamma=7.0
):
    """
    A lazy way to get the right `param_order_help` dictionary for a varied gamma bin_orf model
    """
    logamp_offset = 0.5 * jnp.log10(renorm_const)
    chosen_psd_model = GWBFunctions.powerlaw
    chosen_psd_model_params = np.array(
        [
            str(_)
            for _ in inspect.signature(chosen_psd_model).parameters
            if not "args" in str(_)
        ][2:]
    )

    chosen_orf_model = GWBFunctions.bin_orf
    chosen_orf_model_params = np.array(
        [
            str(_)
            for _ in inspect.signature(chosen_orf_model).parameters
            if not "args" in str(_)
        ][1:]
    )

    return (
        chosen_psd_model,
        chosen_orf_model,
        param_order_help(
            list_of_psd_params=chosen_psd_model_params,
            lower_bound_array=jnp.array(
                [
                    lower_amp + logamp_offset,
                    lower_gamma,
                    *jnp.ones(len(GWBFunctions.bins_decent)) * -1.0,
                ]
            ),
            upper_bound_array=jnp.array(
                [
                    upper_amp + logamp_offset,
                    upper_gamma,
                    *jnp.ones(len(GWBFunctions.bins_decent)) * 1.0,
                ]
            ),
            fixed_gwb_psd_param_values=[],
            list_of_orf_params=chosen_orf_model_params,
        ),
    )


##############################################################################################
