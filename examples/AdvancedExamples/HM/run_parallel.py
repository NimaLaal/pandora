# Load Needed Packages
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jar
# jax.config.update('jax_platform_name', 'cuda')
# jax.config.update("jax_enable_x64", False)
import numpy as np
from pandora import models, utils, GWBFunctions
from pandora import LikelihoodCalculator as LC
from enterprise_extensions.model_utils import get_tspan
import pickle, json, os, corner, glob, random, copy, time, inspect
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.lines as mlines
import ray

plt.style.use('dark_background')
hist_settings = dict(
    bins = 40,
    histtype = 'step',
    lw = 3,
    density = True
)
ray.init()
@ray.remote(num_gpus=.33)
def doit(idx):

    datadir = '/home/koonima/FAST/Data/Pickle/'
    with open(datadir + f'v1p1_de440_pint_bipm2019.pkl', 'rb') as fin:
        psrs = pickle.load(fin)
    psrlist = [psr.name for psr in psrs]
    with open(datadir + f'v1p1_all_dict.json', 'r') as fin:
        noise_dict = json.load(fin)

    # Step 1: Model Construction
    ## Frequency-bins
    Tspan = get_tspan(psrs) # The time-span of the entire PTA
    crn_bins = 14 # number of frequency-bins for the GWB
    int_bins = 30 # number of frequency-bins for the non-GWB (IRN) red noise
    assert int_bins >= crn_bins
    f_intrin = jnp.arange(1/Tspan, (int_bins + 0.01)/Tspan, 1/Tspan) # an array of frequency-bins for the IRN process
    f_common = f_intrin[:crn_bins] # an array of frequency-bins for the common process
    renorm_const = 1 # the factor by which the units are going to change (divided by). Set it to `1` for no unit change (seconds), or let it be `1e9` (nano seconds)
    ## GWB PSD Model 1
    chosen_psd_model1, chosen_orf_model1, gwb_helper_dictionary1 = utils.varied_gamma_hd_pl(renorm_const=renorm_const, lower_amp=-18.0, upper_amp=-11.0)
    gwb_helper_dictionary1
    o1 = models.UniformPrior(gwb_psd_func = chosen_psd_model1,
                    orf_func = chosen_orf_model1,
                    crn_bins = crn_bins,
                    int_bins = int_bins,
                    f_common = f_common, 
                    f_intrin = f_intrin,
                    df = 1/Tspan,
                    Tspan = Tspan, 
                    Npulsars = len(psrs),
                    psr_pos = [psr.pos for psr in psrs],
                    gwb_helper_dictionary = gwb_helper_dictionary1,
                    gamma_min = 0,
                    gamma_max = 7,
                    log10A_min = -20. + 0.5 * jnp.log10(renorm_const), #`0.5 * jnp.log10(renorm_const)` is added to account for change in units,
                    log10A_max = -11. + 0.5 * jnp.log10(renorm_const), #`0.5 * jnp.log10(renorm_const)` is added to account for change in units,
                    renorm_const = renorm_const)
    m1  = LC.MultiPulsarModel(psrs = psrs,
                            device_to_run_likelihood_on = 'cuda',
                            TNr=jnp.array([False]),
                            TNT=jnp.array([False]),
                            run_type_object = o1,
                            noise_dict = noise_dict, 
                            backend = 'backend', 
                            tnequad = False, 
                            inc_ecorr = True, 
                            del_pta_after_init = True,
                            matrix_stabilization = False)
    ## GWB PSD Model 2
    chosen_psd_model2, chosen_orf_model2, gwb_helper_dictionary2 = utils.varied_gamma_hd_pl(renorm_const=renorm_const, lower_amp=-18.0, upper_amp=-11.0)
    gwb_helper_dictionary2
    o2 = models.UniformPrior(gwb_psd_func = chosen_psd_model2,
                    orf_func = chosen_orf_model2,
                    crn_bins = crn_bins,
                    int_bins = int_bins,
                    f_common = f_common, 
                    f_intrin = f_intrin,
                    df = 1/Tspan,
                    Tspan = Tspan, 
                    Npulsars = len(psrs),
                    psr_pos = [psr.pos for psr in psrs],
                    gwb_helper_dictionary = gwb_helper_dictionary2,
                    gamma_min = 0,
                    gamma_max = 7,
                    log10A_min = -20 + 0.5 * jnp.log10(renorm_const), #`0.5 * jnp.log10(renorm_const)` is added to account for change in units,
                    log10A_max = -11 + 0.5 * jnp.log10(renorm_const), #`0.5 * jnp.log10(renorm_const)` is added to account for change in units,
                    renorm_const = renorm_const)
    m2  = LC.CURN(psrs = psrs,
                    device_to_run_likelihood_on = 'cuda',
                    TNr=jnp.array([False]),
                    TNT=jnp.array([False]),
                    run_type_object = o2,
                    noise_dict = noise_dict, 
                    backend = 'backend', 
                    tnequad = False, 
                    inc_ecorr = True, 
                    del_pta_after_init = True,
                    matrix_stabilization = True)
    # Construct the HM
    hm_object = LC.TwoModelHyperModel(model1=m1, model2=m2, 
                                        log_weights = [0., np.log(250)], device='cuda'
                                        )
    x0 = hm_object.make_initial_guess(101)

## Sampling

    savedir = f'./Chain/HM/{idx}/'
    os.makedirs(savedir, exist_ok=True)
    hm_object.sample(x0 = np.array(x0), niter = int(1e6), savedir = savedir, resume=False)

lazy_values = [doit.remote(_) for _ in range(3)]
values = ray.get(lazy_values)

ray.shutdown()