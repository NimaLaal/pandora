# Load Needed Packages
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jar
# jax.config.update('jax_platform_name', 'cuda')
jax.config.update("jax_enable_x64", True)
import numpy as np

from pandora import models, utils, GWBFunctions
from pandora import LikelihoodCalculator as LC


import numpyro
from numpyro import distributions as dist
from numpyro import infer

import pickle, json, os, corner, glob, random, copy, time, inspect
import cloudpickle as cpickle

# Choose a data set
datadir = '/home/koonima/FAST/Data/Pickle/'
with open(datadir + f'v1p1_de440_pint_bipm2019.pkl', 'rb') as fin:
    psrs = pickle.load(fin)
psrlist = [psr.name for psr in psrs]
with open(datadir + f'v1p1_all_dict.json', 'r') as fin:
    noise_dict = json.load(fin)

## Frequency-bins
Tspan = np.load('/home/koonima/pandora/data/15yr_Tspan.npy', mmap_mode = 'r')
crn_bins = 30 # number of frequency-bins for the GWB
int_bins = 30 # number of frequency-bins for the non-GWB (IRN) red noise
assert int_bins >= crn_bins
f_intrin = jnp.arange(1/Tspan, (int_bins + 0.01)/Tspan, 1/Tspan) # an array of frequency-bins for the IRN process
f_common = f_intrin[:crn_bins] # an array of frequency-bins for the common process

# Building the Run in `pandora`
chosen_psd_model, chosen_orf_model, gwb_helper_dictionary = utils.hd_spectrum(renorm_const = 1, 
                                                                            crn_bins = crn_bins, lower_halflog10_rho=-9, upper_halflog10_rho=-4)
### Now, construct the model using `models.UniformPrior`
o = models.UniformPrior(gwb_psd_func = chosen_psd_model,
                orf_func = chosen_orf_model,
                crn_bins = crn_bins,
                int_bins = int_bins,
                f_common = f_common, 
                f_intrin = f_intrin,
                df = 1/Tspan,
                Tspan = Tspan, 
                Npulsars = len(psrs),
                psr_pos = np.load('/home/koonima/pandora/data/15yr_pulsar_positions.npy', mmap_mode = 'r'),
                gwb_helper_dictionary = gwb_helper_dictionary,
                gamma_min = 0,
                gamma_max = 7,
                log10A_min = -20 + 0.5,
                log10A_max = -11 + 0.5,
                renorm_const = 1)
m  = LC.CURN(psrs = None,
            device_to_run_likelihood_on = 'cuda',
            TNr = jnp.load('./TNr.npy', mmap_mode = 'r'),
            TNT = jnp.load('./TNT.npy', mmap_mode = 'r'),
            run_type_object = o,
            noise_dict = None, 
            backend = None, 
            tnequad = False, 
            inc_ecorr = True, 
            del_pta_after_init = True,
            matrix_stabilization = False)
x0 = o.make_initial_guess(key = jar.key(100)) # Some random draw from the prior given an RNG key
ll = jnp.array(m.lower_prior_lim_all)
ul = jnp.array(m.upper_prior_lim_all)

def model():
    xs = numpyro.sample("xs", dist.Uniform(low = ll, 
                                           high = ul))
    numpyro.factor("ll", m.get_lnliklihood(xs))

infer_object = infer.MCMC(
    infer.NUTS(model),
    num_warmup=10,
    num_samples=int(1e1) + 250,
    num_chains=4,
    progress_bar=True,
    chain_method='vectorized')

infer_object.run(jax.random.key(100))
savepath = './Chain/pandora/'
os.makedirs(savepath, exist_ok=True)
try:
    with open(savepath + '/infer.pkl', 'wb') as fout:
        cpickle.dump(infer_object, fout)
except:
    print('Saving the infer object failed! Saving the samples instead.')
    with open(savepath + '/samples.pkl', 'wb') as fout:
        cpickle.dump(infer_object.get_samples(), fout)