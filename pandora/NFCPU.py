import numpy as np
import scipy as sp
import scipy.linalg as sl
from tqdm import tqdm
import torch
from functools import cached_property, partial
import os, random
from itertools import combinations
from enterprise_extensions import model_utils, blocks
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
from enterprise.signals import signal_base, gp_signals
from enterprise_extensions import sampler as samp
import torch.utils.dlpack as torchdlpack


class NF_distribution_conditional:
    """
    To use a normalizing flow pyro object as a distribution in numpyro

    pending...
    """
    def __init__(
        self,
        pyro_nf_object,
        mean,
        half_range,
        scale,
        gwb_freq_idxs,
        ast_param_idxs,
        lower_bound_gwb,
        upper_bound_gwb):

        self.nf = pyro_nf_object
        self.B = scale
        self.mean_gwb = mean[gwb_freq_idxs][None]
        self.mean_ast = mean[ast_param_idxs][None]
        self.half_range_gwb = half_range[gwb_freq_idxs][None]
        self.half_range_ast = half_range[ast_param_idxs][None]
        self.low_gwb = lower_bound_gwb
        self.high_gwb = upper_bound_gwb

    def convert_torch_to_numpy(self, torch_tensor):
        return torch_tensor.detach().numpy()

    def convert_numpy_to_torch(self, numpy_array):
        return torchdlpack.from_dlpack(numpy_array.__dlpack__()).float()
        
    def transform_rho_to_scaled_interval(self, gwb_rho):
        return self.B * (gwb_rho - self.mean_gwb)/self.half_range_gwb
    
    def transform_params_to_scaled_interval(self, ast_params):
        return self.B * (ast_params - self.mean_ast)/self.half_range_ast

    def transform_rho_to_physical_interval(self, gwb_rho):
        return gwb_rho * self.half_range_gwb/self.B + self.mean_gwb
    
    def transform_params_to_physical_interval(self, ast_params):
        return ast_params * self.half_range_ast/self.B + self.mean_ast

    def log_prob(self, gwb_rho, astro_params):
        scaled_ast = self.convert_numpy_to_torch(self.transform_params_to_scaled_interval(astro_params))
        scaled_gwb = self.convert_numpy_to_torch(self.transform_rho_to_scaled_interval(gwb_rho))
        return self.convert_torch_to_numpy(self.nf.condition(scaled_ast).log_prob(scaled_gwb))

    def sample(self, batch_shape, astro_params):
        scaled_ast = self.convert_numpy_to_torch(self.transform_params_to_scaled_interval(astro_params))
        scaled_gwb = self.convert_torch_to_numpy(self.nf.condition(scaled_ast).sample(batch_shape))
        return self.transform_rho_to_physical_interval(scaled_gwb)

    def avg_sample_and_prob_of_avg(self, batch_shape, astro_params):
        scaled_ast = self.convert_numpy_to_torch(self.transform_params_to_scaled_interval(astro_params))
        scaled_gwb_avg = torch.mean(self.nf.condition(scaled_ast).sample(batch_shape), dim = 0)[None]
        log_prob = self.nf.condition(scaled_ast).log_prob(scaled_gwb_avg)
        return self.transform_rho_to_physical_interval(self.convert_torch_to_numpy(scaled_gwb_avg))[0], self.convert_torch_to_numpy(log_prob)[0]

class CPUInferWithNF(object):
    '''
    A class to perform astrophysical inference using normalizing-flows, JAX, and numpyro.

    param: `nf`: the normalizing flow object you want to use. You may need to change the source-code if your nf is not built by pyro!
    param: `psrs`: a list of enterprise pulsar objects with lenght equal to Npulsars
    param: `crn_bins`: number of frequency-bins for the GWB
    param: `int_bins`: number of frequency-bins for the non-GWB (IRN) red noise
    param: `gwb_freq_idxs`: gwb indicies used in normalizing flows training
    param: `ast_freq_idxs`: astro parameter indicies used in normalizing flows training
    param: `Tspan`: Observational baseline of the PTA (optional)
    param: `noise_dict`: WN and IRN noise dictionary (optional if `pta` is provided)
    param: `backend`: telescope backend (optional if `pta` is provided)
    param: `tnequad`: do you want to use the temponest equad/efac convention? (optional if `pta` is provided)
    param: `inc_ecorr`: do you want to add ECORR? (optional if `pta` is provided)
    param: `int_rn`: do you want to search for non-GWB red noise as well? (optional if `pta` is provided)
    param: `uniform_priors`: uniform priors for the astroparameters aranged in the form of a JAX array (n_pars, 2). [:, 0] is lower-bound.
    param: `gwb_gamma`: GWB gamma (optional if `pta` is provided)
    param: `pta`: do you want to supply your own enterprise pta object?
    param: `curn`: do you want to model the correlations as HD? If yes, choose `hd`. Else, choose `curn`.(optional if `pta` is provided)
    param: `matrix_stabilization`: do you want to stabelize the important matrices (optional but recommended)?
    param: `no_likelihood`: only uses the `nf` object to perform the analysis. 
    param: `fixed_gwb_psd`: when `no_likelihood` is set to True, you need to supply the gwb_psd you want to the astro-parameters to be inferred from.  
    param: `renorm_const`: the factor by which the units are going to change? Set it to `1` for no unit change, or let it be `1e9` for better performance.

    Author:
    Nima Laal (01/25/2025)
    '''
    def __init__(self,
                nf, 
                 psrs,
                crn_bins,
                int_bins,
                gwb_freq_idxs,
                ast_param_idxs,
                Tspan = None, 
                noise_dict = None, 
                backend = 'none', 
                tnequad = False, 
                inc_ecorr = False, 
                int_rn = True,
                uniform_priors = None,
                gwb_gamma = None,
                pta = None,
                curn = False,
                matrix_stabilization = True,
                no_likelihood = False,
                fixed_gwb_psd = None,
                renorm_const = 1e9):

        self.no_likelihood = no_likelihood
        if self.no_likelihood:
            if not np.any(fixed_gwb_psd):
                raise AssertionError('A no-likelihood run requires a fixed gwb_spectrum (0.5log10rho)')
            else:

                self.fixed_gwb_psd = fixed_gwb_psd

        if not np.any(uniform_priors):
            raise AssertionError('Give an array for your priors with the shape (n_par, 2). [:, 0] is lower-bound.')
        if uniform_priors.shape[-1] != 2 or uniform_priors.ndim != 2:
            raise AssertionError('Only an array with the shape (n_par, 2) is accepted. For non-uniform priors, you need to wait...')
        else:
            self.upper_prior_lim_astro = uniform_priors[:, 1]
            self.lower_prior_lim_astro = uniform_priors[:, 0]

        if Tspan:
            self.Tspan = Tspan
        else:
            self.Tspan = model_utils.get_tspan(psrs)
        self.fref = 1/(1 * 365.25 * 24 * 60 * 60)
        self.psr = psrs
        self.nf_object, self.half_range, self.B, self.mean = nf
        self.noise_dict = noise_dict
        self.Npulsars = len(psrs)
        self.renorm_const = renorm_const ## re-normalization constant for some matricies to avoid nearing machine precision!
        self.half_log_psd_offset = np.log10(renorm_const)/2
        self.crn_bins = crn_bins
        self.int_bins = int_bins
        self.number_astro_pars = self.upper_prior_lim_astro.shape[0]
        assert self.crn_bins <= self.int_bins
        self.f_intrin = np.arange(1/self.Tspan, (self.int_bins + 0.01)/self.Tspan, 1/self.Tspan)
        self.f_common = self.f_intrin[:crn_bins]
        self.kmax = 2 * self.crn_bins
        self.diag_idx = np.arange(0, self.Npulsars)
        self.diag_idx_large = np.arange(0, self.Npulsars * self.kmax)
        self.k_idx = np.arange(0, self.kmax)
        self.c_idx = np.arange(0, self.crn_bins)
        self.int_rn = int_rn
        self._eye = torch.tensor(np.repeat(np.eye(self.Npulsars)[None], self.crn_bins, axis = 0), device = 'cpu', dtype = torch.float64)
        self.log10A_min = -18 + 0.5 * np.log10(renorm_const)
        self.log10A_max = -11 + 0.5 * np.log10(renorm_const)
        self.gamma_min = 0
        self.gamma_max = 7

        self.upper_prior_lim_all = np.zeros(2 * self.Npulsars)
        self.upper_prior_lim_all[0::2] = self.gamma_max
        self.upper_prior_lim_all[1::2] = self.log10A_max
        self.upper_prior_lim_all = np.append(self.upper_prior_lim_all, self.upper_prior_lim_astro)

        self.lower_prior_lim_all = np.zeros(2 * self.Npulsars)
        self.lower_prior_lim_all[0::2] = self.gamma_min
        self.lower_prior_lim_all[1::2] = self.log10A_min
        self.lower_prior_lim_all = np.append(self.lower_prior_lim_all, self.lower_prior_lim_astro)

        self.gwb_freq_idxs = gwb_freq_idxs
        self.ast_param_idxs = ast_param_idxs
        self.ppair_idx = np.arange(0, int(self.Npulsars * (self.Npulsars - 1) * 0.5))

        self.nf_dist = NF_distribution_conditional(pyro_nf_object = self.nf_object,
                                                    mean = self.mean,
                                                    half_range = self.half_range,
                                                    scale = self.B,
                                                    gwb_freq_idxs = self.gwb_freq_idxs,
                                                    ast_param_idxs = self.ast_param_idxs,
                                                    lower_bound_gwb = -14.0,
                                                    upper_bound_gwb = -3.0)

        if not pta:
            if curn:
                self.orf = 'crn'
            else:
                self.orf = 'hd'
            tm = gp_signals.MarginalizingTimingModel(use_svd=True)
            wn = blocks.white_noise_block(vary=False, inc_ecorr=inc_ecorr, gp_ecorr=False, select = backend,tnequad = tnequad)
            rn = blocks.red_noise_block(psd='powerlaw', prior='log-uniform',Tspan = self.Tspan,
                                                    components=self.crn_bins, gamma_val=None)
            gwb = blocks.common_red_noise_block(psd='powerlaw', prior='log-uniform', Tspan = self.Tspan,
                                                    components=self.crn_bins, gamma_val=gwb_gamma, name = 'gw', orf = self.orf)           
            if int_rn:
                s = tm + wn + rn + gwb
            else:
                s = tm + wn + gwb

            self.pta = signal_base.PTA([s(p) for p in self.psr])
            self.pta.set_default_params(self.noise_dict)
        else:
            raise Warning("The code only deals with powerlaw modeling for now.\nLet the author know if you will need another type of modeling (or make your own `get_phi_mat` function in the class!)")
        if not self.orf == 'crn':
            self._TNT = sp.linalg.block_diag(*self.pta.get_TNT(params = {}))/self.renorm_const
            self._TNr = np.concatenate(self.pta.get_TNr(params = {}))/np.sqrt(self.renorm_const)
        else:
            self._TNT = np.array(self.pta.get_TNT(params = {}))/self.renorm_const
            self._TNr = np.array(self.pta.get_TNr(params = {}))/np.sqrt(self.renorm_const)

        if not self.orf == 'crn' and matrix_stabilization:
            print(f'Condition number of the TNT matrix before stabilizing is: {np.format_float_scientific(np.linalg.cond(self._TNT))}')
            D = np.outer(np.sqrt(self._TNT.diagonal()), np.sqrt(self._TNT.diagonal()))
            corr = self._TNT/D
            corr = (corr + 1e-3 * np.eye(self._TNT.shape[0]))/(1 + 1e-3)
            self._TNT = D * corr
            print(f'Condition number of the TNT matrix after stabilizing is: {np.format_float_scientific(np.linalg.cond(self._TNT))}')

    def HD_ORF(self, angle):
        return 3/2*( (1/3 + ((1-np.cos(angle))/2) * (np.log((1-np.cos(angle))/2) - 1/6)))

    def to_ent_phiinv(self, phiinv):
        '''
        Changes the format of the phiinv matrix from (n_freq, n_pulsar, n_pulsar) to (n_freq * n_pulsar by n_freq * n_pulsar).

        param: `phiinv`: the phiinv matrix.
        '''
        phiinv_ent = np.zeros((self.Npulsars, self.kmax, self.Npulsars, self.kmax))
        phiinv_ent[:, self.k_idx, :, self.k_idx] = phiinv
        return phiinv_ent.reshape((self.Npulsars * self.kmax, self.Npulsars * self.kmax))

    @cached_property
    def get_cross_info(self):
        
        I, J = np.tril_indices(self.Npulsars)
        a = np.zeros(len(self.ppair_idx), dtype = int)
        b = np.zeros(len(self.ppair_idx), dtype = int)
        ct = 0
        for i, j in zip(I, J):
            if not i == j:
                a[ct] = i
                b[ct] = j
                ct+=1
        self.xi = np.array([np.arccos(np.dot(self.psr[I].pos, self.psr[J].pos)) for I, J in zip(a, b)])
        self.I = a
        self.J = b

    @cached_property
    def get_hd_val(self):
        return self.HD_ORF(self.xi)

    def common_pl_to_rho(self, log10amp, gamma):
        return 10**(2*log10amp)/(12 * np.pi**2 * self.f_common[:, None]**3 * self.Tspan) * (self.f_common[:, None]/self.fref)**(3-gamma)
    
    def intrin_pl_to_rho(self, log10amp, gamma):
        return 10**(2*log10amp)/(12 * np.pi**2 * self.f_intrin[:, None]**3 * self.Tspan) * (self.f_intrin[:, None]/self.fref)**(3-gamma)

    def get_phi_mat(self, log10amp, gamma, half_log10_rho):

        psd_common = 10**(2*half_log10_rho)[:, None]

        if self.int_rn:
            phi = np.zeros((self.int_bins, self.Npulsars, self.Npulsars))
            psd_intrin = self.intrin_pl_to_rho(log10amp, gamma)
            phi[:self.crn_bins, self.diag_idx, self.diag_idx] = psd_intrin[:self.crn_bins] + psd_common
            if self.crn_bins != self.int_bins:
                phi[self.crn_bins:, self.diag_idx, self.diag_idx] = psd_intrin[self.crn_bins:]
        else:
            phi = np.zeros((self.crn_bins, self.Npulsars, self.Npulsars))
            phi[:, self.diag_idx, self.diag_idx] = psd_common

        if self.orf == 'hd':
            phi[:self.crn_bins, self.I, self.J] = self.get_hd_val * psd_common
            return phi
        else:
            return phi[:, self.diag_idx, self.diag_idx]

    def get_mean(self, phiinv):
        '''
        Estimates the mean and the variance of the `a` distribution given an estimate of the `phiinv` matrix.

        param: `phiinv`: the phiinv matrix. The dimensions are (2*n_freq, n_pulsar, n_pulsar).
        params: `pivot`: whether to use the pivoted cholesky decomposition for the dense Sigma matrix. The saparse CHOLMOD
        package does this automatically.
        '''        
        cf = sp.linalg.cho_factor(self._TNT + self.to_ent_phiinv(phiinv), lower=False)
        return sp.linalg.cho_solve(cf, self._TNr), 2 * np.log(cf[0].diagonal()).sum()

    def solve_tri_in_batches(self, mats, b, upper):
        return np.from_dlpack(torch.linalg.solve_triangular(torch.from_dlpack(mats), b, 
                            upper = upper, left=True, unitriangular=False, out=None))

    def get_lnprior(self, xs):
        '''
        A function to return natural log prior (uniform)

        param: `xs`: powerlaw model parameters
        '''
        if self.no_likelihood:
            if np.any(xs < self.lower_prior_lim_astro) or np.any(xs > self.upper_prior_lim_astro):
                return -np.inf
            else:
                return -8.01            
        else:
            if np.any(xs < self.lower_prior_lim_all) or np.any(xs > self.upper_prior_lim_all):
                return -np.inf
            else:
                return -8.01
          
    def get_lnliklihood(self, xs):
        try:
            gamma = xs[0:-self.number_astro_pars:2]
            log10amp = xs[1:-self.number_astro_pars:2]
            half_log10_rho, ln_cond_prior = self.rhos_given_astro(xs[-self.number_astro_pars:])
            if self.no_likelihood:
                return ln_cond_prior
            else:
                phi = self.get_phi_mat(log10amp, gamma, half_log10_rho + self.half_log_psd_offset)
                cp = np.linalg.cholesky(phi)
                logdet_phi = 4 * np.sum(np.log(cp.diagonal(axis1 = 1, axis2 = 2)))
                phiinv_dense = np.repeat(self.solve_tri_in_batches(mats = phi, b = self._eye, upper = False), 2, axis = 0)
                expval, logdet_sigma  = self.get_mean(phiinv_dense)
                loglike = 0.5 * (np.dot(self._TNr, expval) - logdet_sigma - logdet_phi)
                return loglike + ln_cond_prior
        except np.linalg.LinAlgError:
            return -np.inf

    def get_lnliklihood_numpyro(self, gamma, log10amp, half_log10_rho):
        phi = self.get_phi_mat(log10amp, gamma, half_log10_rho + self.half_log_psd_offset)
        cp = np.linalg.cholesky(phi)
        logdet_phi = 4 * np.sum(np.log(cp.diagonal(axis1 = 1, axis2 = 2)))
        phiinv_dense = np.repeat(self.solve_tri_in_batches(mats = phi, b = self._eye, upper = False), 2, axis = 0)
        expval, logdet_sigma  = self.get_mean(phiinv_dense)
        loglike = 0.5 * (np.dot(self._TNr, expval) - logdet_sigma - logdet_phi)
        return loglike

    # def rhos_given_astro(self, astro_params):
    #     cond_dist = NF_distribution_conditional(astro_params = astro_params[None],
    #                                                                             pyro_nf_object = self.nf_object,
    #                                                                             mean = self.mean,
    #                                                                             half_range = self.half_range,
    #                                                                             scale = self.B,
    #                                                                             gwb_freq_idxs = self.gwb_freq_idxs,
    #                                                                             ast_param_idxs = self.ast_param_idxs,
    #                                                                             lower_bound_gwb = -14.0,
    #                                                                             upper_bound_gwb = -3.0)
    #     rho = cond_dist.sample((int(2e3), ))
    #     log_prob = cond_dist.log_prob(rho)
    #     return np.mean(rho, axis = 0), np.mean(log_prob, axis = 0)
    #     # return astro_params[None], [0]

    def rhos_given_astro(self, astro_params):
        return astro_params, 0
        # if self.no_likelihood:
        #     return self.fixed_gwb_psd, self.nf_dist.log_prob(self.fixed_gwb_psd, astro_params)
        # else:
        #     return self.nf_dist.avg_sample_and_prob_of_avg(astro_params = astro_params[None], batch_shape=(int(1e3), ))

        
    def make_initial_guess(self, seed = None):
        if seed:
            np.random.seed(seed)
        if self.no_likelihood:
            return np.random.uniform(self.lower_prior_lim_astro, self.upper_prior_lim_astro)
        else:
            x0 = np.zeros(2 * self.Npulsars)
            ast_x0 = np.random.uniform(self.lower_prior_lim_astro, self.upper_prior_lim_astro)
            gamma_x0 = np.random.uniform(self.gamma_min, self.gamma_max, size = self.Npulsars)
            log10amp_x0 = np.random.uniform(self.log10A_min, self.log10A_max, size = self.Npulsars)
            x0[0::2] = gamma_x0
            x0[1::2] = log10amp_x0
            return np.append(x0, ast_x0)

    def sample(self, niter, savedir, resume = True, seed = None):
        '''
        A function to perform the sampling

        param: `niter`: the number of sampling iterations
        param: `savedir`: the directory to save the chains
        param: `CURNiters`: the number of MCMC steps per analytic Fourier coefficient draws
        param: `a0`: if given, the starting guess is for Fourier coefficient rather than `PSD` params. 
        '''
        x0 = self.make_initial_guess(seed)
        ndim = len(x0)
        cov = np.diag(np.ones(ndim) * 0.01**2)
        groups = [list(np.arange(0, ndim))]
        if not self.no_likelihood:
            important_idxs = np.array(range(2*self.Npulsars, 2*self.Npulsars + len(self.upper_prior_lim_astro)))
            print(important_idxs)
            [groups.append(important_idxs) for ii in range(2)]

        sampler = ptmcmc(ndim, self.get_lnliklihood, self.get_lnprior, cov, groups = groups,
                        outDir=savedir, 
                        resume=resume)
                        
        if not self.no_likelihood:
            sampler.addProposalToCycle(self.draw_from_prior, 15)

        sampler.sample(x0, niter, SCAMweight=30, AMweight=15, DEweight=50, )

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