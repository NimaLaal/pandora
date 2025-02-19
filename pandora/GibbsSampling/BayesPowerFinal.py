import numpy as np
from tqdm import tqdm
import scipy.linalg as sl
from functools import cached_property
import os, time, glob, warnings, random, torch
from enterprise_extensions import model_utils, blocks
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
from enterprise.signals import signal_base, gp_signals
from la_forge.core import Core
from sksparse.cholmod import cholesky, CholmodError
import scipy.sparse as sps
from scipy.linalg import solve_triangular as st_solve
from itertools import combinations
from scipy.linalg import cho_factor, cho_solve
from enterprise_extensions import sampler as samp
from ceffyl import densities, Ceffyl, models
from scipy.linalg import ldl
from enterprise.signals.parameter import Uniform

class FailException(Exception):
    pass

#####################################Some Custom ORFs###################################
# bins = np.array([0.03759145, 0.41650786, 0.56739343, 0.70452728, 0.80722737,
#        0.93432403, 1.0396636 , 1.14780095, 1.26771677, 1.37786293,
#        1.4783429 , 1.60906508, 1.75493235, 1.98867019, 2.21283648,
#        2.90677661])
# bins[-1] = np.pi
bins = np.array([1e-3, 30.0, 50.0, 80.0, 100.0,
                         120.0, 150.0, 180.0]) * np.pi/180.0
def HD_ORF(angle):
    return 3/2*( (1/3 + ((1-np.cos(angle))/2) * (np.log((1-np.cos(angle))/2) - 1/6)))

def bin_orf(angle, params):
    '''
    Agnostic binned spatial correlation function. Bin edges are
    placed at edges and across angular separation space. Changing bin
    edges will require manual intervention to create new function.

    :param: params
        inter-pulsar correlation bin amplitudes.

    Author: S. R. Taylor (2020)

    '''
    idx = np.digitize(angle, bins)
    return params[idx-1]

def gt_orf(angle, tau):
    """
    General Transverse (GT) Correlations. This ORF is used to detect the relative
    significance of all possible correlation patterns induced by the most general
    family of transverse gravitational waves.

    :param: tau
        tau = 1 results in ST correlations while tau = -1 results in HD correlations.

    Author: N. Laal (2020)

    """
    k = 1/2*(1-np.cos(angle))
    return 1/8 * (3+np.cos(angle)) + (1-tau)*3/4*k*np.log(k)
    # return np.ones(len(angle))
##################################################################################

 
class BayesPowerMulti(object):
    '''
    A class to perform a multi-pulsar Gibbs sampling

    param: `psrs`: a list of enterprise pulsar objects (Npulsars)
    param: `crn_bins`: the number of frequency bins
    param: `df`: the degrees of freedom of the Inverse-Wishart distribution 
    param: `Tspan`: the baseline of the PTA in secodns
    param: `noise_dict`: noise-dictionary containing the white noise params.
    param: `gamma_spectrum`: the spectral index of the common red noise used for chi-squared fitting
    param: `backend`: the backend to use
    param: `inc_ecorr`: whether to include ecorr
    param: 'int_rn': whether to include non-gwb red noise
    param: `half_logphi_ii_lower`:the lowest value of 0.5log_10 of the diagonals of the phi-matrix in units of seconds
    param: `half_logphi_ii_upper`:the highest value of 0.5log_10 of the diagonals of the phi-matrix in units of seconds
    param: `renorm_const`: the constant to change the units of the matricies from seconds to something else (e.g., nano seconds)
    param: `empirical_nd_orf`: the orf used in the empirical noise distribution run
    param: `fail_trial_count`: the number of times you allow linalg operations to fail without force-kicking the Gibbs sampler

    Author:
    Nima Laal (04/11/2024)
    '''
    def __init__(self, 
                 psrs,
                crn_bins,
                df = None, 
                Tspan = None, 
                noise_dict = None, 
                backend = 'none', 
                tnequad = False, 
                inc_ecorr = False, 
                int_rn = True,
                gamma_spectrum = 13/3,
                half_logphi_ii_lower = -9,
                half_logphi_ii_upper = -2,
                renorm_const = 1e9,
                pta = None,
                fail_trial_count = 1000):
        
        self.psr = psrs
        self.Npulsars = len(self.psr)
        if not df:
            self.df = self.Npulsars + 1  ##degrees of freedom of the inverse-wishart distribution
        else:
            self.df = df
        self.renorm_const = renorm_const ## re-normalization constant for some matricies to avoid nearing machine precision!
        self.noise_dict = noise_dict
        self.gamma = gamma_spectrum

        self.crn_bins = crn_bins
        self.kmax = 2 * self.crn_bins
        if Tspan:
            self.Tspan = Tspan
        else:
            self.Tspan = model_utils.get_tspan(self.psr)

        self.fail_trial_count = fail_trial_count

        self.diag_idx = np.arange(0, self.Npulsars, 1, int)
        self.k_idx = np.arange(0, self.kmax, 1, int)
        self.c_idx = np.arange(0, self.crn_bins, 1, int)
        self.ppair_idx = np.arange(0, int(self.Npulsars * (self.Npulsars - 1) * 0.5), 1, int)
        self.diag_offset = np.log10(renorm_const)/2
        self.lower_auto = half_logphi_ii_lower + self.diag_offset
        self.upper_auto = half_logphi_ii_upper + self.diag_offset
        self.eye = np.eye(self.Npulsars * self.kmax)

        ###Make a CCRN PTA Object (needed to construct matricies from ENTERPRISE)
        ###The exact model used for the red noise does not matter. Below works just fine. 
        if not pta:
            tm = gp_signals.MarginalizingTimingModel(use_svd=True)
            wn = blocks.white_noise_block(vary=False, inc_ecorr=inc_ecorr, gp_ecorr=False, select = backend,tnequad = tnequad)
            rn = blocks.red_noise_block(psd='powerlaw', prior='log-uniform',Tspan = self.Tspan,
                                                    components=self.crn_bins, gamma_val=None)
            gwb = blocks.common_red_noise_block(psd='powerlaw', prior='log-uniform', Tspan = self.Tspan,
                                                    components=self.crn_bins, gamma_val=None, name = 'gw', orf = 'gt')            
            if int_rn:
                s = tm + wn + rn + gwb
            else:
                s = tm + wn + gwb
            self.pta = signal_base.PTA([s(p) for p in self.psr])
            self.pta.set_default_params(self.noise_dict)
            self.use_ent_phi = False
        else:
            self.pta = pta
            self.use_ent_phi = True

        self.x0 = np.hstack([p.sample() for p in self.pta.params])
        self._TNr = np.concatenate(self.pta.get_TNr(params = {}))/np.sqrt(self.renorm_const)
        self._TNT = sl.block_diag(*self.pta.get_TNT(params = {}))/self.renorm_const
        ##############Make TNT More Stable:
        print(f'Condition number of the TNT matrix before stabilizing is: {np.format_float_scientific(np.linalg.cond(self._TNT))}')
        D = np.outer(np.sqrt(self._TNT.diagonal()), np.sqrt(self._TNT.diagonal()))
        corr = self._TNT/D
        corr = (corr + 1e-3 * np.eye(self._TNT.shape[0]))/(1 + 1e-3)
        self._TNT = D * corr
        print(f'Condition number of the TNT matrix after stabilizing is: {np.format_float_scientific(np.linalg.cond(self._TNT))}')
    ################Utility Functions################

    def pl_to_rho(self, log10amp, gamma):
        '''
        Function to convert a powerlaw model to a free-spectral model

        param:`log10amp`: the log10 of the amplitude of the red noise
        param: `gamma`: the spectral index of the red noise 
        '''
        return 10**(2*log10amp)/(12 * np.pi**2 * self.freqs[:, None]**3 * self.Tspan) * (self.freqs[:, None]/self.fref)**(3-gamma)

    def get_phi_mat(self, xs):
        '''
        Function to construct a dense phi-matrix (nf, np, np) given a CCRN model.

        param: `xs`: the list of the CCRN model parameters given in the correct order (gamma1, log10A1, ..., gamma_GWB, log10A_GW, bin_orf params)
        '''
        end_idx = 2 * self.Npulsars
        end_orf_idx = end_idx + 2

        phi = np.zeros((self.crn_bins, self.Npulsars, self.Npulsars))
        psd_intrin = self.pl_to_rho(xs[1:end_idx:2], xs[0:end_idx:2])
        psd_common = self.pl_to_rho(xs[end_idx + 1], xs[end_idx])
        phi[:, self.diag_idx, self.diag_idx] = psd_intrin + psd_common
        # phi[:, self.I_for_phi, self.J_for_phi] = bin_orf(self.xi_for_phi, xs[end_orf_idx:]) * psd_common
        # phi[:, self.I_for_phi, self.J_for_phi] = HD_ORF(self.xi_for_phi) * psd_common
        phi[:, self.I_for_phi, self.J_for_phi] = gt_orf(self.xi_for_phi, xs[end_orf_idx:]) * psd_common
        phi[:, self.J_for_phi, self.I_for_phi] = phi[:, self.I_for_phi, self.J_for_phi]
        return phi
    
    def rho_give_a(self, a):
        """
        Function to perform rho updates given the Fourier coefficients.

        param: `a`: the Fourier coefficients given in the form (2nf, np, np)
        """
        low = 10**(2 * self.lower_auto)
        high = 10**(2 * self.upper_auto)
        tau = a ** 2
        tau = (tau[0::2] + tau[1::2]) / 2

        Norm = 1 / (np.exp(-tau / high) - np.exp(-tau / low))
        x = np.random.default_rng().uniform(0, 1, size=tau.shape)
        return -tau / np.log(x / Norm + np.exp(-tau / low))

    def get_ln_pos(self, dens_phi, expval, cf):
        '''
        Function to calculate the marginalized posterior distribution
        '''
        LP = self.regularize(dens_phi, return_fac=True)

        logdet_phi = 2 * np.sum(np.log(LP[:, self.diag_idx, self.diag_idx]))

        logdet_sigma = 2 * np.sum(np.log(cf.diagonal()))

        loglike = 0.5 * (np.dot(self._TNr, expval) - logdet_sigma - 2 * logdet_phi)

        logprior = -self.Npulsars * logdet_phi
            
        return loglike + logprior
    
    def kick(self, phi, phi_new, mean, mean_new, chol_var, chol_var_new):
        '''
        Metrapolis Hastings algorithm
        '''
        lnhastings = self.get_ln_pos(phi_new, mean_new, chol_var_new) - self.get_ln_pos(phi, mean, chol_var)
        if np.log(random.random()) < lnhastings:
            return True, lnhastings
        else:
            return False, lnhastings
                       

    def chi_sq_fit(self, phi_ij):
        '''
        chi-squared fit to the cross-corrlation values with fixed spectrum
        ''' 
        return 0.5 * np.log10(np.einsum('qkn,k,n->q',phi_ij, self.spectrum, self.orf_val, optimize = 'greedy')/self.bot)
    
    def to_ent_phiinv(self, phiinv):
        '''
        Changes the format of the phiinv matrix from (2n_freq, n_pulsar, n_pulsar) to (2n_freq * n_pulsar by 2n_freq * n_pulsar).

        param: `phiinv`: the phiinv matrix.
        '''
        phiinv_ent = np.zeros((self.Npulsars * self.kmax, self.Npulsars * self.kmax))
        for fidx in self.k_idx:
            phiinv_ent[fidx::self.kmax,fidx::self.kmax] = phiinv[fidx]
        # phiinv_ent = np.zeros((self.Npulsars, self.kmax, self.Npulsars, self.kmax))
        # phiinv_ent[:, self.k_idx, :, self.k_idx] = phiinv
        # return phiinv_ent.reshape((self.Npulsars * self.kmax, self.Npulsars * self.kmax))
        return phiinv_ent
        
    @cached_property
    def wishart_helper(self):
        '''
        This function caches the indicies needed to sample from a standard-wishart distribution
        '''
        self.I, self.J = np.tril_indices(self.Npulsars)
        i_cross = []
        j_cross = []
        for i, j in zip(self.I, self.J):
            if not i == j:
                i_cross.append(i)
                j_cross.append(j)
        self.i_cross = np.array(i_cross)
        self.j_cross = np.array(j_cross)
        self.dfs = np.array([self.Npulsars + 1 - i for i in range(self.Npulsars)])

        self.freqs = np.arange(1/self.Tspan, (self.crn_bins + .01)/self.Tspan, 1/self.Tspan)
        self.fref = 1/(60 * 60 * 24 * 365.25)
        self.spectrum = 1/(12 * np.pi**2 * self.freqs**3 * self.Tspan) * (self.freqs/self.fref)**(3-self.gamma)
        self.xi = np.array([np.arccos(np.dot(self.psr[I].pos, self.psr[J].pos)) for I, J in zip(self.i_cross, self.j_cross)])
        # self.orf_val = HD_ORF(self.xi)
        self.orf_val = gt_orf(self.xi, tau = 0.5)
        self.bot = np.einsum('k,n->', self.spectrum**2, self.orf_val**2, optimize = 'greedy') * self.renorm_const

        I, J = np.array(list(combinations(range(self.Npulsars),2))).T
        npairs = int(self.Npulsars * (self.Npulsars - 1) * 0.5)
        a = np.zeros(npairs, dtype = int)
        b = np.zeros(npairs, dtype = int)
        ct = 0
        for i, j in zip(I, J):
            if not i == j:
                a[ct] = i
                b[ct] = j
                ct+=1
        self.I_for_phi = a
        self.J_for_phi = b
        self.xi_for_phi = np.array([np.arccos(np.dot(self.psr[I].pos, self.psr[J].pos)) for I, J in zip(a, b)])

    def standard_wishart(self, all_freqs_diff = False):
        '''
        Samples from the cholesky decomposition of the standard-wishart distribution.
        `wishart_helper` function needs to have been called once prior to calling this
        function.

        param: `all_freqs_diff = True`: whether to use different set of random numbers
        for each frequency bin
        '''
        if all_freqs_diff:
            A = np.zeros((self.crn_bins, self.Npulsars, self.Npulsars))
            A[:, self.i_cross, self.j_cross] = np.random.normal(size = (self.crn_bins, len(self.I) - self.Npulsars))
            A[:, self.diag_idx, self.diag_idx] = np.sqrt(np.random.chisquare(self.dfs, size = (self.crn_bins, self.Npulsars)))
        else:
            A = np.zeros((self.Npulsars, self.Npulsars))
            A[self.i_cross, self.j_cross] = np.random.normal(size = (len(self.I) - self.Npulsars))
            A[self.diag_idx, self.diag_idx] = np.sqrt(np.random.chisquare(self.dfs, size = (self.Npulsars)))
        return A

    def regularize(self, herm_mat, return_fac):
        '''
        Regularizes a rank-deficient real symmetric matrix into a full-rank matrix
        param: `herm_mat`: the matrix to perform regularization on. The shape must be
                            (n_freq by n_pulsar by n_pulsar)
        param: `return_fac`: wheher to return the square-root factorization
        '''
        delta_est = np.random.uniform(1e-8, 1e-4, size = (self.crn_bins, self.Npulsars))
        sqr_diags = np.sqrt(herm_mat.diagonal(axis1 = 1, axis2 = 2))[..., None]
        D = sqr_diags @ sqr_diags.transpose(0, 2, 1)
        corr = herm_mat/D
        corr[:, self.diag_idx, self.diag_idx]+=delta_est
        # corr = corr/(1+delta_est[..., None])
        if return_fac:
            return sqr_diags * np.linalg.cholesky(corr)
        else:
            return corr * D
            
    def get_scale_mat(self, a):
        scale = np.transpose(a, (0, 2, 1)) @ a
        scale = scale[0::2] + scale[1::2]
        return self.regularize(scale, return_fac=True)
        
    ################Conditional Distribution Functions################
    
    def get_scale_mat_avg(self, a):
        scale = np.mean(np.transpose(a, (0, 1, 3, 2)) @ a, axis = 0)
        scale = scale[0::2] + scale[1::2]
        return np.linalg.cholesky(scale)  
    
    def get_mean(self, phiinv):
        '''
        Estimates the mean and the variance of the `a` distribution given an estimate of the `phiinv` matrix.

        param: `phiinv`: the phiinv matrix. The dimensions are (2*n_freq, n_pulsar, n_pulsar).
        params: `pivot`: whether to use the pivoted cholesky decomposition for the dense Sigma matrix. The saparse CHOLMOD
        package does this automatically.
        '''        
        Sigma = self._TNT + self.to_ent_phiinv(phiinv)
        cf = sl.cho_factor(Sigma, lower=False)
        return sl.cho_solve(cf, self._TNr), cf[0]
                
    def a_given_phiinv(self, mean, chol_var):
        '''
        Performs the `a given phiinv` step of the Gibbs sampling.

        param: `phiinv`: the phiinv matrix. The dimensions are (2*n_freq, n_pulsar, n_pulsar).
        param: `return_mean`: whether to only return the mean of the `a` distribution
        '''
        rand_vec = np.random.normal(loc=0, scale=1, size=(self.kmax * self.Npulsars))
        a = mean + st_solve(chol_var, rand_vec, trans=0, lower=False, unit_diagonal=False, overwrite_b=False, check_finite=False)
        a = a.reshape(self.Npulsars, self.kmax).T
        return a[:, None, :]

    def empiricalphi_to_iwphi_coeff(self, x0):
        '''
        Converts an empirical estimate of PSD to an IW phi matrix.

        param: `x0`: the parameterized PSD estimates 
        '''
        if self.use_ent_phi:
            phiinv = np.array(self.pta.get_phiinv(self.pta.map_params(x0), logdet=False))
        else:
            phiinv = np.linalg.inv(self.get_phi_mat(x0))
        emp_phiinv = np.repeat(phiinv, 2, axis = 0)/self.renorm_const
        mean, chol_var = self.get_mean(emp_phiinv)
        return self.a_given_phiinv(mean, chol_var)
    
    def empiricalphi_to_iwphi(self, x0, chi_fit = True):
        '''
        Converts an empirical estimate of PSD to an IW phi matrix.

        param: `x0`: the parameterized PSD estimates 
        '''
        for _ in range(self.fail_trial_count):

            if self.use_ent_phi:
                phiinv = np.array(self.pta.get_phiinv(self.pta.map_params(x0), logdet=False))
            else:
                phiinv = np.linalg.inv(self.get_phi_mat(x0))

            emp_phiinv = np.repeat(phiinv, 2, axis = 0)/self.renorm_const
            mean, chol_var = self.get_mean(emp_phiinv)
            a = self.a_given_phiinv(mean, chol_var)
            try:
                return self.phi_given_a(a, chi_sq = chi_fit)
            except FailException:
                continue

    def empiricalcoeff_to_iwphi(self, a_saved, a_first_idx, a_last_idx, a_is_scaled):
        '''
        Converts an empirical estimate of PSD to an IW phi matrix.

        param: `x0`: the parameterized PSD estimates 
        '''
        if not a_is_scaled:
            ac = a_saved[random.sample(range(a_first_idx, a_last_idx), k = int(5e3))] * np.sqrt(self.renorm_const)
        else:
            ac = a_saved[random.sample(range(a_first_idx, a_last_idx), k = int(5e3))]
        scale = torch.mean(ac.permute((0, 1, 3, 2)) @ ac, dim = 0).detach().cpu().numpy()
        scale = scale[0::2] + scale[1::2]
        return self.phi_given_a(a = None, sp = np.linalg.cholesky(scale), chi_sq=False)
    
    def phi_given_a(self, a, sp = None, chi_sq = True, cache_chol_phi = False,
                    lower_GWB_amp = -16, upper_GWB_amp = -13):
        '''
        Performs the `phiinv given a` step of the Gibbs sampling.
        This function samples from a non-truncated wishart distribution.

        param: `a`: the set of Fourier coefficients. The size must be (n_freq, 1, n_pulsar)
        param: `tol`: the amount added to the diagonals of the scale-matrtix to make the cholesky factorization stable.
        param: `check_diags`: whether to apply rejection sampling to the Inverse-wishart distribution
        param: `lower`: the lower limit of 0.5log10rho used in the rejection sampling
        param: `upper`: the upper limit of 0.5log10rho used in the rejection sampling

        Notes:
        L is an upper-triangular matrix
        SP is a lower-triangular matrix
        L @ L.T = S^-1
        SP @ SP.T = S
        A = Standard-Wishart
        phiinv = (LA) @ (LA).T
        phi = ((LA)^-1).T @ (LA)^-1
        LA = x = solve(SP.T, A)
        ((LA)^-1) = y.T = solve(A, SP.T)
        phiinv = x @ x.T
        phi = y @ y.T
        '''
        
        if not np.any(sp):
            sp  = self.get_scale_mat(a)
            
        for _ in range(self.fail_trial_count):

            A = self.standard_wishart()
            if A.ndim == 3:
                y = np.array([st_solve(A[fidx], sp[fidx].T, trans=0, lower=True, unit_diagonal=False, overwrite_b=False, check_finite=True) 
                        for fidx in self.c_idx])
            else:
                y = st_solve(A, sp.transpose((2, 1, 0)), trans=0, lower=True, unit_diagonal=False, overwrite_b=False, check_finite=True).transpose((2, 0, 1))
            yT = np.transpose(y, (0, 2, 1))
            
            phi = yT @ y
            if cache_chol_phi:
                self.chol_phi = yT

            phi_diags_raw = phi.diagonal(axis1 = 1, axis2 = 2)
            diags = 0.5 * np.log10(phi_diags_raw) 
            
            ## Check 1: Is phi bounded?
            if np.any(diags < self.lower_auto) or np.any(diags > self.upper_auto):
                bounds_flag = False   
                continue
            else:
                bounds_flag = True
                
            # Check 2: Are correlations ok?
            if chi_sq:
                amp_fit = self.chi_sq_fit(phi[None, :, self.i_cross, self.j_cross])
                if np.any(amp_fit < lower_GWB_amp) or np.any(amp_fit > upper_GWB_amp) or np.any(~np.isfinite(amp_fit)):
                # if np.any(~np.isfinite(amp_fit)):
                    bounds_flag = False   
                    continue
                else:
                    bounds_flag = True

            phiinv = np.repeat(np.linalg.inv(self.regularize(phi, return_fac=False)), 2, axis = 0)
        
            # Check 3: Is Sigma-matrix factorizable?
            try:
                mean, chol_var = self.get_mean(phiinv)
                sigma_flag = True
                break
            except np.linalg.LinAlgError:
                sigma_flag = False
                continue
        
        if bounds_flag and sigma_flag:
            # self.uppsilon = yT
            return phi, diags, mean, chol_var
        elif not bounds_flag:
            raise FailException(f'Wishart sampling failed due to phi matrix being unbounded. min = {diags.min()}, max = {diags.max()}')
        elif not sigma_flag:
            raise FailException(f'Wishart sampling failed due to phi matrix being numerically unstable.')


    def phi_given_a_little(self, sp, lower_GWB_amp = -16, upper_GWB_amp = -13):
        '''
        Performs the `phiinv given a` step of the Gibbs sampling.
        This function samples from a non-truncated wishart distribution.

        param: `a`: the set of Fourier coefficients. The size must be (n_freq, 1, n_pulsar)
        param: `tol`: the amount added to the diagonals of the scale-matrtix to make the cholesky factorization stable.
        param: `check_diags`: whether to apply rejection sampling to the Inverse-wishart distribution
        param: `lower`: the lower limit of 0.5log10rho used in the rejection sampling
        param: `upper`: the upper limit of 0.5log10rho used in the rejection sampling

        Notes:
        L is an upper-triangular matrix
        SP is a lower-triangular matrix
        L @ L.T = S^-1
        SP @ SP.T = S
        A = Standard-Wishart
        phiinv = (LA) @ (LA).T
        phi = ((LA)^-1).T @ (LA)^-1
        LA = x = solve(SP.T, A)
        ((LA)^-1) = y.T = solve(A, SP.T)
        phiinv = x @ x.T
        phi = y @ y.T
        '''

        for _ in range(self.fail_trial_count):

            A = self.standard_wishart()
            if A.ndim == 3:
                y = np.array([st_solve(A[fidx], sp[fidx].T, trans=0, lower=True, unit_diagonal=False, overwrite_b=False, check_finite=True) 
                        for fidx in self.c_idx])
            else:
                y = st_solve(A, sp.transpose((2, 1, 0)), trans=0, lower=True, unit_diagonal=False, overwrite_b=False, check_finite=True).transpose((2, 0, 1))
            yT = np.transpose(y, (0, 2, 1))
            
            phi = yT @ y
            diags = 0.5 * np.log10(phi.diagonal(axis1 = 1, axis2 = 2)) 
            
            ## Check 1: Is phi bounded?
            if np.any(diags < self.lower_auto) or np.any(diags > self.upper_auto):
                continue

            # Check 2: Are correlations ok?
            amp_fit = self.chi_sq_fit(phi[None, :, self.i_cross, self.j_cross])
            if np.any(amp_fit < lower_GWB_amp) or np.any(amp_fit > upper_GWB_amp) or np.any(~np.isfinite(amp_fit)):
                continue

            return diags - self.diag_offset, [amp_fit[0], self.chi_sq_fit_st(phi[None, :, self.i_cross, self.j_cross])[0]], phi[:, self.i_cross, self.j_cross]/self.renorm_const 

    def do_multi_gibbs(self, 
                       savedir,
                       emp_chain_path,
                       niter,
                       gibbs_weight = int(1e4),
                       empir_weight = 10,
                       gibbs_average_weight = 10,
                       resume = False,
                       save_diag_only = False, 
                       pbar_freq = int(5e3),
                       torch_device = 'cuda',
                       num_saved_samples = int(2e5),
                       a_is_scaled = False,
                       save_chol_phi = False, 
                       progress_bar = True):
        '''
        Performs a multi-pulsar gibbs sampling routine

        param: `niter`: The number of iterations
        param: `savedir`: the directory to save the outpur of the sampling
        param: `save_scale`: whether to save the sacale-matrix
        param: `save_phi`: whether to save the phi-matrix
        param: `save_diag_only`: whether to save the diagonals of the phi-matrix instead
        '''
        self.wishart_helper
        # ndraws = int(np.ceil(self.Npulsars/2))
        ndraws = int(1e4)
        pos = list(range(3))
        ew = empir_weight + gibbs_weight
        gw = empir_weight + gibbs_weight + gibbs_average_weight
        no_save_flag_thrshold = 2500

        os.makedirs(savedir, exist_ok=True)

        ###Empirical Noise Distribution
        try:
            phi_saved = np.load(emp_chain_path + '/chain_1.npy', mmap_mode='r')[:, :-4]
        except FileNotFoundError:
            phi_saved = np.load(emp_chain_path + '/chain.npy', mmap_mode='r')[:, :-4]
        print(phi_saved.shape)
        phi_saved_last_idx = phi_saved.shape[0] - 1
        phi, diags, mean, chol_var = self.empiricalphi_to_iwphi(phi_saved[random.randint(0, phi_saved_last_idx)])
        # a_saved = torch.tensor(np.load(emp_chain_path + '/amat.npy', mmap_mode='r')[..., None, :], dtype = torch.float32, device = torch_device)
        # burn_a_saved = int(0.25 * a_saved.shape[0])
        # a_last_idx = a_saved.shape[0]
        # phi, diags, mean, chol_var = self.empiricalcoeff_to_iwphi(a_saved, burn_a_saved, a_last_idx, a_is_scaled = a_is_scaled)
        a = self.a_given_phiinv(mean, chol_var)

        bools = np.ones(niter, dtype = bool)        
        bools[:int(0.25 * niter)] = False
        save_idxs = np.sort(random.sample(np.where(bools)[0].tolist(), k = num_saved_samples))

        if resume and os.path.isfile(savedir + '/phimat_diag.npy'):
            mode = 'r+'
            start_idx = np.nonzero(np.load(savedir + '/phimat_diag.npy', mmap_mode='r')[:, 0, 0])[0].max()
            print(f'Resumed with start index at {start_idx}.')
        else:
            mode = 'w+'
            start_idx = 0

        chain_diag = np.lib.format.open_memmap(savedir + '/phimat_diag.npy', 
                            mode=mode, 
                            dtype='float64', 
                            shape=(num_saved_samples, self.crn_bins, self.Npulsars), 
                            fortran_order=False)
        if not save_diag_only:                
            chain_phi = np.lib.format.open_memmap(savedir + '/phimat.npy', 
                            mode=mode, 
                            dtype='float64', 
                            shape=(num_saved_samples, self.crn_bins, len(self.i_cross)), 
                            fortran_order=False)
        if save_chol_phi:
            chain_chol_phi = np.lib.format.open_memmap(savedir + '/chol_phi.npy', 
                            mode=mode, 
                            dtype='float64', 
                            shape=(num_saved_samples, self.crn_bins, self.Npulsars, self.Npulsars), 
                            fortran_order=False)
            
        chain_a = np.lib.format.open_memmap(savedir + '/amat.npy', 
                    mode=mode, 
                    dtype='float64',
                    shape=(num_saved_samples, self.kmax, 1, self.Npulsars), 
                    fortran_order=False)

        # chain_a[0] = a
        
        ###Step 1 to n:
        if progress_bar:
            pbar = tqdm(range(start_idx, niter), colour="GREEN")
        else:
            pbar = range(start_idx, niter)
            st = time.time()

        no_save_flag = 0
        save_idx = 0
        stamp = 0
        for ii in pbar:
            try:

                if not ii%pbar_freq and ii and not progress_bar:
                    print(f'{round(ii/niter * 100, 2)} Percent Done in {round((time.time() - st)/60, 2)} Minutes.', end='\r')

                phi, diags, mean, chol_var = self.phi_given_a(a, cache_chol_phi = save_chol_phi)

                jump = random.choices(pos, cum_weights=[gibbs_weight, ew, gw])[0]

                if jump == 1:
                    no_save_flag = 0
                    phi_new, diags_new, mean_new, chol_var_new = self.empiricalphi_to_iwphi(phi_saved[random.randint(0, phi_saved_last_idx)])
                    # phi_new, diags_new, mean_new, chol_var_new = self.empiricalcoeff_to_iwphi(a_saved, burn_a_saved, a_last_idx, a_is_scaled = a_is_scaled)
                    if self.kick(phi, phi_new, mean, mean_new, chol_var, chol_var_new):
                        phi = phi_new
                        mean = mean_new
                        diags = diags_new
                        chol_var = chol_var_new  
                    else:
                        print('No empirical jump...')
                elif jump == 2 and save_idx > ndraws:
                    phi_new, diags_new, mean_new, chol_var_new = self.empiricalcoeff_to_iwphi(torch.tensor(chain_a[0:save_idx], dtype = torch.float32, device = 'cuda'), 0, save_idx, a_is_scaled = True)
                    if self.kick(phi, phi_new, mean, mean_new, chol_var, chol_var_new):
                        phi = phi_new
                        mean = mean_new
                        diags = diags_new
                        chol_var = chol_var_new
                        stamp = save_idx    
                    else:
                        print('No Gibbs average jump...')

            except FailException:
                print('Gibbs sampling failed! Force-kicking Gibbs...')
                # phi, diags, mean, chol_var = self.empiricalcoeff_to_iwphi(a_saved, burn_a_saved, a_last_idx, a_is_scaled = a_is_scaled)
                phi, diags, mean, chol_var = self.empiricalphi_to_iwphi(phi_saved[random.randint(0, phi_saved_last_idx)], chi_fit = False)

            a = self.a_given_phiinv(mean, chol_var)
            no_save_flag+=1
            
            ##################Saving###########################
            if save_idx < num_saved_samples:
                if ii == save_idxs[save_idx]:
                    chain_diag[save_idx] = diags - self.diag_offset
                    chain_a[save_idx] = a/np.sqrt(self.renorm_const)
                    if not save_diag_only:
                        chain_phi[save_idx] = phi[:, self.i_cross, self.j_cross]/self.renorm_const
                    if save_chol_phi:
                        chain_chol_phi[save_idx] = self.chol_phi/np.sqrt(self.renorm_const)
                    save_idx+=1
            else:
                break

    def gibbs_to_core(self, phi_diag, remove_txt = True, ceffyle_outdir = None):

        tm = gp_signals.MarginalizingTimingModel(use_svd=True)
        wn = blocks.white_noise_block(vary=False, inc_ecorr=False, select = 'none')
        rn = blocks.common_red_noise_block(psd='spectrum', prior='log-uniform', logmin=phi_diag.min(), logmax=phi_diag.max(),
                                                components=self.crn_bins)
        s = tm + wn + rn

        for pidx, psr in enumerate(self.psr):

            pta = signal_base.PTA([s(p) for p in [psr]], lnlikelihood = signal_base.LogLikelihoodDenseCholesky)
            outdir = ceffyle_outdir + '/core/' + f'/psr_{pidx}'
            os.makedirs(outdir, exist_ok = True)
            np.savetxt(outdir+'/pars.txt',list(map(str, pta.param_names)), fmt='%s')
            np.savetxt(outdir+'/priors.txt',list(map(lambda x: str(x.__repr__()), pta.params)), fmt='%s')
            np.savetxt(outdir + '/chain_1.txt', np.concatenate((phi_diag[:, :, pidx], np.ones((phi_diag.shape[0], 4))), 1))
            c = Core(chaindir = outdir, label = f'MultiPulsarGibbs psr {pidx}')
            freqs = np.arange(1/self.Tspan, (self.crn_bins + .01)/self.Tspan, 1/self.Tspan)
            c.set_rn_freqs(freqs)
            c.save(outdir + '/core.core')
            if remove_txt:
                paths = glob.glob(outdir + '/*.txt')
                for path in paths:
                    os.remove(path)

        kdes = densities.DE_factory(coredir=ceffyle_outdir + '/core',
                            pulsar_names = [psr.name for psr in self.psr],
                            recursive=True)
        kdes.setup_densities(outdir=ceffyle_outdir + '/kde')
        
    def do_PL_ceffyl(self, kde_dir, niter = int(1e6)):

        pta = Ceffyl.ceffyl(datadir=kde_dir, Tspan = self.Tspan)
        gw = Ceffyl.signal(psd=models.powerlaw, N_freqs=self.crn_bins, params=[Uniform(-18, -11)('log10_A'),
                                              Uniform(0, 7)('gamma')])
        if self.int_rn:
            irn = Ceffyl.signal(psd=models.powerlaw, N_freqs=self.crn_bins, common_process=False, selected_psrs=pta.pulsar_list,
                                params=[Uniform(-20, -11)('log10_A'),
                                              Uniform(0, 7)('gamma')])
            pta.add_signals([irn, gw])
        else:
            pta.add_signals([gw])

        x0 = pta.initial_samples()
        ndim = len(x0)
        cov = np.diag(np.ones(ndim) * 0.01**2)

        sampler = ptmcmc(ndim, pta.ln_likelihood, pta.ln_prior, cov,
                        outDir=kde_dir + '/MCMCChain', 
                        resume=False)
        jp = samp.JumpProposal(self.pta)
        sampler.addProposalToCycle(jp.draw_from_prior, 15)
        sampler.addProposalToCycle(jp.draw_from_red_prior, 15)
        sampler.sample(x0, niter, SCAMweight=30, AMweight=15, DEweight=50, )
        # sampler = Sampler.setup_sampler(pta, 
        #                     outdir=kde_dir + '/MCMCChain',
        #                     logL=pta.ln_likelihood,
        #                     logp=pta.ln_prior, resume=False)
        # sampler.sample(x0, niter)
            

class BayesPowerCURN(object):
    '''
    A class to perform a multi-pulsar analysis (CURN-type) using the ouput of Gibbs sampling

    param: `psrs`: a list of enterprise pulsar objects (Npulsars)
    param: `trees`: a list of KD-tree objects for all of the pulsars (Npulsar)
    param: `data_a`: a list of Fourier coefficients of all pulsars (Npulsar, Gsamples, 2 * Nbins)
    param: `joint_prob`: a list of joint probablity for p({a},{rho}) in case you want to re-weight the liklihood (Npulsar, Gsamples)
    param: `crn_bins`: the number of frequency bins
    param: `gw_psd_model`: the model to consider for the common red noise
    param: `Btrees`: a list of Ball-trees. It is faster to query, compared to KD-trees, for cases of crn_bins > 5
    param: `Tspan`: the baseline of the PTA in secodns
    param: `noise_dict`: noise-dictionary containing the white noise params.
    param: `gamma_val`: the spectral index of the common red noise
    param: `backend`: the backend to use
    param: `inc_ecorr`: whether to include ecorr
    param: 'int_rn': whether to include non-gwb red noise
    param: `distance_limit`: kd-querry distance threshold. Distances larger than this will result in `-np.inf` value for the likelihood 

    '''
    def __init__(self, psrs,
                 crn_bins, tnequad = True, 
                #  gw_psd_model = 'powerlaw', 
                 Tspan = None, noise_dict = None, gamma_val = 13/3,
                backend = 'none', inc_ecorr = False, int_rn = True):

        self.psr = psrs
        self.Npulsars = len(self.psr)
        self.noise_dict = noise_dict
        self.crn_bins = crn_bins
        self.kmax = 2 * self.crn_bins
        if Tspan:
            self.Tspan = Tspan
        else:
            self.Tspan = model_utils.get_tspan(self.psr)
        self.freqs = np.arange(1/self.Tspan, (self.crn_bins + .01)/self.Tspan, 1/self.Tspan)
        self.fref = 1/(1 * 365.25 * 24 * 60 * 60)
        
        self.diag_idx = np.arange(0, self.Npulsars, 1, int)
        self.c_idx = np.arange(0, self.crn_bins, 1, int)
        self.k_idx = np.arange(0, self.kmax, 1, int)
        self.total_dim = self.Npulsars * self.kmax
        self.int_rn = int_rn

        self.sampler_CURN = None

        ###Make a CURN PTA Object (needed to construct matricies from ENTERPRISE)
        tm = gp_signals.MarginalizingTimingModel(use_svd=True)
        wn = blocks.white_noise_block(vary=False, inc_ecorr=inc_ecorr, gp_ecorr=False, select = backend, tnequad = tnequad)
        rn = blocks.red_noise_block(psd='powerlaw', prior='log-uniform',Tspan = self.Tspan,
                                                components=self.crn_bins, gamma_val=None)
        crn = blocks.common_red_noise_block(psd='powerlaw', prior='log-uniform', Tspan = self.Tspan, orf = 'crn',
                                                components=self.crn_bins, gamma_val=gamma_val, name = 'gw')
        if int_rn:
            s = tm + wn + rn + crn
        else:
            s = tm + wn + crn

        self.pta = signal_base.PTA([s(p) for p in self.psr], lnlikelihood = signal_base.LogLikelihoodDenseCholesky)
        self.pta.set_default_params(self.noise_dict)

        # self._TNT_sparse = sps.block_diag(self.pta.get_TNT(params = {}), "csc")
        self._TNT = np.array(self.pta.get_TNT(params = {}))
        self._TNr = np.concatenate(self.pta.get_TNr(params = {}))
        self.x0 = np.hstack([p.sample() for p in self.pta.params])

        self.pmin = []
        self.pmax = []
        for p in self.pta.params:
            sp = str(p)
            val_min = float(sp.split('pmin=')[1].split(',')[0])
            val_max = float(sp.split('pmax=')[1].split(')')[0])
            # if not 'rho' in sp:
            if not ')[' in sp:
                self.pmin.append(val_min)
                self.pmax.append(val_max)
            else:
                nf_bins = int(sp.split(')[')[1].split(']')[0])
                for _ in range(nf_bins):
                    self.pmin.append(val_min)
                    self.pmax.append(val_max) 

    def pl_to_rho(self, log10amp, gamma):
        '''
        A function to turn log10 of amplitude and spectral index of a red noise process 
        into PSD values. 

        param: `log10amp`: log10 of amplitude of the red noise process
        param: `gamma`: spectral index of the red noise process
        '''
        return 10**(2*log10amp)/(12 * np.pi**2 * self.freqs[:, None]**3 * self.Tspan) * (self.freqs[:, None]/self.fref)**(3-gamma)
    
    def get_phi_mat(self, xs):
        '''
        A function to turn the model parameters into a red noise covaraince matrix (phi)

        param: `xs`: the model parameters. The model must be a powerlaw model!
        '''
        if self.int_rn: 
            end_idx = 2 * self.Npulsars
            phi = np.zeros((self.crn_bins, self.Npulsars))

            psd_intrin = self.pl_to_rho(xs[1:end_idx:2], xs[0:end_idx:2])
            psd_common = self.pl_to_rho(xs[end_idx + 1], xs[end_idx])
        
            phi[:, self.diag_idx] = psd_intrin + psd_common

            return np.repeat(phi.T, 2, axis = 1)
        else:
            end_idx = 0
            phi = np.zeros((self.crn_bins, self.Npulsars))

            psd_common = self.pl_to_rho(xs[end_idx + 1], xs[end_idx])
        
            phi[:, self.diag_idx] = psd_common

            return np.repeat(phi.T, 2, axis = 1)            

    def get_mean_var(self, phiinv):
        '''
        Estimates the mean and the variance of the `a` distribution given an estimate of the `phhinv` matrix.

        param: `phiinv`: the phiinv matrix. The dimensions are (n_freq * n_pulsar by n_freq * n_pulsar).
        param: `tol`: the amount added to the diagonals of TNT + phiinv to make the cholesky factorization stable.
        '''
        Sigma = np.zeros((self.Npulsars, self.kmax, self.Npulsars, self.kmax))
        x = self._TNT.copy()
        x[:, self.k_idx, self.k_idx] += phiinv
        Sigma[self.diag_idx, :, self.diag_idx, :] = x
        Sigma = Sigma.reshape(self.total_dim, self.total_dim)
        try:
            cf = sl.cho_factor(Sigma, lower=False)
            return sl.cho_solve(cf, self._TNr), cf[0]
        except np.linalg.LinAlgError:
            return -np.inf
    
    def a_given_rho(self, phiinv):
        '''
        Performs the analytical draw of `a given phiinv`.

        param: `phiinv`: the phiinv matrix. The dimensions are (n_freq * n_pulsar by n_freq * n_pulsar).
        '''
        mean, chol_var = self.get_mean_var(phiinv)
        rand_vec = np.random.randn(self.total_dim)
        a = mean + st_solve(chol_var, rand_vec, trans=0, lower=False, unit_diagonal=False, overwrite_b=True, check_finite=False)
        return a.reshape(self.Npulsars, self.kmax)

    def get_lnliklihood(self, xs, phi = [None]):
        '''
        A function to return natural log of the CURN likelihood

        param: `xs`: powerlaw model parameters
        param: `phi`: if given the model parameters are ignored.
        '''
        if not np.any(phi):
            phi = self.get_phi_mat(xs)
        return - 0.5 * np.sum(np.log(phi) + self._a**2/phi)

    def get_lnprior(self, xs):
        '''
        A function to return natural log prior (uniform)

        param: `xs`: powerlaw model parameters
        '''
        if np.any(xs < self.pmin) or np.any(xs > self.pmax):
            return -np.inf
        else:
            return -8.01
          
    def do_CURN(self, xs, iters=30):
        '''
        Function to perform CURN param updates.
        '''
        x0 = xs.copy()
        lnlike0, lnprior0  = self.get_lnliklihood(x0), self.get_lnprior(x0)
        lnprob0 = lnlike0 + lnprior0

        for ii in range(self.start_CURN_iter + 1, self.start_CURN_iter + iters + 1):
            x0, lnlike0, lnprob0 = self.sampler_CURN.PTMCMCOneStep(x0, lnlike0, lnprob0, ii)
        xnew = x0
        self.start_CURN_iter = ii
        return xnew 
    
    def sample(self, niter, savedir, resume = True, CURNiters = 30, a0 = None):
        '''
        A function to perform the sampling for CURN

        param: `niter`: the number of sampling iterations
        param: `savedir`: the directory to save the chains
        param: `CURNiters`: the number of MCMC steps per analytic Fourier coefficient draws
        param: `a0`: if given, the starting guess is for Fourier coefficient rather than `PSD` params. 
        '''
        self.start_CURN_iter = 0

        if not self.sampler_CURN:
            print('Setting up the sampler for the first time...')
            x0 = self.x0
            ndim = len(x0)
            isave = int(4e9)
            thin = 1
            Niter = int(CURNiters * (niter + 1) + 1)
            cov = np.diag(np.ones(ndim) * 0.01**2) # helps to tune MCMC proposal distribution
            pars = self.pta.param_names
            idx_gw_params = [list(pars).index(pp) for pp in pars if 'gw' in pp]
            groups = [list(np.arange(0, ndim))]
            [groups.append(idx_gw_params) for ii in range(5)]

            self.sampler_CURN = ptmcmc(ndim, self.get_lnliklihood, self.get_lnprior, cov,
                                    groups = groups,
                                    outDir = savedir,
                                    resume=False)
            jp = samp.JumpProposal(self.pta)
            self.sampler_CURN.addProposalToCycle(jp.draw_from_prior, 15)
            if self.int_rn:
                self.sampler_CURN.addProposalToCycle(jp.draw_from_red_prior, 15)
            self.sampler_CURN.initialize(Niter = Niter, isave = isave, thin = thin, SCAMweight=30, #mmap = True,
                                        AMweight=15, DEweight=50, covUpdate = 1000,
                                        burn = 10000)
            
        if resume and os.path.isfile(savedir + '/chain.npy') and os.path.isfile(savedir + '/amat.npy'):
        
            chain = np.lib.format.open_memmap(savedir + '/chain.npy', 
                            mode='r+', 
                            dtype='float64',
                            shape=(niter, len(x0)))
            amat = np.lib.format.open_memmap(savedir + '/amat.npy', 
                            mode='r+', 
                            dtype='float64',
                            shape=(niter, self.kmax, self.Npulsars))
            start_idx = np.nonzero(chain[:, 0])[0].max()
            if niter <= start_idx:
                print('The run is already done. Exitting...')
                return None
            
            print(f'Resumed from {savedir}. {np.round(start_idx/niter * 100, 2)} percent done already')
            xnew = chain[start_idx]
        else:
            chain = np.lib.format.open_memmap(savedir + '/chain.npy', 
                            mode='w+', 
                            dtype='float64',
                            shape=(niter, len(x0)))
            amat = np.lib.format.open_memmap(savedir + '/amat.npy', 
                            mode='w+', 
                            dtype='float64',
                            shape=(niter, self.kmax, self.Npulsars))    
            start_idx = 0
            ######Start of the 1st Step
            flag = True
            while(flag):
                x0 = np.hstack([p.sample() for p in self.pta.params])
                phiinv = 1/self.get_phi_mat(x0)
                if not np.any(a0):
                    self._a = self.a_given_rho(phiinv)
                else:
                    self._a = a0
                if not np.all(self._a == -np.inf):
                    self.x0 = x0
                    flag = False
            xnew = self.do_CURN(x0, iters = CURNiters)
        
        st = time.time()
        pbar_freq = int(5e3)
        ######End of the 1st Step
        # for ii in range(start_idx, niter):
        for ii in tqdm(range(start_idx, niter)):
            if not ii%pbar_freq and ii:
                print(f'{round(ii/niter * 100, 2)} Percent Done in {round((time.time() - st)/60, 2)} Minutes.\n')
                
            phiinv = np.array(self.pta.get_phiinv(self.pta.map_params(xnew), logdet=False))
            self._a = self.a_given_rho(phiinv)
            xnew = self.do_CURN(xnew, iters = CURNiters)

            amat[ii] = self._a.T
            chain[ii] = xnew
        return chain
    
class BayesPowerResample(object):
    '''
    A class to perform a multi-pulsar analysis (CURN-type) using the ouput of Gibbs sampling

    param: `psrs`: a list of enterprise pulsar objects (Npulsars)
    param: `trees`: a list of KD-tree objects for all of the pulsars (Npulsar)
    param: `data_a`: a list of Fourier coefficients of all pulsars (Npulsar, Gsamples, 2 * Nbins)
    param: `joint_prob`: a list of joint probablity for p({a},{rho}) in case you want to re-weight the liklihood (Npulsar, Gsamples)
    param: `crn_bins`: the number of frequency bins
    param: `gw_psd_model`: the model to consider for the common red noise
    param: `Btrees`: a list of Ball-trees. It is faster to query, compared to KD-trees, for cases of crn_bins > 5
    param: `Tspan`: the baseline of the PTA in secodns
    param: `noise_dict`: noise-dictionary containing the white noise params.
    param: `gamma_val`: the spectral index of the common red noise
    param: `backend`: the backend to use
    param: `inc_ecorr`: whether to include ecorr
    param: 'int_rn': whether to include non-gwb red noise
    param: `distance_limit`: kd-querry distance threshold. Distances larger than this will result in `-np.inf` value for the likelihood 

    '''
    def __init__(self, psrs,
                 crn_bins, orf = 'crn', gw_psd_model = 'powerlaw', int_psd_model = 'powerlaw', Tspan = None, noise_dict = None,
                 tnequad = False, int_bins = 30,
                 smat_dir = None,smat_burn_idx = None, smat_last_idx = None, data_base = None,
                backend = 'none', inc_ecorr = False, int_rn = True):

        self.psr = psrs
        self.data_base = data_base
        self.orf = orf
        self.noise_dict = noise_dict
        self.Npulsars = len(self.psr)
        self.crn_bins = crn_bins
        self.int_bins = int_bins
        self.kmax = 2 * self.int_bins
        self.fref = 1/(1 * 365.25 * 24 * 60 * 60)
        self.int_rn = int_rn
        self.gw_psd_model = gw_psd_model
        self.int_psd_model = int_psd_model

        if Tspan:
            self.Tspan = Tspan
        else:
            self.Tspan = model_utils.get_tspan(self.psr)

        self.f_intrin = np.arange(1/self.Tspan, (self.int_bins + 0.01)/self.Tspan, 1/self.Tspan)
        self.f_common = self.f_intrin[:crn_bins]
        
        self.diag_idx = np.arange(0, self.Npulsars, 1, int)
        self.diag_large = np.arange(0, self.Npulsars * self.kmax, 1, int)
        self.c_idx = np.arange(0, self.crn_bins, 1, int)
        self.k_idx = np.arange(0, self.kmax, 1, int)
        self.ppair_idx = np.arange(0, int(self.Npulsars * (self.Npulsars - 1) * 0.5), 1, int)

        self.smat_dir = smat_dir
        self.smat_burn_idx = smat_burn_idx
        self.smat_last_idx = smat_last_idx

        ###Make a PTA Object
        tm = gp_signals.MarginalizingTimingModel(use_svd=True)
        wn = blocks.white_noise_block(vary=False, inc_ecorr=inc_ecorr, gp_ecorr=False, select = backend, tnequad = tnequad)
        rn = blocks.red_noise_block(psd=self.int_psd_model, prior='log-uniform',Tspan = self.Tspan,
                                    #logmin=-11, logmax=-0.5,
                                                components=int_bins, gamma_val=None)
        crn = blocks.common_red_noise_block(psd=self.gw_psd_model, prior='log-uniform', Tspan = self.Tspan, orf = orf,
                                            #logmin=-11, logmax=-0.5,
                                                components=self.crn_bins, gamma_val=None, name = 'gw')
        if int_rn:
            s = tm + wn + rn + crn
        else:
            s = tm + wn + crn

        self.pta = signal_base.PTA([s(p) for p in self.psr])
        self.pta.set_default_params(self.noise_dict)
        self.x0 = np.hstack([p.sample() for p in self.pta.params])

        # self._TNT = sps.block_diag(self.pta.get_TNT(params = {}), "csc")
        self._TNT = sl.block_diag(*self.pta.get_TNT(params = {}))
        try:
            self._facTNT = np.linalg.cholesky(self._TNT)
        except np.linalg.LinAlgError:
            U, S, Vh = np.linalg.svd(self._TNT)
            self._TNT = U @ np.diag(S) @ U.T
            self._facTNT = U @ np.diag(np.sqrt(S))
        self._TNT = sps.csc_matrix(self._TNT)
        self._TNr = np.concatenate(self.pta.get_TNr(params = {}))
        self._rNr_logdet = -0.5 * np.sum([ell for ell in self.pta.get_rNr_logdet(params = {})])

        self.pmin = []
        self.pmax = []
        for p in self.pta.params:
            sp = str(p)
            val_min = float(sp.split('pmin=')[1].split(',')[0])
            val_max = float(sp.split('pmax=')[1].split(')')[0])
            # if not 'rho' in sp:
            if not ')[' in sp:
                self.pmin.append(val_min)
                self.pmax.append(val_max)
            else:
                nf_bins = int(sp.split(')[')[1].split(']')[0])
                for _ in range(nf_bins):
                    self.pmin.append(val_min)
                    self.pmax.append(val_max) 

    def sparcify(self, phiinv):
        '''
        Overwrites the `data` content of the cached sparse phiinv matrix.
        This is needed for maximum speed! `random_phiinv(self, set_sparse = False)`
        must have been called once prior to calling this function.

        param: `phiinv`: the dense phiinv matrix (2*n_freq, n_pulsar, n_pulsar).
        '''
        if self.crn_bins == self.int_bins:
            self._sparse_phiinv.data = phiinv.transpose((1, 0, 2)).flatten()
        else:
            x = phiinv.transpose((1, 0, 2)).flatten()
            self._sparse_phiinv.data = x[np.nonzero(x)]
        return self._sparse_phiinv

    def to_ent_phiinv(self, phiinv):
        '''
        Changes the format of the phiinv matrix from (n_freq, n_pulsar, n_pulsar) to (n_freq * n_pulsar by n_freq * n_pulsar).

        param: `phiinv`: the phiinv matrix.
        '''
        phiinv_ent = np.zeros((self.Npulsars, self.kmax, self.Npulsars, self.kmax))
        phiinv_ent[:, self.k_idx, :, self.k_idx] = phiinv
        return phiinv_ent.reshape((self.Npulsars * self.kmax, self.Npulsars * self.kmax))

    def random_phiinv(self, x0, set_sparse = False, dense = False):
        '''
        Returns a random phiinv matrix.

        param: `set_sparse`: whether to cache a sparse phiinv matrix.
        param: `dense`: whether to return a dense (n_freq, n_pulsar, n_pulsar)
        phiinv matrix instead of a (n_freq * n_pulsar by n_freq * n_pulsar) matrix
        '''
        phiinv_ent = np.array(self.pta.get_phiinv(self.pta.map_params(x0), logdet=False))
        if set_sparse:
            self._sparse_phiinv = sps.csc_matrix(phiinv_ent)
        if dense:
            return np.array([phiinv_ent[fidx::self.kmax,fidx::self.kmax] for fidx in self.k_idx])
        else:
            return phiinv_ent

    def prepare_standard_run(self, x0):

        self.get_cross_info
        self.random_phiinv(x0, set_sparse = True, dense = True)

    def get_standard_ln_like(self, xs):
        
        params = self.pta.map_params(xs)
        loglike = 0
        loglike += -0.5 * np.sum([ell for ell in self.pta.get_rNr_logdet(params)])
        loglike += sum(self.pta.get_logsignalprior(params))

        if self.int_psd_model == 'powerlaw':
            if self.int_rn:
                dens_phi = self.get_phi_mat(xs)
            else:
                dens_phi = self.get_phi_mat_no_IRN(xs)
        elif self.int_psd_model == 'spectrum':
            if self.int_rn:
                dens_phi = self.get_phi_mat_spec(xs)
            else:
                dens_phi = self.get_phi_mat_no_IRN(xs)

        try:
            logdet_phi = 4 * np.sum(np.log(np.linalg.cholesky(dens_phi)[:, self.diag_idx, self.diag_idx]))
            phiinv_dense = np.linalg.inv(dens_phi)

        except np.linalg.LinAlgError:
            return -np.inf
        try:
            cf = cholesky(self._TNT + self.sparcify(np.repeat(phiinv_dense, 2, axis = 0)))
            expval = cf(self._TNr)
            logdet_sigma = cf.logdet()
        except CholmodError:
            return -np.inf

        loglike += 0.5 * (np.dot(self._TNr, expval) - logdet_sigma - logdet_phi)
        return loglike

    # def get_standard_ln_like(self, xs):
        
    #     loglike = self.TN
    #     if self.int_psd_model == 'powerlaw':
    #         if self.int_rn:
    #             dens_phi = self.get_phi_mat(xs)
    #         else:
    #             dens_phi = self.get_phi_mat_no_IRN(xs)
    #     elif self.int_psd_model == 'spectrum':
    #         if self.int_rn:
    #             dens_phi = self.get_phi_mat_spec(xs)
    #         else:
    #             dens_phi = self.get_phi_mat_no_IRN(xs)

    #     try:
    #         logdet_phi = 4 * np.sum(np.log(np.linalg.cholesky(dens_phi)[:, self.diag_idx, self.diag_idx]))
    #         phiinv_dense = np.linalg.inv(dens_phi)

    #         Sigma = self._TNT + self.to_ent_phiinv(np.repeat(phiinv_dense, 2, axis = 0))
    #         cf = sl.cho_factor(Sigma, lower=False)
    #         expval = sl.cho_solve(cf, self._TNr)
    #         logdet_sigma = 2 * np.sum(np.log(cf[0].diagonal()))
    #         loglike += 0.5 * (np.dot(self._TNr, expval) - logdet_sigma - logdet_phi)
    #         return loglike
    #     except np.linalg.LinAlgError:
    #         return -np.inf

    @cached_property
    def get_hd_val(self):
        return HD_ORF(self.xi)
    
    @cached_property
    def get_cross_info(self):
        
        I, J = np.array(list(combinations(range(self.Npulsars),2))).T
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


    def common_pl_to_rho(self, log10amp, gamma):
        return 10**(2*log10amp)/(12 * np.pi**2 * self.f_common[:, None]**3 * self.Tspan) * (self.f_common[:, None]/self.fref)**(3-gamma)

    def intrin_pl_to_rho(self, log10amp, gamma):
        return 10**(2*log10amp)/(12 * np.pi**2 * self.f_intrin[:, None]**3 * self.Tspan) * (self.f_intrin[:, None]/self.fref)**(3-gamma)

    def get_phi_mat(self, xs, return_full = True):

        end_idx = 2 * self.Npulsars

        phi = np.zeros((self.int_bins, self.Npulsars, self.Npulsars))

        psd_intrin = self.intrin_pl_to_rho(xs[1:end_idx:2], xs[0:end_idx:2])
        if self.gw_psd_model == 'powerlaw':
            end_orf_idx = end_idx + 2
            psd_common = self.common_pl_to_rho(xs[end_idx + 1], xs[end_idx])
        else:
            end_orf_idx = end_idx + self.crn_bins
            psd_common = 10**(2*xs[end_idx:end_orf_idx])[:, None]

        phi[:self.crn_bins, self.diag_idx, self.diag_idx] = psd_intrin[:self.crn_bins] + psd_common
        phi[self.crn_bins:, self.diag_idx, self.diag_idx] = psd_intrin[self.crn_bins:]

        if self.orf == 'hd':
            phi[:self.crn_bins, self.I, self.J] = self.get_hd_val * psd_common

        elif self.orf == 'crn':
            return phi[:, self.diag_idx, self.diag_idx]
        
        elif self.orf == 'bin_orf':
            phi[:self.crn_bins, self.I, self.J] = bin_orf(self.xi, xs[end_orf_idx:]) * psd_common

        elif self.orf == 'gt':
            phi[:self.crn_bins, self.I, self.J] = gt_orf(self.xi, xs[end_orf_idx:]) * psd_common

        if return_full:
            phi[:self.crn_bins, self.J, self.I] = phi[:self.crn_bins, self.I, self.J]

        return phi

    def get_phi_mat_no_IRN(self, xs, return_full = True):

        end_idx = 0

        phi = np.zeros((self.crn_bins, self.Npulsars, self.Npulsars))

        if self.gw_psd_model == 'powerlaw':
            end_orf_idx = end_idx + 2
            psd_common = self.common_pl_to_rho(xs[end_idx + 1], xs[end_idx])
        else:
            end_orf_idx = end_idx + self.crn_bins
            psd_common = 10**(2*xs[end_idx:end_orf_idx])[:, None]

        phi[:, self.diag_idx, self.diag_idx] = psd_common

        if self.orf == 'hd':
            phi[:, self.I, self.J] = self.get_hd_val * psd_common

        elif self.orf == 'crn':
            return phi[:, self.diag_idx, self.diag_idx]
        
        elif self.orf == 'bin_orf':
            phi[:, self.I, self.J] = bin_orf(self.xi, xs[end_orf_idx:]) * psd_common

        elif self.orf == 'gt':
            phi[:, self.I, self.J] = gt_orf(self.xi, xs[end_orf_idx:]) * psd_common

        if return_full:
            phi[:, self.J, self.I] = phi[:self.crn_bins, self.I, self.J]

        return phi
    
    def get_phi_mat_spec(self, xs, return_full = True):
        
        if self.gw_psd_model == 'spectrum':
            end_idx = self.int_bins * self.Npulsars

            phi = np.zeros((self.int_bins, self.Npulsars, self.Npulsars))

            psd_intrin = 10**(2*xs[0:end_idx]).reshape(self.Npulsars, self.int_bins).T

            end_orf_idx = end_idx + self.crn_bins
            psd_common = 10**(2*xs[end_idx:end_orf_idx])[:, None]

            phi[:self.crn_bins, self.diag_idx, self.diag_idx] = psd_intrin[:self.crn_bins] + psd_common
            phi[self.crn_bins:, self.diag_idx, self.diag_idx] = psd_intrin[self.crn_bins:]

            if self.orf == 'hd':
                phi[:self.crn_bins, self.I, self.J] = self.get_hd_val * psd_common

            elif self.orf == 'crn':
                return phi[:, self.diag_idx, self.diag_idx]
            
            elif self.orf == 'bin_orf':
                phi[:self.crn_bins, self.I, self.J] = bin_orf(self.xi, xs[end_orf_idx:]) * psd_common

            elif self.orf == 'gt':
                phi[:self.crn_bins, self.I, self.J] = gt_orf(self.xi, xs[end_orf_idx:]) * psd_common

            if return_full:
                phi[:self.crn_bins, self.J, self.I] = phi[:self.crn_bins, self.I, self.J]

            return phi

    def regularize(self, herm_mat, return_fac):
        '''
        Regularizes a rank-deficient real symmetric matrix into a full-rank matrix
        param: `herm_mat`: the matrix to perform regularization on. The shape must be
                            (n_freq by n_pulsar by n_pulsar)
        param: `return_fac`: wheher to return the square-root factorization
        '''
        delta_est = np.random.uniform(1e-8, 1e-5, size = (self.crn_bins, self.Npulsars))
        sqr_diags = np.sqrt(herm_mat.diagonal())[..., None]
        D = sqr_diags @ sqr_diags.T
        corr = herm_mat/D
        corr[self.diag_large, self.diag_large]+=delta_est
        # corr = corr/(1+delta_est[..., None])
        if return_fac:
            return sqr_diags * np.linalg.cholesky(corr)
        else:
            return corr * D
        
    def get_lnliklihood(self, xs):
        '''
        likelihood
        '''  
        if self.int_rn:
            phi = self.get_phi_mat(xs, return_full = True)
        else:
            phi = self.get_phi_mat_no_IRN(xs, return_full = True)
        if self.orf == 'crn':
            tau = self._s**2
            x = 0.5 * (tau[0::2] + tau[1::2])/phi
            return -np.sum(x + np.log(phi))
        else:
            try:
                cp = np.repeat(np.linalg.cholesky(phi), 2, axis = 0)
                logdet_p = 2 * np.sum(np.log(cp[:, self.diag_idx, self.diag_idx]), axis = 1)[:, None, None]
                x = np.array([st_solve(cp[fidx], self._s[fidx], trans=0, lower=True, unit_diagonal=False, overwrite_b=False, check_finite=False) 
                        for fidx in self.k_idx])[..., None]
                return -0.5 * np.sum(logdet_p + x.transpose((0, 2, 1)) @ x)
            except np.linalg.LinAlgError:
                return -np.inf  
                            
    # def get_lnliklihood(self, xs):
    #     '''
    #     likelihood
    #     '''  
    #     if self.int_rn:
    #         phi = self.get_phi_mat(xs, return_full = True)
    #     else:
    #         phi = self.get_phi_mat_no_IRN(xs, return_full = True)
    #     if self.orf == 'crn':
    #         x = 0.5 * self._s/phi
    #         return -np.sum(x + np.log(phi))
    #     else:
    #         try:
    #             cp = np.linalg.cholesky(phi)
    #             logdet_p = 2 * np.sum(np.log(cp[:, self.diag_idx, self.diag_idx]), axis = 1)   
    #             x = np.linalg.solve(phi, self._s)
    #             return - 0.5 * np.sum(2*logdet_p + x.trace(axis1 = 1, axis2 = 2))
    #         except np.linalg.LinAlgError:
    #             return -np.inf  

    def get_lnprior(self, xs):

        if np.any(xs < self.pmin) or np.any(xs > self.pmax):
            return -np.inf
        else:
            return -8.01

    def do_MCMC(self, xs, iters=10):
        '''
        Function to perform param updates.
        '''
        x0 = xs.copy()
        lnlike0, lnprior0  = self.get_lnliklihood(x0), self.get_lnprior(x0)
        lnprob0 = lnlike0 + lnprior0

        for ii in range(self.start_CURN_iter + 1, self.start_CURN_iter + iters + 1):
            x0, lnlike0, lnprob0 = self.sampler_CURN.PTMCMCOneStep(x0, lnlike0, lnprob0, ii)
        xnew = x0
        self.start_CURN_iter = ii
        return xnew 
    
    def sample(self, niter, outdir, savepath, CURNiters = 30, SCAMweight=30, 
               AMweight=15, DEweight=50, 
               covUpdate = 1000, burn = 10000):

        self.start_CURN_iter = 0
        x0 = self.x0
        ndim = len(x0)
        isave = int(1e10)
        thin = 1
        Niter = int(CURNiters * (niter + 1) + 1)
        cov = np.diag(np.ones(ndim) * 0.01**2) # helps to tune MCMC proposal distribution
        pars = self.pta.param_names

        groups = [list(np.arange(0, ndim))]
        idx_orf_params0 = None
        if self.orf == 'bin_orf':
            idx_orf_params0 = [list(pars).index(pp) for pp in pars if 'gw_orf_bin' in pp]
            x0[idx_orf_params0] = .1
        elif self.orf == 'gt':
            idx_orf_params0 = [list(pars).index(pp) for pp in pars if 'tau' in pp]
            x0[idx_orf_params0] = 0.5
        elif self.orf == 'crn':
            print('good')
            idx_gw_params = [list(pars).index(pp) for pp in pars if 'gw' in pp]
            [groups.append(idx_gw_params) for ii in range(5)]
        
        if np.any(idx_orf_params0):    
            [groups.append(idx_orf_params0) for ii in range(2)]

        self.sampler_CURN = ptmcmc(ndim, self.get_lnliklihood, self.get_lnprior, cov, groups = groups,
                                outDir = savepath,
                                resume=False)
        jp = samp.JumpProposal(self.pta)
            
        self.sampler_CURN.addProposalToCycle(jp.draw_from_prior, 5)
        # self.sampler_CURN.addProposalToCycle(jp.draw_from_par_prior(
        #         par_names=[str(p).split(':')[0] for
        #                    p in list(self.pta.params)
        #                    if 'gw' in str(p)]), 10)
        
        if self.orf == 'bin_orf':
            self.sampler_CURN.addProposalToCycle(jp.draw_from_par_prior(
                    par_names=[str(p).split(':')[0] for
                            p in list(self.pta.params)
                            if 'bin' in str(p)]), 10)

        if self.orf == 'gt':
            self.sampler_CURN.addProposalToCycle(jp.draw_from_par_prior(
                par_names=[str(p).split(':')[0] for
                           p in list(self.pta.params)
                           if 'tau' in str(p)]), 10)
        if self.int_rn:
            self.sampler_CURN.addProposalToCycle(jp.draw_from_red_prior, 10)
        
        self.sampler_CURN.initialize(Niter = Niter, isave = isave, thin = thin, SCAMweight=SCAMweight,
                                    AMweight=AMweight, DEweight=DEweight, covUpdate = covUpdate,
                                    burn = burn)

        os.makedirs(outdir, exist_ok=True)
        chain = np.lib.format.open_memmap(outdir + '/Resample_chain_2.npy', 
                                        mode='w+', 
                                        dtype='float32', 
                                        shape=(niter, len(x0)), 
                                        fortran_order=False)

        ######Start of the 1st Step
        # if not self.orf == 'crn':
        #     self.get_cross_info
        # try:
        #     smat = np.load(self.smat_dir, mmap_mode='r')[:, :, 0, :]#/np.sqrt(1e9)
        # except IndexError:
        #     smat = np.load(self.smat_dir, mmap_mode='r')#/np.sqrt(1e9)

        # self._s = smat[random.randint(self.smat_burn_idx, self.smat_last_idx)]
        # print(self._s)
        # xnew = self.do_MCMC(x0, iters = CURNiters)
        # print(f'The last ln-likelihood of the first step is: {self.get_lnliklihood(xnew)}')
        # ######End of the 1st Step
        # pbar = tqdm(range(niter), colour="BLUE")
        # for ii in pbar:
        #     randidx = self.data_base.search(xnew, 1).keys[0]
        #     self._s = smat[randidx]
        #     xnew = self.do_MCMC(xnew, iters = CURNiters)
        #     chain[ii, :] = xnew
        
        ######Start of the 1st Step
        if not self.orf == 'crn':
            self.get_cross_info
        try:
            smat = np.load(self.smat_dir, mmap_mode='r')[:, :, 0, :]/np.sqrt(1e9)
        except IndexError:
            smat = np.load(self.smat_dir, mmap_mode='r')/np.sqrt(1e9)

        self._s = smat[random.randint(self.smat_burn_idx, self.smat_last_idx)]
        print(self._s)
        xnew = self.do_MCMC(x0, iters = CURNiters)
        print(f'The last ln-likelihood of the first step is: {self.get_lnliklihood(xnew)}')
        ######End of the 1st Step
        pbar = tqdm(range(niter), colour="BLUE")
        for ii in pbar:
            self._s = smat[random.randint(self.smat_burn_idx, self.smat_last_idx)]
            xnew = self.do_MCMC(xnew, iters = CURNiters)
            chain[ii, :] = xnew
    

class BayesPowerKDE(object):
    '''
    A class to perform a multi-pulsar analysis (CURN-type) using the ouput of Gibbs sampling

    param: `psrs`: a list of enterprise pulsar objects (Npulsars)
    param: `trees`: a list of KD-tree objects for all of the pulsars (Npulsar)
    param: `data_a`: a list of Fourier coefficients of all pulsars (Npulsar, Gsamples, 2 * Nbins)
    param: `joint_prob`: a list of joint probablity for p({a},{rho}) in case you want to re-weight the liklihood (Npulsar, Gsamples)
    param: `crn_bins`: the number of frequency bins
    param: `gw_psd_model`: the model to consider for the common red noise
    param: `Btrees`: a list of Ball-trees. It is faster to query, compared to KD-trees, for cases of crn_bins > 5
    param: `Tspan`: the baseline of the PTA in secodns
    param: `noise_dict`: noise-dictionary containing the white noise params.
    param: `gamma_val`: the spectral index of the common red noise
    param: `backend`: the backend to use
    param: `inc_ecorr`: whether to include ecorr
    param: 'int_rn': whether to include non-gwb red noise
    param: `distance_limit`: kd-querry distance threshold. Distances larger than this will result in `-np.inf` value for the likelihood 

    '''
    def __init__(self, psrs, grid, den,
                 crn_bins, gw_psd_model = 'powerlaw', int_psd_model = 'powerlaw', Tspan = None, noise_dict = None,
                 tnequad = False, int_bins = 5,
                backend = 'none', inc_ecorr = False, int_rn = True):

        self.psr = psrs
        self.grid = grid
        self.den = den
        self.noise_dict = noise_dict
        self.Npulsars = len(self.psr)
        self.crn_bins = crn_bins
        self.int_bins = int_bins
        self.kmax = 2 * self.int_bins
        self.fref = 1/(1 * 365.25 * 24 * 60 * 60)
        self.int_rn = int_rn
        self.gw_psd_model = gw_psd_model
        self.int_psd_model = int_psd_model

        if Tspan:
            self.Tspan = Tspan
        else:
            self.Tspan = model_utils.get_tspan(self.psr)

        self.f_intrin = np.arange(1/self.Tspan, (self.int_bins + 0.01)/self.Tspan, 1/self.Tspan)
        self.f_common = self.f_intrin[:crn_bins]

        self.freqs = np.arange(1/self.Tspan, (self.crn_bins + .01)/self.Tspan, 1/self.Tspan)
        self.fref = 1/(1 * 365.25 * 24 * 60 * 60)
        
        self.diag_idx = np.arange(0, self.Npulsars, 1, int)
        self.c_idx = np.arange(0, self.crn_bins, 1, int)
        self.k_idx = np.arange(0, self.kmax, 1, int)
        self.ppair_idx = np.arange(0, int(self.Npulsars * (self.Npulsars - 1) * 0.5), 1, int)

        self.u = np.ones((self.crn_bins, self.Npulsars), dtype = int)
        self.s = np.ones((self.crn_bins, self.Npulsars), dtype = int)
        for pidx in range(self.Npulsars):
            self.u[:, pidx] = range(self.crn_bins)
        for fidx in range(self.crn_bins):
            self.s[fidx, :] = range(self.Npulsars)

        ###Make a PTA Object
        tm = gp_signals.MarginalizingTimingModel(use_svd=True)
        wn = blocks.white_noise_block(vary=False, inc_ecorr=inc_ecorr, gp_ecorr=False, select = backend, tnequad = tnequad)
        rn = blocks.red_noise_block(psd=self.int_psd_model, prior='log-uniform',Tspan = self.Tspan,
                                                components=int_bins, gamma_val=None)
        crn = blocks.common_red_noise_block(psd=self.gw_psd_model, prior='log-uniform', Tspan = self.Tspan, orf = 'crn',
                                                components=self.crn_bins, gamma_val=None, name = 'gw')
        if int_rn:
            s = tm + wn + rn + crn
        else:
            s = tm + wn + crn

        self.pta = signal_base.PTA([s(p) for p in self.psr])
        self.pta.set_default_params(self.noise_dict)
        self.x0 = np.hstack([p.sample() for p in self.pta.params])

        self.pmin = []
        self.pmax = []
        for p in self.pta.params:
            sp = str(p)
            val_min = float(sp.split('pmin=')[1].split(',')[0])
            val_max = float(sp.split('pmax=')[1].split(')')[0])
            # if not 'rho' in sp:
            if not ')[' in sp:
                self.pmin.append(val_min)
                self.pmax.append(val_max)
            else:
                nf_bins = int(sp.split(')[')[1].split(']')[0])
                for _ in range(nf_bins):
                    self.pmin.append(val_min)
                    self.pmax.append(val_max) 

    def pl_to_rho(self, log10amp, gamma):
        return 10**(2*log10amp)/(12 * np.pi**2 * self.freqs[:, None]**3 * self.Tspan) * (self.freqs[:, None]/self.fref)**(3-gamma)
    
    def get_phi_mat(self, xs):

        end_idx = 2 * self.Npulsars
        phi = np.zeros((self.crn_bins, self.Npulsars))

        psd_intrin = self.pl_to_rho(xs[1:end_idx:2], xs[0:end_idx:2])
        psd_common = self.pl_to_rho(xs[end_idx + 1], xs[end_idx])
       
        phi[:, self.diag_idx] = psd_intrin + psd_common

        return phi

    def get_lnprior(self, xs):
        if np.any(xs < self.pmin) or np.any(xs > self.pmax):
            return -np.inf
        else:
            return -8.01

    def get_lnliklihood(self, xs):
        '''
        KDE-liklihood
        '''
        phi = self.get_phi_mat(xs)
        idxs = np.digitize(0.5 * np.log10(phi), bins = self.grid) - 1
        return self.den[self.s, self.u, idxs].sum()

    def sample(self, niter, savepath, SCAMweight=30, 
               AMweight=15, DEweight=50, 
               covUpdate = 1000):

        x0 = self.x0
        ndim = len(x0)
        cov = np.diag(np.ones(ndim) * 0.01**2) # helps to tune MCMC proposal distribution
        pars = self.pta.param_names
        groups = [list(np.arange(0, ndim))]

        self.sampler_CURN = ptmcmc(ndim, self.get_lnliklihood, self.get_lnprior, cov, groups = groups,
                                outDir = savepath,
                                resume=False)
        jp = samp.JumpProposal(self.pta)
            
        self.sampler_CURN.addProposalToCycle(jp.draw_from_prior, 5)
        self.sampler_CURN.addProposalToCycle(jp.draw_from_par_prior(
                par_names=[str(p).split(':')[0] for
                           p in list(self.pta.params)
                           if 'gw' in str(p)]), 10)
        if self.int_rn:
            self.sampler_CURN.addProposalToCycle(jp.draw_from_red_prior, 10)

        # self.sampler_CURN.initialize(Niter = niter, SCAMweight=SCAMweight,
        #                             AMweight=AMweight, DEweight=DEweight, covUpdate = covUpdate)
        
        self.sampler_CURN.sample(self.x0, niter, SCAMweight=30, AMweight=15, DEweight=50, )

class BayesPowerSingle(object):

    """
    The Gibbs Method class used for single-pulsar noise analyses.

    Based on:

        Article by van Haasteren & Vallisneri (2014),
        "New advances in the Gaussian-process approach
        to pulsar-timing data analysis",
        Physical Review D, Volume 90, Issue 10, id.104012
        arXiv:1407.1838

        Initial structure of the code is based on https://github.com/jellis18/gibbs_student_t

    Authors:

        S. R. Taylor
        N. Laal
    """

    def __init__(
        self,
        psr=None,
        Tspan=None,
        select="backend",
        white_vary=False,
        inc_ecorr=False,
        ecorr_type="kernel",
        noise_dict=None,
        tm_marg=False,
        freq_bins=None,
        tnequad=True,
        log10rhomin=-9.0,
        log10rhomax=-1.0,
    ):
        """
        Parameters
        -----------

        psr : object
            instance of an ENTERPRISE psr object for a single pulsar

        Tspan: float (optional)
            if given, the baseline of the pulsar is fixed to the input value. If not,
            baseline is determined inetrnally

        select: str
            the selection of backend ('backend' or 'none') for the white-noise parameters

        white_vary: bool
            whether to vary the white noise

        inc_ecorr: bool
            whether to include ecorr

        ecorr_type: str
            the type of ecorr to use. Choose between 'basis' or 'kernel'

        noise_dict: dict
            white noise dictionary in case 'white_vary' is set to False

        tm_marg: bool
            whether to marginalize over timing model parameters (do not use this if you are varying the white noise!)

        freq_bins: int
            number of frequency bins for the red noise process

        log10rhomin: float
            lower bound for the log10 of the rho parameter.

        log10rhomax: float
            upper bound for the log10 of the rho parameter

        tnequad: string
            whether to use the temponest convension of efac and equad
        """

        self.psr = [psr]
        if Tspan:
            self.Tspan = Tspan
        else:
            self.Tspan = model_utils.get_tspan(self.psr)
        self.name = self.psr[0].name
        self.inc_ecorr = inc_ecorr
        self.ecorr_type = ecorr_type
        self.white_vary = white_vary
        self.tm_marg = tm_marg
        self.wn_names = ["efac", "equad", "ecorr"]
        self.rhomin = log10rhomin
        self.rhomax = log10rhomax
        self.freq_bins = freq_bins
        self.low = 10 ** (2 * self.rhomin)
        self.high = 10 ** (2 * self.rhomax)

        # Making the pta object
        if self.tm_marg:
            tm = gp_signals.MarginalizingTimingModel(use_svd=True)
            if self.white_vary:
                warnings.warn(
                    "***FYI: the timing model is marginalized for. This will slow down the WN sampling!!***"
                )
        else:
            tm = gp_signals.TimingModel(use_svd=True)

        if self.ecorr_type == "basis":
            wn = blocks.white_noise_block(
                vary=self.white_vary,
                inc_ecorr=self.inc_ecorr,
                gp_ecorr=True,
                select=select,
                tnequad=tnequad,
            )
        else:
            wn = blocks.white_noise_block(
                vary=self.white_vary,
                inc_ecorr=self.inc_ecorr,
                gp_ecorr=False,
                select=select,
                tnequad=tnequad,
            )

        rn = blocks.common_red_noise_block(
            psd="spectrum",
            prior="log-uniform",
            Tspan=self.Tspan,
            logmin=self.rhomin,
            logmax=self.rhomax,
            components=freq_bins,
            gamma_val=None,
            name="gw",
        )
        s = tm + wn + rn
        self.pta = signal_base.PTA(
            [s(p) for p in self.psr],
            lnlikelihood=signal_base.LogLikelihoodDenseCholesky,
        )
        if not white_vary:
            self.pta.set_default_params(noise_dict)
            self.Nmat = self.pta.get_ndiag(params={})[0]
            self.TNr = self.pta.get_TNr(params={})[0]
            self.TNT = self.pta.get_TNT(params={})[0]
        else:
            self.Nmat = None

        if self.inc_ecorr and "basis" in self.ecorr_type:
            # grabbing priors on ECORR params
            for ct, par in enumerate(self.pta.params):
                if "ecorr" in str(par):
                    ind = ct
            ecorr_priors = str(self.pta.params[ind].params[0])
            ecorr_priors = ecorr_priors.split("(")[1].split(")")[0].split(", ")
            self.ecorrmin, self.ecorrmax = (
                10 ** (2 * float(ecorr_priors[0].split("=")[1])),
                10 ** (2 * float(ecorr_priors[1].split("=")[1])),
            )

        # Getting residuals
        self._residuals = self.pta.get_residuals()[0]
        # Intial guess for the model params
        self._xs = np.array([p.sample()
                            for p in self.pta.params], dtype=object)
        # Initializign the b-coefficients. The shape is 2*freq_bins if tm_marg
        # = True.
        self._b = np.zeros(self.pta.get_basis(self._xs)[0].shape[1])
        self.Tmat = self.pta.get_basis(params={})[0]
        self.phiinv = None

        # find basis indices of GW process
        self.gwid = []
        ct = 0
        psigs = [sig for sig in self.pta.signals.keys() if self.name in sig]
        for sig in psigs:
            Fmat = self.pta.signals[sig].get_basis()
            if "gw" in self.pta.signals[sig].name:
                self.gwid.append(ct + np.arange(0, Fmat.shape[1]))
            # Avoid None-basis processes.
            # Also assume red + GW signals share basis.
            if Fmat is not None and "red" not in sig:
                ct += Fmat.shape[1]

    @cached_property
    def params(self):
        return self.pta.params

    @cached_property
    def param_names(self):
        return self.pta.param_names

    def map_params(self, xs):
        return self.pta.map_params(xs)

    @cached_property
    def get_red_param_indices(self):
        ind = []
        for ct, par in enumerate(self.param_names):
            if "log10_A" in par or "gamma" in par or "rho" in par:
                ind.append(ct)
        return np.array(ind)

    @cached_property
    def get_efacequad_indices(self):
        ind = []
        if "basis" in self.ecorr_type:
            for ct, par in enumerate(self.param_names):
                if "efac" in par or "equad" in par:
                    ind.append(ct)
        else:
            for ct, par in enumerate(self.param_names):
                if "ecorr" in par or "efac" in par or "equad" in par:
                    ind.append(ct)
        return np.array(ind)

    @cached_property
    def get_basis_ecorr_indices(self):
        ind = []
        for ct, par in enumerate(self.param_names):
            if "ecorr" in par:
                ind.append(ct)
        return np.array(ind)

    def update_red_params(self, xs):
        """
        Function to perform log10_rho updates given the Fourier coefficients.
        """
        tau = self._b[tuple(self.gwid)] ** 2
        tau = (tau[0::2] + tau[1::2]) / 2

        Norm = 1 / (np.exp(-tau / self.high) - np.exp(-tau / self.low))
        x = np.random.default_rng().uniform(0, 1, size=tau.shape)
        rhonew = -tau / np.log(x / Norm + np.exp(-tau / self.low))
        xs[-1] = 0.5 * np.log10(rhonew)
        return xs

    def update_b(self, xs):
        """
        Function to perform updates on Fourier coefficients given other model parameters.
        """
        params = self.pta.map_params(np.hstack(xs))
        self._phiinv = self.pta.get_phiinv(params, logdet=False)[0]

        try:
            TNT = self.TNT.copy()
        except BaseException:
            T = self.Tmat
            TNT = self.Nmat.solve(T, left_array=T)
        try:
            TNr = self.TNr.copy()
        except BaseException:
            T = self.Tmat
            TNr = self.Nmat.solve(self._residuals, left_array=T)

        np.fill_diagonal(TNT, TNT.diagonal() + self._phiinv)
        try:
            chol = cho_factor(
                TNT,
                lower=True,
                overwrite_a=False,
                check_finite=False)
            mean = cho_solve(
                chol,
                b=TNr,
                overwrite_b=False,
                check_finite=False)
            self._b = mean + st_solve(
                chol[0],
                np.random.normal(loc=0, scale=1, size=TNT.shape[0]),
                lower=True,
                unit_diagonal=False,
                overwrite_b=False,
                check_finite=False,
                trans=1,
            )
        except np.linalg.LinAlgError:
            print('Single-pulsar Cholesky Failed.')
            bchain = np.memmap(
                self._savepath + f'/{self.psr[0].name}.npy',
                dtype="float32",
                mode="r",
                shape=(self.niter, self.len_x + self.len_b),
            )[:, -len(self._b):]
            self._b = bchain[np.random.default_rng().integers(
                0, len(bchain))]

    def update_white_params(self, xs, iters=10):
        """
        Function to perform WN updates given other model parameters.
        If kernel ecorr is chosen, WN includes ecorr as well.
        """
        # get white noise parameter indices
        wind = self.get_efacequad_indices
        xnew = xs
        x0 = xnew[wind].copy()
        lnlike0, lnprior0 = self.get_lnlikelihood_white(
            x0), self.get_wn_lnprior(x0)
        lnprob0 = lnlike0 + lnprior0

        for ii in range(
                self.start_wn_iter + 1,
                self.start_wn_iter + iters + 1):
            x0, lnlike0, lnprob0 = self.sampler_wn.PTMCMCOneStep(
                x0, lnlike0, lnprob0, ii
            )
        xnew[wind] = x0
        self.start_wn_iter = ii

        # Do some caching of "later needed" parameters for improved performance
        self.Nmat = self.pta.get_ndiag(self.map_params(xnew))[0]
        Tmat = self.Tmat
        if "basis" not in self.ecorr_type:
            self.TNT = self.Nmat.solve(Tmat, left_array=Tmat)
        else:
            TN = Tmat / self.Nmat[:, None]
            self.TNT = Tmat.T @ TN
            residuals = self._residuals
            self.rNr = np.sum(residuals**2 / self.Nmat)
            self.logdet_N = np.sum(np.log(self.Nmat))
            self.d = TN.T @ residuals

        return xnew

    def update_basis_ecorr_params(self, xs, iters=10):
        """
        Function to perform basis ecorr updates.
        """
        # get white noise parameter indices
        eind = self.get_basis_ecorr_indices
        xnew = xs
        x0 = xnew[eind].copy()
        lnlike0, lnprior0 = self.get_basis_ecorr_lnlikelihood(
            x0
        ), self.get_basis_ecorr_lnprior(x0)
        lnprob0 = lnlike0 + lnprior0

        for ii in range(
                self.start_ec_iter + 1,
                self.start_ec_iter + iters + 1):
            x0, lnlike0, lnprob0 = self.sampler_ec.PTMCMCOneStep(
                x0, lnlike0, lnprob0, ii
            )
        xnew[eind] = x0
        self.start_ec_iter = ii

        return xnew

    def get_lnlikelihood_white(self, xs):
        """
        Function to calculate WN log-liklihood.
        """
        x0 = self._xs.copy()
        x0[self.get_efacequad_indices] = xs

        params = self.map_params(x0)
        Nmat = self.pta.get_ndiag(params)[0]
        # whitened residuals
        yred = self._residuals - self.Tmat @ self._b
        try:
            if "basis" not in self.ecorr_type:
                rNr, logdet_N = Nmat.solve(yred, left_array=yred, logdet=True)
            else:
                rNr = np.sum(yred**2 / Nmat)
                logdet_N = np.sum(np.log(Nmat))
        except BaseException:
            return -np.inf
        # first component of likelihood function
        loglike = -0.5 * (logdet_N + rNr)

        return loglike

    def get_basis_ecorr_lnlikelihood(self, xs):
        """
        Function to calculate basis ecorr log-liklihood.
        """
        x0 = np.hstack(self._xs.copy())
        x0[self.get_basis_ecorr_indices] = xs

        params = self.map_params(x0)
        # start likelihood calculations
        loglike = 0
        # get auxiliaries
        phiinv, logdet_phi = self.pta.get_phiinv(params, logdet=True)[0]
        # first component of likelihood function
        loglike += -0.5 * (self.logdet_N + self.rNr)
        # Red noise piece
        Sigma = self.TNT + np.diag(phiinv)
        try:
            cf = sl.cho_factor(Sigma)
            expval = sl.cho_solve(cf, self.d)
        except np.linalg.LinAlgError:
            return -np.inf

        logdet_sigma = np.sum(2 * np.log(np.diag(cf[0])))
        loglike += 0.5 * (self.d @ expval - logdet_sigma - logdet_phi)

        return loglike

    def get_wn_lnprior(self, xs):
        """
        Function to calculate WN log-prior.
        """
        x0 = self._xs.copy()
        x0[self.get_efacequad_indices] = xs

        return np.sum([p.get_logpdf(value=x0[ct])
                      for ct, p in enumerate(self.params)])

    def get_basis_ecorr_lnprior(self, xs):
        """
        Function to calculate basis ecorr log-prior.
        """
        x0 = self._xs.copy()
        x0[self.get_basis_ecorr_indices] = xs

        return np.sum([p.get_logpdf(value=x0[ct])
                      for ct, p in enumerate(self.params)])

    def sample(
        self,
        niter=int(1e4),
        resume = True,
        wniters=30,
        eciters=10,
        savepath=None,
        SCAMweight=30,
        AMweight=15,
        DEweight=50,
        covUpdate=1000,
        burn=10000,
        **kwargs
    ):
        """
        Gibbs Sampling

        Parameters
        -----------
        niter: integer
            total number of Gibbs sampling iterations

        wniters:
            number of white noise MCMC sampling iterations within each Gibbs step

        eciters:
            number of basis ecorr MCMC sampling iterations within each Gibbs step

        savepath: str
            the path to save the chains

        covUpdate: integer
            Number of iterations between AM covariance updates

        SCAMweight: integer
            Weight of SCAM jumps in overall jump cycle

        AMweight: integer
            Weight of AM jumps in overall jump cycle

        DEweight: integer
            Weight of DE jumps in overall jump cycle

        kwargs: dict
            PTMCMC initialization settings not mentioned above
        """
        self.start_wn_iter = 0
        self.start_ec_iter = 0

        os.makedirs(savepath, exist_ok=True)

        if self.white_vary:
            # large number to avoid saving the white noise choice in a txt file
            isave = int(4e9)
            thin = 1
            Niter = int(niter * wniters + 1)

            x0 = self._xs[self.get_efacequad_indices]
            ndim = len(x0)
            cov = np.diag(
                np.ones(ndim) * 0.01**2
            )  # helps to tune MCMC proposal distribution
            self.sampler_wn = ptmcmc(
                ndim,
                self.get_lnlikelihood_white,
                self.get_wn_lnprior,
                cov,
                outDir=savepath,
                resume=False,
            )
            self.sampler_wn.initialize(
                Niter=Niter,
                isave=isave,
                thin=thin,
                SCAMweight=SCAMweight,
                AMweight=AMweight,
                DEweight=DEweight,
                covUpdate=covUpdate,
                burn=burn,
                **kwargs
            )

            if "basis" in self.ecorr_type and self.white_vary:
                x0 = self._xs[self.get_basis_ecorr_indices]
                ndim = len(x0)
                cov = np.diag(np.ones(ndim) * 0.01**2)
                self.sampler_ec = ptmcmc(
                    ndim,
                    self.get_basis_ecorr_lnlikelihood,
                    self.get_basis_ecorr_lnprior,
                    cov,
                    outDir=savepath,
                    resume=False,
                )
                self.sampler_ec.initialize(
                    Niter=Niter,
                    isave=isave,
                    thin=thin,
                    SCAMweight=SCAMweight,
                    AMweight=AMweight,
                    DEweight=DEweight,
                    covUpdate=covUpdate,
                    burn=burn,
                    **kwargs
                )

        np.savetxt(savepath + "/pars.txt",
                   list(map(str, self.pta.param_names)), fmt="%s")
        np.savetxt(
            savepath + "/priors.txt",
            list(map(lambda x: str(x.__repr__()), self.pta.params)),
            fmt="%s",
        )
        freqs = np.arange(
            1 / self.Tspan,
            (self.freq_bins + 0.001) / self.Tspan,
            1 / self.Tspan)
        np.save(savepath + "/freqs.npy", freqs)
        [os.remove(dpa) for dpa in glob.glob(savepath + "/*jump.txt")]

        xnew = self._xs.copy()

        len_b = len(self._b)
        len_x = len(np.hstack(self._xs))
        self._savepath = savepath

        if resume and os.path.isfile(savepath + f'/{self.psr[0].name}.npy'):
                
            fp = np.lib.format.open_memmap(
                savepath + f'/{self.psr[0].name}.npy',
                mode="r+",
                dtype="float64",
                shape=(niter, len_x + len_b),
                fortran_order=False,
            )
            start_idx = np.nonzero(fp[:, 0])[0].max()
            if niter <= start_idx:
                print('The run is already done. Exitting...')
                return None
        else:
            fp = np.lib.format.open_memmap(
                savepath + f'/{self.psr[0].name}.npy',
                mode="w+",
                dtype="float64",
                shape=(niter, len_x + len_b),
                fortran_order=False,
            )
            start_idx = 0

        pbar = tqdm(range(niter), colour="GREEN")
        pbar.set_description("Sampling %s" % self.name)
        # st = time.time()
        # pbar_freq = int(5e3)
        # pbar = range(start_idx, niter)
        for ii in pbar:
            
            # if not ii%pbar_freq and ii:
            #     print(f'{round(ii/niter * 100, 2)} Percent Done in {round((time.time() - st)/60, 2)} Minutes.\n')
                
            if self.white_vary:
                xnew = self.update_white_params(xnew, iters=wniters)

            if self.inc_ecorr and "basis" in self.ecorr_type:
                xnew = self.update_basis_ecorr_params(xnew, iters=eciters)

            self.update_b(xs=xnew)
            xnew = self.update_red_params(xs=xnew)

            fp[ii, -len_b:] = self._b
            fp[ii, 0:len_x] = np.hstack(xnew)