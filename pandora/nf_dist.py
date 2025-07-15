import torch
import torch.utils.dlpack as torchdlpack
import zuko
import numpy as np
import os
import torch
from tqdm import tqdm
from tqdm.auto import trange
import random
import jax
import jax.numpy as jnp
torch.set_default_dtype(torch.float64)
########################Plotting Settings#########################
import matplotlib.pyplot as plt
hist_settings = dict(
    bins = 40,
    histtype = 'step',
    lw = 3,
    density = True
)

class NFastroinference(object):
    """
    A wrapper for pyro's spline flow to make its use more
    user-friendly for astro inference purposes. The conversion
    from the normalized domain to the real domain or from the 
    real domain to the normalized domain is handeled by this wrapper.
    Arrays are converted from numpy to Pytorch in this wrapper.

    :param: nf_object
        zuko's flow object
    
    :param: scale
        The positive scale where all samples are normalized to. It is 
        denoted by `B` in some scripts. The normalized interval
        will be [-B, B].
    
    :param: nf_type
        The type of flow. One of ['theta->rho', 'rho->theta', `rho].
        'theta|rho' means theta conditioned on rho, 'rho|theta' means
        rho conditioned on theta, and `rho` means unconditional flow.

    :param: mean
        The mean of the samples used for normalizing the context and the input
        distributions of the flow.

    :param: half_range
        The half_range of the samples (i.e., (max - min)/2) used for normalizing 
        the context and the input distributions of the flow.  

    :param: rho_idxs
        The indices that orders the GWB spectrum. For example, mean[rho_idxs]
        selects the relevant values of `mean` that corresponds to the GWB
        spectrumn parameter rho.

    :param: ast_param_idxs
        The indices that orders the astro params. For example, mean[ast_param_idxs]
        selects the relevant values of `mean` that corresponds to the astro
        parameters.

    Documentation pending...
    """

    def __init__(
        self, 
        nf_object, 
        nf_type, 
        mean, 
        half_range, 
        scale, 
        rho_idxs, 
        ast_param_idxs,
        nf_object_device = 'cuda',
    ):
        self.nf = nf_object
        self.nf_object_device = nf_object_device,
        self.B = scale
        self.mean_gwb = mean[rho_idxs]
        self.mean_ast = mean[ast_param_idxs]
        self.half_range_gwb = half_range[rho_idxs]
        self.half_range_ast = half_range[ast_param_idxs]
        self.nf_type = nf_type
        if self.nf_type not in ['theta|rho', 'rho|theta', 'rho']:
            raise ValueError("`nf_type` must be one of ['theta|rho', 'rho|theta', 'rho'].")
        
    def convert_torch_to_numpy(self, torch_tensor):
        return torch_tensor.detach().cpu().numpy()

    def convert_numpy_to_torch(self, numpy_array):
        if self.nf_object_device == 'cpu':
            return torchdlpack.from_dlpack(numpy_array.__dlpack__())
        else:
            return torch.from_numpy(numpy_array).to('cuda', non_blocking=True)

    def transform_rho_to_scaled_interval(self, gwb_rho):
        return self.B * (gwb_rho - self.mean_gwb) / self.half_range_gwb

    def transform_params_to_scaled_interval(self, ast_params):
        return self.B * (ast_params - self.mean_ast) / self.half_range_ast

    def transform_rho_to_physical_interval(self, gwb_rho):
        return gwb_rho * self.half_range_gwb / self.B + self.mean_gwb

    def transform_params_to_physical_interval(self, ast_params):
        return ast_params * self.half_range_ast / self.B + self.mean_ast

    def log_prob(self, gwb_rho, astro_params):
        scaled_gwb = self.convert_numpy_to_torch(
            self.transform_rho_to_scaled_interval(gwb_rho)
        )
        if self.nf_type == 'rho|theta':
            scaled_ast = self.convert_numpy_to_torch(
                self.transform_params_to_scaled_interval(astro_params)
            )
            return self.convert_torch_to_numpy(
                self.nf(scaled_ast).log_prob(scaled_gwb)
        )
        elif self.nf_type == 'theta|rho':
            scaled_ast = self.convert_numpy_to_torch(
                self.transform_params_to_scaled_interval(astro_params)
            )
            return self.convert_torch_to_numpy(
                self.nf(scaled_gwb).log_prob(scaled_ast)
        )
        elif self.nf_type == 'rho':
            return self.convert_torch_to_numpy(
                self.nf.log_prob(scaled_gwb)
        )           

    def sample(self, batch_shape, context_params):
        if self.nf_type == 'rho|theta':
            scaled_ast = self.convert_numpy_to_torch(
                self.transform_params_to_scaled_interval(context_params)
            )
            scaled_gwb = self.convert_torch_to_numpy(
                self.nf(scaled_ast).sample(batch_shape)
            )
            return self.transform_rho_to_physical_interval(scaled_gwb)
        
        elif self.nf_type == 'theta|rho':
            scaled_gwb = self.convert_numpy_to_torch(
                self.transform_rho_to_scaled_interval(context_params)
            )
            scaled_ast = self.convert_torch_to_numpy(
                self.nf(scaled_gwb).sample(batch_shape)
            )
            return self.transform_params_to_physical_interval(scaled_ast)
        
        elif self.nf_type == 'rho':
            scaled_gwb = self.convert_torch_to_numpy(
                self.nf().sample(batch_shape)
            )
            return self.transform_rho_to_physical_interval(scaled_gwb)

class DataSplitter(object):
    """
    To split a set of samples between validation and training.
    """

    def splitter_mesh(samples, validation_sample_size, seeds):
        '''
        :param: samples
            The samples used for validation and training
        
        :param: validation_sample_size
            The number of samples per each distribution axis of `samples`
            that needs to be considered as validation.
            Provide a numpy array with length equal to the 
            number of distribution axes of `samples`. For example,
            if samples are form holodeck and are GWB spectrum,
            the `samples` has the shape (n_pop_draws, n_spectrum_draws, n_freqs).
            In this case, the `validation_sample_size` can be np.array([100, 200]).
            This means that 100 random samples from the first axis and 200
            from the second axis is being splitted from `samples` to make 
            the validation set.

        :param: seeds
            The RNG seeds. It must be an iterator of length 2.
        ''' 
        ndim = samples.ndim

        ## Looking at the distribution axes only
        total_sample_size = np.array(samples.shape[:-1], dtype = int)
        training_sample_size = total_sample_size - validation_sample_size

        ## Randomly selecting indices
        random.seed(seeds[0])
        v0 = random.sample(range(total_sample_size[0]), k = validation_sample_size[0])
        bools0 = np.ones(samples.shape[0], dtype = bool)
        bools0[v0] = False

        if ndim == 3:
            random.seed(seeds[1])
            v1 = random.sample(range(total_sample_size[1]), k = validation_sample_size[1])
            bools1 = np.ones(samples.shape[1], dtype = bool)
            bools1[v1] = False

            validation_set = samples[np.ix_(v0, v1)]
            training_set = samples[np.ix_(bools0, bools1)]

        elif ndim == 2:
            validation_set = samples[v0]
            training_set = samples[bools0]

        else:
            raise ValueError('The `samples` must be 2D or 3D')

        return training_set, validation_set

    def splitter(samples, validation_sample_size, seed):
        '''
        :param: samples
            The samples used for validation and training
        
        :param: validation_sample_size
            The number of samples to take as validation.
            This must be an integer.

        :param: seed
            The RNG seeds.
        ''' 
        ndim = samples.ndim

        ## Looking at the distribution axes only
        total_sample_size = samples.shape[0]
        training_sample_size = total_sample_size - validation_sample_size

        ## Randomly selecting indices
        random.seed(seed)
        v0 = random.sample(range(total_sample_size), k = validation_sample_size)
        bools0 = np.ones(total_sample_size, dtype = bool)
        bools0[v0] = False

        validation_set = samples[v0]
        training_set = samples[bools0]

        return training_set, validation_set

class ValidationHell(object):
    """
    Flow Training Early Stopage Decision Based on Hellinger Distance
    Estimation.

    :param: plot_save_path
        The path to save a pdf version of the hellinger distance distributin
        plot.

    :param: device
        The hardware to use to sample from the flow object.

    :param: q
        The quantile used to profile a hellinger distance distribution.
        The lower and upper q quantile of a hellinger distribution is
        used to make decision on whether more flow training improves
        the quality of emulation.

    :param: is_conditional
        The is flow conditional?

    """
    def __init__(
        self,
        plot_save_path,
        device,
        q = .158,
        is_conditional = True,
        ):
        self.plot_save_path = plot_save_path
        self.device = device
        self.is_conditional = is_conditional
        self.q = q
        self.ll = []
        self.ul = []
        self.hell = []
        self.counter = 0

    def histogram_1d(self, 
                    samples, 
                    bins, 
                    lower_bound, 
                    upper_bound):
            '''
            A wrapper for jnp.histogram.
            '''
            return jnp.histogram(samples, bins = bins, range=(lower_bound, upper_bound))[0]

    def batched_histogram(self, 
                          batched_samples,
                          lower_bounds,
                          upper_bounds,
                          bins = 15):
        '''
        :param: batched_samples
            Samples to make histogram out of. The shape must be (batchs, features)

        :param: lower_bounds
            The lower bound for the samples. The shape must be (bathces)

        :param: upper_bounds
            The upper bound for the samples. The shape must be (bathces)
        
        :param: bins
            The number of bins used for making histograms. It must be an int
        '''
        vmapped_histogram = jax.vmap(self.histogram_1d, in_axes=(0, None, 0, 0))
        return vmapped_histogram(batched_samples, bins, lower_bounds, upper_bounds)

    def dohell(self, hist1, hist2):
        '''
        :param: hist1
            One of the two histograms used for Hellinger distance computation.
            The shape must be (batches, bins).

        :param: hist2
            The other of the two histograms used for Hellinger distance computation.
            The shape must be (batches, bins).  
        '''   
        p = hist1/hist1.sum(axis = -1)[:, None]
        q = hist2/hist2.sum(axis = -1)[:, None]
        diff = jnp.sqrt(p) - jnp.sqrt(q)
        return 1/jnp.sqrt(2) * jnp.sqrt(jnp.sum(diff**2, axis = -1))

    def sample_from_nf(self, 
                        ndraws, 
                        input_dim, 
                        context_samples,
                        nf_save_path,
                        progress_bar):
        '''
        :param: ndraws
            The number of sample draws from the flow object.

        :param: input_dim
            The number of features of the generated distribution.
            For example, if the flow is generating GWB spectrum,
            `input_dim` must match to the number of frequency bins
            considered.

        :param: context_samples
            Context samples if the flow is a conditional flow

        :param: nf_save_path
             The path to load a pickle file containing the flow object.

        :param: progress_bar
            Do you want tqdm progress bar?
        '''
        nf, *_ = torch.load(nf_save_path, map_location = self.device,  weights_only = False,)
        if self.is_conditional:
            gen_samples = jnp.zeros((context_samples.shape[0], input_dim, ndraws))

            if progress_bar:
                pbar = trange(context_samples.shape[0], colour="green", desc = "Validation: Sampling from NF")
            else:
                pbar = range(context_samples.shape[0])

            for ii in pbar:
                gen_samples = gen_samples.at[ii].set(jnp.array(nf(context_samples[ii]).sample((ndraws, )).T))
            return gen_samples.reshape(context_samples.shape[0] * input_dim, ndraws)
        else:
            return jnp.array(nf().sample((ndraws, )).T)
            
    def plot_hell_dist(self):
        '''
        Plots the hellinger distance distribution and saves the output to a pdf file
        '''
        for ct, hell in enumerate(self.hell):
            plt.hist(hell, range = (0, 1), **hist_settings, label = f'Checkpoint {ct+1}; ll = {self.ll[ct]}; ul = {self.ul[ct]}')
            ct+=1
        plt.legend()
        plt.ylabel('Count')
        plt.xlabel('Hellinger Distances')
        plt.tight_layout()
        plt.savefig(self.plot_save_path)
        plt.close()

    def make_decision(self,
                      validation_histograms,
                      validation_context_samples,
                      ndraws,
                      input_dim,
                      lower_bounds,
                      upper_bounds,
                      nf_save_path,
                      threshold = .02,
                      progress_bar = True):
        '''
        :param: validation_histograms
            The pre-computed histograms from the validation set

        :param: validation_context_samples
            The validation samples for the context params. The shape must be
            (batches, features)

        :param: ndraws
            The number of sample draws from the flow object.

        :param: input_dim
            The number of features of the generated distribution.
            For example, if the flow is generating GWB spectrum,
            `input_dim` must match to the number of frequency bins
            considered.

        :param: lower_bounds
            The lower_bounds of the histograms. The shape must be 
            broadcastable to the number of batches.

        :param: upper_bounds
            The upper_bounds of the histograms. The shape must be 
            broadcastable to the number of batches.

        :param: threshold
            The threshold under which two hellinger distance distributions
            are deemed close enough. The threshold is for the lower and upper
            `q` quantile of the two hellinger distributions.

        :param: nf_save_path
            The path to a flow pickle file.

        :param: progress_bar
            Do you want tqdm progress bar?
        '''
        ## First, sample from the flow
        gen_samples = self.sample_from_nf(ndraws, 
                                          input_dim, 
                                          validation_context_samples,
                                          nf_save_path,
                                          progress_bar = progress_bar)
        
        ## Second, turn the samples into histograms
        hs = self.batched_histogram(gen_samples,
                          lower_bounds,
                          upper_bounds,
                          bins = validation_histograms.shape[-1])

        ## Third, estimate the hellinger distances
        new_hell = np.array(self.dohell(validation_histograms, hs))
        self.hell.append(new_hell)

        ## Fourth, estimate the 1-sigma level of the hellinger distances
        self.ll.append(np.round(np.quantile(new_hell, q = self.q), 2))
        self.ul.append(np.round(np.quantile(new_hell, q = 1 - self.q), 2))

        ## Fifth, plot the distances
        self.plot_hell_dist()

        if self.counter > 1:
            ## Sixth, make a decision!
            cond1 = np.abs(self.ll[-1] - self.ll[-2]) < threshold
            cond2 = np.abs(self.ul[-1] - self.ul[-2]) < threshold
            self.counter+=1
            return cond1 and cond2
        else:
            self.counter+=1
            
class NFMaker(object):
    """
    To construct and train an AQRQS normalizing flow

    :param: to_be_learned_dist
        Samples from the distribution whose pdf needs to be learned
        the shape must be (n_samples, n_features) or (n_samples, n_features, n_features).

    :param: context_dist
        Samples from the distribution that will be used as the context for the
        `to_be_learned_dist` distribution.
        the shape must be (n_samples, n_features) or (n_samples, n_features, n_features)
        If you want to context distribution, set this to np.array([False]) or
        torch.tensor([False]).

    :param: nf_save_dir
        The path to which the flow object will be saved
    
    :param: B
        The positive scale where all samples are normalized to. The normalized interval
        will be [-B, B]. Ignored if the distributions are both normalized already

    :param: device
        The hardware to run the trianing on.

    :param: spline_bins
        The number of bins used for constructing the spline flow. Keep it 16
        if you do not know what you are doing!

    :param: hidden_dims = [512] * 2
        The number of hidden dimensions and the neurons used in the AR part of ARQS
        Keep the default values if yu do not know what you are doing!

    :param: start_from_loaded_nf
        Do you wish to train an already constructed spline flow?

    :param: path_to_load_nf
        Path to the spline flow object you want to load.
    """

    def __init__(
        self, 
        to_be_learned_dist,
        context_dist,
        nf_save_dir,
        normalized_interval_scale,
        device = torch.device("cuda"),
        spline_bins = 8,
        hidden_dims = [512] * 2,
        start_from_loaded_nf = False,
        path_to_load_nf = None
        ):
        ## Check to see if you have access to a GPU
        if torch.cuda.is_available():
            print("GPU is available.")
        else:
            print("GPU is not available.")
        self.device = device
        self.nf_save_dir = nf_save_dir
        self.B = normalized_interval_scale

        ## Are we dealing with a simple 2D array
        ## or a 3D array like the output of holodeck
        self.input_axes = to_be_learned_dist.ndim
        self.context_axes = context_dist.ndim

        self.chain_context = context_dist
        self.chain_input = to_be_learned_dist  

        ## Do you want a conditional flow?
        self.context_exist = context_dist.any()

        ## Caching the trianing data and its features
        self.input_dim = to_be_learned_dist.shape[-1]
        if self.context_exist:
            self.context_dim = context_dist.shape[-1]   
        else:
            self.context_dim = 0

        ## Things must be Pytorch compatible!
        self.chain_input = torch.tensor(self.chain_input, device = self.device)
        self.chain_context = torch.tensor(self.chain_context, device = self.device)

        ## Constructing the spline flow either as a conditional or as a marginal flow
        if not start_from_loaded_nf:
            self.flow = zuko.flows.spline.NSF(
                                                    self.input_dim,
                                                    self.context_dim,
                                                    bins=spline_bins,  # Number of bins for the spline
                                                    passes=2,  # Number of passes (2 for coupling)
                                                    hidden_features = hidden_dims
                                                )

            if self.device == torch.device("cuda"):
                self.flow = self.flow.cuda()
        else:
            print('Loading an NF Object...')
            self.flow, B_loaded, *_= \
            torch.load(path_to_load_nf, map_location = self.device,  weights_only = False,)  
            assert self.B == B_loaded, f"The normalization scaling needs to be consistent. Loaded NF gives {B_loaded} while you gave {self.B}."
            
    def sample_from_dist(self,
                         batch_size,
                         input_dist,
                         context_dist, 
                         repeat_input,
                         repeat_context,
                         mode = 'diagonal',
                         seed = None):
        '''
        A function to help sample from the normalized distributions.

        :param: batch_size
            The batch size for training

        :param: input_dist
            The entire input distribution

        :param: context_dist:
            The entire context distribution

        :param: repeat_input
            This choice is needed for holodeck distributions that 
            give different GWB distributions for a single population 
            vector. Different GWB samples can share the same population
            sample. Only relevant if `mesh` indexing is chosen.

        :param: repeat_context
            This choice is needed for holodeck distributions that 
            give different GWB distributions for a single population 
            vector. Different GWB samples can share the same population
            sample. Only relevant if `mesh` indexing is chosen.

        :param: mode
            The type of slicing done on the arrays to sample from the 
            distributions. The options are `diagonal` and 'mesh'. 
            'mesh' uses a 2D grid to collect all the possible samples
            from the grid while `diagonal` goes over the diagonal parts of 
            the grid. 
        '''
        if seed:
            random.seed(seed)

        if mode == 'mesh':

            if self.input_axes == 3:
                input_idxs0 = random.sample(range(input_dist.shape[0]), k = batch_size)
                input_idxs1 = random.sample(range(input_dist.shape[1]), k = batch_size)
                chosen_inputs = input_dist[np.ix_(input_idxs0, input_idxs1)].reshape(batch_size**2, self.input_dim)
            elif self.input_axes == 2:
                input_idxs0 = random.sample(range(input_dist.shape[0]), k = batch_size)
                chosen_inputs = input_dist[input_idxs0]
                if repeat_input:
                    # this is very specific to holodeck
                    chosen_inputs = torch.repeat_interleave(chosen_inputs, repeats=batch_size, dim = 0)

            if self.context_exist:
                if self.context_axes == 3:
                    context_idxs = random.sample(range(context_dist.shape[1]), k = batch_size)
                    chosen_contexts = context_dist[np.ix_(input_idxs0, context_idxs)].reshape(batch_size**2, self.context_dim)
                elif self.context_axes == 2:
                    chosen_contexts = context_dist[input_idxs0]
                    if repeat_context:
                        # this is very specific to holodeck
                        chosen_contexts = torch.repeat_interleave(chosen_contexts, repeats=batch_size, dim = 0)
            else:
                chosen_contexts = None

        else:
            if self.input_axes == 3:
                input_idxs0 = random.sample(range(input_dist.shape[0]), k = batch_size)
                input_idxs1 = random.sample(range(input_dist.shape[1]), k = batch_size)
                chosen_inputs = input_dist[input_idxs0, input_idxs1]
            elif self.input_axes == 2:
                input_idxs0 = random.sample(range(input_dist.shape[0]), k = batch_size)
                chosen_inputs = input_dist[input_idxs0]

            if self.context_exist:
                if self.context_axes == 3:
                    context_idxs = random.sample(range(context_dist.shape[1]), k = batch_size)
                    chosen_contexts = context_dist[input_idxs0, context_idxs]
                elif self.context_axes == 2:
                    chosen_contexts = context_dist[input_idxs0]
            else:
                chosen_contexts = None

        return chosen_inputs, chosen_contexts
    
    def init_validation(self):
        '''
        Initiate the validation class
        '''
        self.validation_class = ValidationHell(\
        plot_save_path = self.nf_save_dir + '/Hell.pdf',
        device = self.device,
        q = 0.158,
        is_conditional = True if self.context_exist else False)

    def train(self, 
              steps,
              batch_size,
              save_freq,
              repeat_input,
              repeat_context,
              do_validation,
              validation_input,
              validation_context,
              mode = 'diagonal',
              val_hist_bins = 15,
              hell_threshold = .02,
              learning_rate = 1e-4,
              patience = 3,
              progress_bar = True):
        
        '''
        The function to train a spline flow

        :param: steps
            The toal number of training steps

        :param: batch_size
            The batch size for training

        :param: save_freq
            The frequency to which the NF flow is saved to file.
            Example could be int(steps/10) so that the flow is saved
            10 times during the training. Saving overwrites!

        :param: repeat_input
            This choice is needed for holodeck distributions that 
            give different GWB distributions for a single population 
            vector. Different GWB samples can share the same population
            sample. Only relevant if `mesh` indexing is chosen.

        :param: repeat_context
            This choice is needed for holodeck distributions that 
            give different GWB distributions for a single population 
            vector. Different GWB samples can share the same population
            sample. Only relevant if `mesh` indexing is chosen.

        :param: do_validation
            Do you want to stop the training when the validation set
            is emulated adequately based on hellinger distance estimates?

        :param: validation_input
            The validation set. The shape must be (batches, features)

        :param: validation_context
            The validation samples for the context params. The shape must be
            (batches, features)

        :param: mode
            The type of slicing done on the arrays to sample from the 
            distributions. The options are `diagonal` and 'mesh'. 
            'mesh' uses a 2D grid to collect all the possible samples
            from the grid while `diagonal` goes over the diagonal parts of 
            the grid. 

        :param: val_hist_bins
            The number of bins for making validation histograms

        :param: hell_threshold:
            The threshold under which two hellinger distance distributions
            are deemed close enough. The threshold is for the lower and upper
            `q` quantile of the two hellinger distributions.
        
        :param: learning_rate
            The ADAM optimizer learning rate.

        :param: patience
            How many times should we tolerate hellinger distances not improving
            after extra training steps?

        :param: progress_bar
            Do you want tqdm progress bar?
        '''
        optimizer = torch.optim.Adam(self.flow.parameters(), lr = learning_rate)
        # optimizer = torch.optim.AdamW(self.flow.parameters(), lr=learning_rate, weight_decay=1e-4)

        input_sample_size = self.chain_input.shape[0]
        if self.context_exist:
            context_sample_size = self.chain_context.shape[0]
        
        if do_validation:
            self.init_validation()
            one_step_towards_stoppage = 0

            val_hists = self.validation_class.batched_histogram(validation_input,
                          lower_bounds = validation_input.min(axis = -1),
                          upper_bounds = validation_input.max(axis = -1),
                          bins = val_hist_bins)

        if progress_bar:
            pbar = trange(steps, colour="blue")
        else:
            pbar = range(steps)

        for step in pbar:
            try:
                optimizer.zero_grad()

                chosen_inputs, chosen_contexts = self.sample_from_dist(batch_size,
                        input_dist = self.chain_input,
                        context_dist = self.chain_context, 
                        repeat_input = repeat_input,
                        repeat_context = repeat_context,
                        mode = mode)
                
                if self.context_exist:
                    ln_p = self.flow(chosen_contexts).log_prob(chosen_inputs)   
                else:
                    ln_p = self.flow().log_prob(chosen_inputs)

                loss = -(ln_p).mean()
                loss.backward()
                optimizer.step()

            except AssertionError:
                print('ARQS Failed. Loading from the last checkpoint...')
                self.flow, self.B = \
                    torch.load(last_nf_saved_path, map_location = self.device,  weights_only = False,)
                continue

            if not step % save_freq and step or step == steps - 1:
                torch.save([self.flow, self.B], self.nf_save_dir + f'/flow_{step}steps.pkl')
                last_nf_saved_path = self.nf_save_dir + f'/flow_{step}steps.pkl'

                if do_validation:
                    dec = self.validation_class.make_decision(validation_histograms = val_hists,
                        ndraws = int(1e5),
                        input_dim = self.input_dim,
                        validation_context_samples = validation_context,
                        lower_bounds = validation_input.min(axis = -1),
                        upper_bounds = validation_input.max(axis = -1),
                        threshold = hell_threshold,
                        nf_save_path = last_nf_saved_path,
                        progress_bar = progress_bar)
                    #print(f'Validation decision {dec}')
                    if dec:
                        one_step_towards_stoppage+=1
                    if one_step_towards_stoppage > patience:
                        print('Stopping the Training Early Based on Hellinger Distances.')
                        break



