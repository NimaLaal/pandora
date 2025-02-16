import jax
import jax.dlpack as jd
import torch
import torch.utils.dlpack as td
import jax.numpy as jnp
import torch.utils.dlpack as torchdlpack
class NFastroinference(object):
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
        ast_param_idxs):

        self.nf = pyro_nf_object
        self.B = scale
        self.mean_gwb = mean[gwb_freq_idxs][None]
        self.mean_ast = mean[ast_param_idxs][None]
        self.half_range_gwb = half_range[gwb_freq_idxs][None]
        self.half_range_ast = half_range[ast_param_idxs][None]


    def convert_torch_to_numpy(self, torch_tensor):
        # return np.from_dlpack(torch_tensor)
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