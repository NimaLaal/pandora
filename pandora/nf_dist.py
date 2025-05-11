import jax
import jax.dlpack as jd
import torch
import torch.utils.dlpack as td
import jax.numpy as jnp
import torch.utils.dlpack as torchdlpack
torch.set_default_dtype(torch.float64)


class NFastroinference(object):
    """
    To use a normalizing flow pyro object as a distribution

    pending...
    """

    def __init__(
        self, pyro_nf_object, nf_type, mean, half_range, scale, gwb_freq_idxs, ast_param_idxs
    ):
        self.nf = pyro_nf_object
        self.B = scale
        self.mean_gwb = mean[gwb_freq_idxs]
        self.mean_ast = mean[ast_param_idxs]
        self.half_range_gwb = half_range[gwb_freq_idxs]
        self.half_range_ast = half_range[ast_param_idxs]
        self.nf_type = nf_type
        if self.nf_type not in ['marg', 'reversed', 'standard']:
            raise ValueError('`nf_type` must be one of marg, reversed, or standard.')
        
    def convert_torch_to_numpy(self, torch_tensor):
        # return np.from_dlpack(torch_tensor)
        return torch_tensor.detach().numpy()

    def convert_numpy_to_torch(self, numpy_array):
        return torchdlpack.from_dlpack(numpy_array.__dlpack__())

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
        if self.nf_type == 'standard':
            scaled_ast = self.convert_numpy_to_torch(
                self.transform_params_to_scaled_interval(astro_params)
            )
            return self.convert_torch_to_numpy(
                self.nf.condition(scaled_ast).log_prob(scaled_gwb)
        )
        elif self.nf_type == 'reversed':
            scaled_ast = self.convert_numpy_to_torch(
                self.transform_params_to_scaled_interval(astro_params)
            )
            return self.convert_torch_to_numpy(
                self.nf.condition(scaled_gwb).log_prob(scaled_ast)
        )
        elif self.nf_type == 'marg':
            return self.convert_torch_to_numpy(
                self.nf.log_prob(scaled_gwb)
        )           

    def sample(self, batch_shape, context_params):
        if self.nf_type == 'standard':
            scaled_ast = self.convert_numpy_to_torch(
                self.transform_params_to_scaled_interval(context_params)
            )
            scaled_gwb = self.convert_torch_to_numpy(
                self.nf.condition(scaled_ast).sample(batch_shape)
            )
            return self.transform_rho_to_physical_interval(scaled_gwb)
        
        elif self.nf_type == 'reversed':
            scaled_gwb = self.convert_numpy_to_torch(
                self.transform_rho_to_scaled_interval(context_params)
            )
            scaled_ast = self.convert_torch_to_numpy(
                self.nf.condition(scaled_gwb).sample(batch_shape)
            )
            return self.transform_params_to_physical_interval(scaled_ast)
        
        elif self.nf_type == 'marg':
            scaled_gwb = self.convert_torch_to_numpy(
                self.nf.sample(batch_shape)
            )
            return self.transform_rho_to_physical_interval(scaled_gwb)
