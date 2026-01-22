"""
General utilities for the diffusion portion of the transfusion workflow
"""

import torch
import torch.nn as nn
import numpy as np
import tqdm
import pickle


class DDPMLoss(nn.Module):
    """
    DDPM score matching loss based on the variance preserving SDE along with
    next-token-prediction CE loss

    Args:
        lambda_diff: float
            Weighting factor for the diffusion portion of the overall loss
        diff_args: dict
            A dictionary which contains the following arguments for transfusion:

            sigma_min: float
                minimum value of the sigma noise scale for forward noising
            sigma_max: float
                maximum value of the sigma noise scale for forward noising
            p_path: str
                path to precomputed p values
            score_path: str
                path to precomputed score values

        ce_ignore_index: int
            Integer index for the token to ignore for the cross entropy loss
    """

    def __init__(
        self, lambda_diff: float, diff_args: dict[str], ce_ignore_index: int
    ) -> None:
        super().__init__()
        self.lambda_diff = lambda_diff
        self.diff_args = diff_args
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss(ignore_index=ce_ignore_index)

    def ddpm_forward(self, x_struct: torch.Tensor) -> tuple[torch.Tensor]:
        """
        Generate the noised input and quantities needed for the loss calculation

        Args:
            x_struct: torch.Tensor
                (B, ...) tensor of input data

        Returns:
            x_t: torch.Tensor
                (B, ...) tensor of noised input
            grad_log_p: torch.Tensor
                (B, ...) tensor of gradient of log probability
            t: torch.Tensor
                (B, 1) tensor of time steps used
            var_t: torch.Tensor
                (B, ...) tensor of variances for scaling the loss
        """
        assert "eps" in self.diff_args
        assert "beta_min" in self.diff_args
        assert "beta_max" in self.diff_args

        eps = self.diff_args["eps"]
        beta_min = self.diff_args["beta_min"]
        beta_max = self.diff_args["beta_max"]
        b = x_struct.shape[0]
        # Sample uniform time steps over the batch dimension
        t = torch.rand(b, device=x_struct.device) * (1 - eps) + eps
        int_beta = (beta_min + 0.5 * (beta_max - beta_min) * t) * t
        dim_diff = len(x_struct.shape) - 1
        int_beta = int_beta.view(b, *([1] * dim_diff))
        mu_t = x_struct * torch.exp(-0.5 * int_beta)
        var_t = -torch.expm1(-int_beta)

        # Sample at timesteps t and compute gradient of score
        x_t = torch.randn_like(x_struct) * torch.sqrt(var_t) + mu_t
        grad_log_p = -(x_t - mu_t) / var_t
        return x_t, grad_log_p, t, var_t

    def forward(
        self,
        model: nn.Module,
        x_token: torch.Tensor,
        x_structure: torch.Tensor,
        struct_mask: torch.Tensor,
        sequence_id: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float]:
        """
        Compute the transfusion loss over both the tokens and the structural outputs
        of the model

        The form of the perturbation kernel for the DDPM process is given by
        Equation 29 in https://arxiv.org/pdf/2011.13456

        A time embedding is added to the noised input before it is transformed
        into the correct dimensionality for passing into the transformer model

        Args:
            model: nn.Module
                The transformer model which acts as a next token generator and score model
            x_token: torch.Tensor
                (B, S) Token sequence for embedding
            x_structure: torch.Tensor
                Structural sequence
            struct_mask: torch.Tensor
                (B, T) mask tensor for selecting out dihedral angles
            sequence_id: torch.Tensor
                (B, N) tensor of sequence ids

        Returns:
            loss: scalar tensor of loss
            ce_loss: Cross-entropy comopnent of the loss for next-token prediction
            struct_loss: MSE component of the loss for structural diffusion
        """
        x_t, grad_log_p, time, var_t = self.ddpm_forward(x_structure)
        out = model(x_token, x_t, struct_mask, sequence_id, time)
        # Need to separate out the structural and token outputs
        token_out, struct_out = out

        # Compute the cross entropy loss over tokens
        b_tok, t_tok = x_token.shape
        x_token_flat = x_token.reshape(b_tok * t_tok)
        sequence_flat = sequence_id.reshape(b_tok * t_tok)
        x_token_selected = x_token_flat[sequence_flat != 2]
        # Prevent loss computation over the sequence stop token
        struct_stop_token = model.smi_token_info["struct_end"]
        struct_stop_token_mask = x_token_selected != struct_stop_token
        x_token_selected = x_token_selected[struct_stop_token_mask]
        token_out = token_out[struct_stop_token_mask]
        ce_loss = self.ce(token_out, x_token_selected)

        # Compute the MSE loss over the structural inputs
        b_str, t_str, e_str = grad_log_p.shape
        grad_log_p_flat = grad_log_p.reshape(b_str * t_str, e_str)
        struct_mask_flat = struct_mask.reshape(b_str * t_str)
        grad_log_p_selected = grad_log_p_flat[~struct_mask_flat]
        struct_loss = (struct_out - grad_log_p_selected) ** 2
        # Ensure that the loss is scaled by the variance
        struct_loss = (struct_loss * var_t).mean()

        return (
            ce_loss + self.lambda_diff * struct_loss,
            ce_loss.item(),
            struct_loss.item(),
        )


class TorDiffLoss(nn.Module):
    """
    Loss module based on torsional diffusion framework on wrapped normals

    Tested in isolation using a standard transformer encoder on a specific set of dihedrals

    For more details, see https://arxiv.org/pdf/2206.01729 and the associated codebase
    """

    def __init__(
        self, lambda_diff: float, diff_args: dict[str], ce_ignore_index: int
    ) -> None:
        """
        Args:
            lambda_diff: float
                Weighting factor for the diffusion portion of the overall loss
            diff_args: dict
                A dictionary which contains the following arguments for transfusion:

                sigma_min: float
                    minimum value of the sigma noise scale for forward noising. This is relative to pi, so will be
                    multiplied by np.pi inside
                sigma_max: float
                    maximum value of the sigma noise scale for forward noising. This is relative to pi, so will be
                    multiplied by np.pi inside
                p_path: str
                    path to precomputed p values
                score_path: str
                    path to precomputed score values

            ce_ignore_index: int
                Integer index for the token to ignore for the cross entropy loss

        Notes:
            Exponential noising is used for this diffusion process, with the parameters chosen relative to pi.
            In exponential noising, the noise scale is defined as:

                \sigma(t) = \sigma_{min}^{1-t} * \sigma_{max}^t

            For this process, we carry out diffusion on the range [-pi, pi] instead of [0, 2pi].

            This implementation is taken from the torsional diffusion codebase which can be found at:
            https://github.com/gcorso/torsional-diffusion
        """
        assert "sigma_min" in diff_args
        assert "sigma_max" in diff_args
        assert "p_path" in diff_args
        assert "score_path" in diff_args

        super().__init__()

        # Constants from the original codebase
        self.X_MIN, self.X_N = 1e-5, 5000  # relative to pi
        self.SIGMA_MIN, self.SIGMA_MAX, self.SIGMA_N = 3e-3, 2, 5000  # relative to pi

        x = 10 ** np.linspace(np.log10(self.X_MIN), 0, self.X_N + 1) * np.pi
        sigma = (
            10
            ** np.linspace(
                np.log10(self.SIGMA_MIN), np.log10(self.SIGMA_MAX), self.SIGMA_N + 1
            )
            * np.pi
        )

        sigma_min = diff_args["sigma_min"]
        sigma_max = diff_args["sigma_max"]
        p_path = diff_args["p_path"]
        score_path = diff_args["score_path"]

        try:
            print("Loading from files")
            self.p_ = np.load(p_path)
            self.score_ = np.load(score_path)
        except FileNotFoundError:
            print("Recomputing p and score values")
            self.p_ = self._p(x, sigma[:, None], N=100)
            self.score_ = self._grad(x, sigma[:, None], N=100) / self.p_
            np.save(p_path, self.p_)
            np.save(score_path, self.score_)

        score_norm = self.get_score(
            self._sample(sigma[None].repeat(10000, 0).flatten()),
            sigma[None].repeat(10000, 0).flatten(),
        ).reshape(10000, -1)
        # Scaling factor for the score
        self.score_norm_ = (score_norm**2).mean(0)

        # These sigmas are for the noising process only! They are defined relative to pi, so
        #   multiply the floating point values accordingly.
        self.sigma_max = sigma_max * np.pi
        self.sigma_min = sigma_min * np.pi
        self.lambda_diff = lambda_diff
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss(ignore_index=ce_ignore_index)

    def _p(self, x: np.ndarray, sigma: float, N: int = 10) -> np.ndarray:
        p_ = 0
        for i in tqdm.trange(-N, N + 1):
            p_ += np.exp(-((x + 2 * np.pi * i) ** 2) / 2 / sigma**2)
        return p_

    def _grad(self, x: np.ndarray, sigma: float, N: int = 10) -> np.ndarray:
        p_ = 0
        for i in tqdm.trange(-N, N + 1):
            p_ += (
                (x + 2 * np.pi * i)
                / sigma**2
                * np.exp(-((x + 2 * np.pi * i) ** 2) / 2 / sigma**2)
            )
        return p_

    def get_score(self, x: np.ndarray, sigma: float) -> np.ndarray:
        x = (x + np.pi) % (2 * np.pi) - np.pi  # On [-pi, pi]
        sign = np.sign(x)
        x = np.log(np.abs(x) / np.pi)
        x = (x - np.log(self.X_MIN)) / (0 - np.log(self.X_MIN)) * self.X_N
        x = np.round(np.clip(x, 0, self.X_N)).astype(int)
        sigma = np.log(sigma / np.pi)
        sigma = (
            (sigma - np.log(self.SIGMA_MIN))
            / (np.log(self.SIGMA_MAX) - np.log(self.SIGMA_MIN))
            * self.SIGMA_N
        )
        sigma = np.round(np.clip(sigma, 0, self.SIGMA_N)).astype(int)
        return -sign * self.score_[sigma, x]

    def get_p(self, x: np.ndarray, sigma: float) -> np.ndarray:
        x = (x + np.pi) % (2 * np.pi) - np.pi  # On [-pi, pi]
        x = np.log(np.abs(x) / np.pi)
        x = (x - np.log(self.X_MIN)) / (0 - np.log(self.X_MIN)) * self.X_N
        x = np.round(np.clip(x, 0, self.X_N)).astype(int)
        sigma = np.log(sigma / np.pi)
        sigma = (
            (sigma - np.log(self.SIGMA_MIN))
            / (np.log(self.SIGMA_MAX) - np.log(self.SIGMA_MIN))
            * self.SIGMA_N
        )
        sigma = np.round(np.clip(sigma, 0, self.SIGMA_N)).astype(int)
        return self.p_[sigma, x]

    def _sample(self, sigma: float) -> np.ndarray:
        out = sigma * np.random.randn(*sigma.shape)
        out = (out + np.pi) % (2 * np.pi) - np.pi  # On [-pi, pi]
        return out

    def get_score_norm(self, sigma: float) -> np.ndarray:
        sigma = np.log(sigma / np.pi)
        sigma = (
            (sigma - np.log(self.SIGMA_MIN))
            / (np.log(self.SIGMA_MAX) - np.log(self.SIGMA_MIN))
            * self.SIGMA_N
        )
        sigma = np.round(np.clip(sigma, 0, self.SIGMA_N)).astype(int)
        return self.score_norm_[sigma]

    def forward(
        self,
        model: nn.Module,
        x_input: torch.Tensor,  # The input of all tokens
        x_target: torch.Tensor,  # The target
        x_structure: torch.Tensor,
        struct_mask: torch.Tensor,
        sequence_id: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float]:
        """
        Args:
            x: torch.Tensor
                Tensor of shape (B, T) where B is the batch size and T is the number of dihedral angles.
                All values are between 0 and 2 * pi
            model: nn.Module
                The score network model to use for computing the score matching loss

        Notes:
            The score model does not have to be intrinsically periodic, as the periodicity is already
            accounted for in the precomputed score values.

            The sampling is done over sigmas. The time step is computed from the resulting sigmas.
        """
        # import pdb; pdb.set_trace()
        # assert x_input.shape == x_target.shape
        # Sample the sigmas and get the time steps
        sigmas = np.exp(
            np.random.uniform(
                low=np.log(self.sigma_min),
                high=np.log(self.sigma_max),
                size=x_structure.shape[0],
            )
        ).reshape(-1, 1)
        ts = (np.log(sigmas) - np.log(self.sigma_min)) / (
            np.log(self.sigma_max) - np.log(self.sigma_min)
        )
        # Sample from the normal distribution and get the score
        #   This term is added to the original x as noise
        x_t = np.random.normal(loc=0.0, scale=sigmas, size=x_structure.shape)
        score = self.get_score(x_t, sigmas)
        score_norm = self.get_score_norm(sigmas)

        x_t = torch.tensor(x_t, device=x_structure.device).float()
        score = torch.tensor(score, device=x_structure.device).float()
        score_norm = torch.tensor(score_norm, device=x_structure.device).float()
        ts = torch.tensor(ts, device=x_structure.device).float()
        # inp here is the noised structural features
        inp = x_t + x_structure

        # Print some shapes for debugging
        # print("x_t shape: ", x_t.shape)
        # print("score shape: ", score.shape)
        # print("score_norm shape: ", score_norm.shape)
        # print("ts shape: ", ts.shape)
        # print("inp shape: ", inp.shape)

        out = model(x_input, inp, struct_mask, sequence_id, ts)
        # Separate out the token and structural outputs
        token_out, struct_out = out

        # Compute the token (cross entropy) loss against a shifted target for next token prediction!
        # Here, x_token is the input token sequence of the form:
        #    <TOK_START><SMILES><STRUCT_START><STRUCT><STRUCT_END>
        # For cross entropy, compute the loss between <TOK_START><SMILES> and <SMILES><STRUCT_START>,
        #    shifting off by one
        token_out_selection_mask = sequence_id == 1
        token_target_selection_mask = token_out_selection_mask[:, :-1]

        b_inp, t_inp = x_input.shape
        token_out_flat = token_out.reshape(b_inp * t_inp, -1)
        token_out_selection_mask = token_out_selection_mask.reshape(b_inp * t_inp)
        token_out_select = token_out_flat[token_out_selection_mask]

        b_tar, t_tar = x_target.shape
        x_target_flat = x_target.reshape(b_tar * t_tar)
        token_target_selection_mask = token_target_selection_mask.reshape(b_tar * t_tar)
        x_target_select = x_target_flat[token_target_selection_mask]

        ce_loss = self.ce(token_out_select, x_target_select)

        # b_tok, t_tok = x_token.shape
        # x_token_flat = x_token.reshape(b_tok * t_tok)
        # token_out = token_out.reshape(b_tok * t_tok, -1)
        # sequence_flat = sequence_id.reshape(b_tok * t_tok)
        # #Select out non-structure tokens. Note that we are NOT removing padding
        # #   tokens. This is because the cross-entropy loss is initialized with
        # #   an ignore index to ignore padding tokens in the sequence.
        # token_selection_mask = (sequence_flat != 2)
        # x_token_selected = x_token_flat[token_selection_mask]
        # token_out = token_out[token_selection_mask]
        # #Prevent loss computation over the structure sequence stop token
        # struct_stop_token = model.token_info['input']['STRUCT']['STRUCT_END']
        # struct_stop_token_mask = x_token_selected != struct_stop_token
        # x_token_selected = x_token_selected[struct_stop_token_mask]
        # token_out = token_out[struct_stop_token_mask]
        # ce_loss = self.ce(token_out, x_token_selected)

        # Compute the structural (MSE) loss, based on testing it seems
        #   diffusion on the angles directly works best with the multihead embedding,
        #   this requires a grouped reduction which we perform here with segment_csr and
        #   the mean reduction.
        # import pdb; pdb.set_trace()
        assert struct_out.shape == score.shape
        b_str, t_str = score.shape
        difference = struct_out - score
        diff_loss = (difference**2) / score_norm
        diff_loss_flattened = diff_loss.reshape(b_str * t_str)
        struct_mask_flattened = struct_mask.reshape(b_str * t_str)
        diff_loss_selected = diff_loss_flattened[~struct_mask_flattened]
        # Take the average loss over non-padding structure elements
        diff_loss = diff_loss_selected.mean()
        if torch.isnan(diff_loss):
            # Compile a dictionary of all the tensors for debugging and terminate training
            tensor_dict = {
                "x_input": x_input,
                "x_target": x_target,
                "x_structure": x_structure,
                "struct_mask": struct_mask,
                "sequence_id": sequence_id,
                "sigmas": sigmas,
                "ts": ts,
                "x_t": x_t,
                "score": score,
                "score_norm": score_norm,
                "inp": inp,
                "token_out": token_out,
                "struct_out": struct_out,
                "ce_loss": ce_loss,
                "diff_loss": diff_loss,
            }
            with open("diffusion_nan_debug_tensors.pkl", "wb") as f:
                pickle.dump(tensor_dict, f)
            raise ValueError(
                "Diffusion loss is NaN, check tensors in diffusion_nan_debug_tensors.pkl"
            )
        return ce_loss, diff_loss, self.lambda_diff


### Sampling methods for transfusion ###


def calc_g_t(t: float, sigma_min: float, sigma_max: float) -> float:
    """Computes the diffusion coefficient at time t for torsional diffusion

    Args:
        t: float
            Time step
        sigma_min: float
            Minimum value of the sigma noise scale
        sigma_max: float
            Maximum value of the sigma noise scale

    Returns:
        g(t): float
            The diffusion coefficient at time t

    Notes:
        The diffusion process for torsional diffusion is an exponential noising process, and g(t) is defined as:

            g(t) = sqrt(d/dt sigma(t)**2)**1/2, where sigma(t) = sigma_min**(1-t) * sigma_max**t
    """
    s_min, s_max = torch.tensor(sigma_min), torch.tensor(sigma_max)
    return torch.sqrt(
        2
        * (torch.log(s_max) - torch.log(s_min))
        * (s_max ** (2 * t))
        * (s_min ** (2 * (1 - t)))
    )


def sample_torsions(
    score_network: nn.Module,
    model_inputs: dict[str, torch.Tensor],
    n_samples: int,
    n_time_steps: int = 1000,
    feat_dim: int = 4,
    sigma_min: float = 0.01 * np.pi,
    sigma_max: float = np.pi,
    device: torch.device = None,
) -> tuple[torch.Tensor]:
    """Samples torsions using with boundaries of [-pi, pi]

    Args:
        score_network: nn.Module
            The score network for denoising diffusion
        model_inputs: tuple[torch.Tensor]
            All other required inputs for the model besides the time step and noised vectors
        n_samples: int
            The number of samples to generate
        n_time_steps: int
            The number of reverse time steps to take
        feat_dim: int
            The feature dimension of the vectors to reverse diffuse
        sigma_min: float
            Minimum value of the sigma noise scale for forward noising
        sigma_max: float
            Maximum value of the sigma noise scale for forward noising
        device: torch.device
            GPU/CPU for sampling

    Returns:
        all_x_t: torch.Tensor
            (n_samples, n_time_steps, feat_dim) tensor of all noised samples

    Notes:
        For torsional diffusion, there is no drift term so f(x, t) = 0
    """
    all_x_t = []
    # Prior is a uniform distribution over the Taurus
    x_t = np.random.uniform(low=-np.pi, high=np.pi, size=(n_samples, feat_dim))
    x_t = torch.tensor(x_t, device=device).float()
    all_t = torch.linspace(1, 0, n_time_steps, device=device)
    all_dt = torch.diff(all_t)
    all_x_t.append(x_t)

    for dt, t in zip(all_dt, all_t[:-1]):
        g_t = calc_g_t(t, sigma_min, sigma_max)
        _, score = score_network(
            struct_input=x_t, ts=torch.ones_like(x_t)[:, 0:1] * t, **model_inputs
        )
        drift = -(g_t**2) * score
        diffusion = g_t
        x_t = (
            x_t
            + drift * dt
            + diffusion * torch.randn_like(x_t) * torch.sqrt(torch.abs(dt))
        )
        # Force on [-pi, pi] on every interval
        x_t = torch.fmod(x_t + 3 * torch.pi, 2 * torch.pi) - torch.pi
        all_x_t.append(x_t)
    all_x_t = torch.stack(all_x_t, dim=1).detach().cpu().numpy()
    return all_x_t
