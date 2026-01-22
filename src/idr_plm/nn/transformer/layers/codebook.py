import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# Original codebook
class EMACodebook(nn.Module):
    def __init__(
        self,
        n_codes,
        embedding_dim,
        no_random_restart=True,
        restart_thres=1.0,
        ema_decay=0.99,
    ):
        super().__init__()
        self.register_buffer("embeddings", torch.randn(n_codes, embedding_dim))
        self.register_buffer("N", torch.zeros(n_codes))
        self.register_buffer("z_avg", self.embeddings.data.clone())

        self.n_codes = n_codes
        self.embedding_dim = embedding_dim
        self._need_init = True
        self.no_random_restart = no_random_restart
        self.restart_thres = restart_thres
        self.freeze_codebook = False
        self.ema_decay = ema_decay

    def reset_parameters(self):
        # For meta init
        pass

    def _tile(self, x):
        d, ew = x.shape
        if d < self.n_codes:
            n_repeats = (self.n_codes + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _init_embeddings(self, z):
        # z: [b, t, c]
        self._need_init = False
        flat_inputs = z.view(-1, self.embedding_dim)
        y = self._tile(flat_inputs)

        y.shape[0]
        _k_rand = y[torch.randperm(y.shape[0])][: self.n_codes]
        if dist.is_initialized():
            dist.broadcast(_k_rand, 0)
        self.embeddings.data.copy_(_k_rand)
        self.z_avg.data.copy_(_k_rand)
        self.N.data.copy_(torch.ones(self.n_codes))

    def forward(self, z):
        # z: [b, t, c]
        if self._need_init and self.training and not self.freeze_codebook:
            self._init_embeddings(z)
        # z is of shape [batch_size, sequence length, channels]
        flat_inputs = z.view(-1, self.embedding_dim)
        distances = (
            (flat_inputs**2).sum(dim=1, keepdim=True)
            - 2 * flat_inputs @ self.embeddings.t()
            + (self.embeddings.t() ** 2).sum(dim=0, keepdim=True)
        )  # [bt, c]

        encoding_indices = torch.argmin(distances, dim=1)
        encoding_indices = encoding_indices.view(*z.shape[:2])  # [b, t, ncode]

        embeddings = F.embedding(encoding_indices, self.embeddings)  # [b, t, c]

        commitment_loss = 0.25 * F.mse_loss(z, embeddings.detach())

        # EMA codebook update
        if self.training and not self.freeze_codebook:
            assert False, "Not implemented"
        embeddings_st = (embeddings - z).detach() + z

        return embeddings_st, encoding_indices, commitment_loss

    def dictionary_lookup(self, encodings):
        embeddings = F.embedding(encodings, self.embeddings)
        return embeddings

    def soft_codebook_lookup(self, weights: torch.Tensor) -> torch.Tensor:
        return weights @ self.embeddings


# Simplified codebook
class MolCodebook(nn.Module):
    def __init__(
        self,
        n_codes,
        embedding_dim,
        pad_value=0,
        no_random_restart=True,
        restart_thres=1.0,
        ema_decay=0.99,
        beta=0.25,
        optim_method="vq",
        train_codebook=True,
    ):
        super().__init__()
        # FH: This is for EMA updates, we are using a parameter and letting the loss
        #   terms handle the updates
        # self.register_buffer("embeddings", torch.randn(n_codes, embedding_dim))
        # self.embeddings = nn.Parameter(torch.randn(n_codes, embedding_dim))

        # FH: These buffers are used for exponential moving average updates. Disabled for now,
        #   optimizing the embeddings through the VQ loss as described in
        #   https://arxiv.org/abs/1711.00937

        if optim_method == "vq":
            self.embeddings = nn.Parameter(torch.randn(n_codes, embedding_dim))
        elif optim_method == "ema":
            self.register_buffer("embeddings", torch.zeros(n_codes, embedding_dim))
        self.register_buffer("N", torch.zeros(n_codes))
        self.register_buffer("z_avg", self.embeddings.data.clone())

        self.n_codes = n_codes
        self.embedding_dim = embedding_dim
        self.pad_value = pad_value
        self.optim_method = optim_method
        # self._need_init = True
        self.no_random_restart = no_random_restart
        self.restart_thres = restart_thres

        # FH: Rather than using a freeze codebook flag, let freezing be handled
        #   at the encoder level
        # self.freeze_codebook = freeze_codebook

        self.ema_decay = ema_decay
        self.optim_method = optim_method
        self.train_codebook = train_codebook
        # Scaling factor for commitment cost
        self.beta = beta
        if train_codebook:
            self.need_init = True
        else:
            self.need_init = False

    # FH: Unnecessary for now
    def _tile(self, x):
        d, ew = x.shape
        if d < self.n_codes:
            n_repeats = (self.n_codes + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    # FH: Let's simplify this initialization to only use glorot uniform
    def _init_embeddings(self, z):
        # z: [b, t, c]
        self.need_init = False
        flat_inputs = z.reshape(-1, self.embedding_dim)
        y = self._tile(flat_inputs)

        _k_rand = y[torch.randperm(y.shape[0])][: self.n_codes]

        self.embeddings.data.copy_(_k_rand)
        self.z_avg.data.copy_(_k_rand)
        self.N.data.copy_(torch.ones(self.n_codes))

    # Random Glorot uniform does not seem to work well as an initialization technique,
    #   try by initalizing using the encoder outputs
    # def _init_embeddings(self):
    #     nn.init.xavier_uniform_(self.embeddings)

    def forward(self, z, mask=None):
        """
        z: [b, t, c]
        mask: [b, t, c]
        """
        # FH: Ths ensures the embeddings are initialized on the first training forward pass
        if self.training and self.need_init:
            print("Initializing embeddings through tiling...")
            self._init_embeddings(z)
        if mask is None:
            # FH: Making the masks the same shape makes masking and averaging in the loss
            #   easier
            mask = torch.ones(z.shape, device=z.device).long()
        # FH: Understand the motivation for this, but we can just load the state dict
        #   and freeze the codebook if necessary.
        # z: [b, t, c]
        # if self._need_init and self.training and not self.freeze_codebook:
        #     self._init_embeddings(z)
        # z is of shape [batch_size, sequence length, channels]
        flat_inputs = z.reshape(-1, self.embedding_dim)
        distances = (
            (flat_inputs**2).sum(dim=1, keepdim=True)
            - 2 * flat_inputs @ self.embeddings.t()
            + (self.embeddings.t() ** 2).sum(dim=0, keepdim=True)
        )  # [bt, c]

        encoding_indices = torch.argmin(distances, dim=1)
        encoding_indices = encoding_indices.view(*z.shape[:2])  # [b, t]

        embeddings = F.embedding(encoding_indices, self.embeddings)  # [b, t, c]

        # Masked commitment loss
        commitment_loss = (z - embeddings.detach()).pow(2)
        commitment_loss *= mask
        commitment_loss = commitment_loss.sum() / mask.sum()
        commitment_loss *= self.beta

        # EMA/VQ codebook updates

        # FH: The self.training flag is a PyTorch internal attribute that is True when the model is
        #   in the training loop and False when in a validation/evaluation loop. Checking the value
        #   of self.training ensures that ema/vq updates do not occur on the validation set
        if self.training and self.train_codebook and self.optim_method == "ema":
            # SI: Use no grad context here to prevent gradient accumulationg
            #   in an update step that does not require gradients, gradient
            #   information is retained later via the straight through estimator
            #   taking gradients from z (the input)
            # FH: Let's double-check this part just to be safe...
            with torch.no_grad():
                encodings_one_hot = F.one_hot(
                    encoding_indices, num_classes=self.n_codes
                ).type_as(flat_inputs)
                encodings_one_hot = encodings_one_hot.view(-1, self.n_codes)
                n_i = encodings_one_hot.sum(dim=0)
                # mask = n_i > 0
                z_i_sum = torch.einsum("bi, bj -> ij", encodings_one_hot, flat_inputs)

                self.N = self.ema_decay * self.N + (1 - self.ema_decay) * n_i
                self.z_avg = (
                    self.ema_decay * self.z_avg + (1 - self.ema_decay) * z_i_sum
                )
                n_i = self.N.unsqueeze(-1)
                self.embeddings.data = self.z_avg / n_i
                vq_loss = 0

        elif self.training and self.train_codebook and self.optim_method == "vq":
            # Masked VQ loss
            vq_loss = (z.detach() - embeddings).pow(2)
            vq_loss *= mask
            vq_loss = vq_loss.sum() / mask.sum()

        elif (not self.training) or (not self.train_codebook):
            vq_loss = 0  # Explicit case for inference

        embeddings_st = (embeddings - z).detach() + z

        return embeddings_st, encoding_indices, commitment_loss, vq_loss

    def dictionary_lookup(self, encodings):
        embeddings = F.embedding(encodings, self.embeddings)
        return embeddings

    def soft_codebook_lookup(self, weights):
        return weights @ self.embeddings
