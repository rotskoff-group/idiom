"""
Methods for top-k, top-p and regular sampling. Also implements a temperature scaling of the logits
"""

import torch
from torch import Tensor


class TokenSampler:
    """
    Class for sampling tokens based on different strategies: top-k, top-p, and full sampling.
    """

    def __init__(
        self,
        method: str = "top_k",
        sample_val: int | float = 5,
        temperature: float = 1.0,
    ):
        """
        The sample val here is:
            top-k: An integer k representing the number of top tokens to consider.
            top-p: A float p representing the cumulative probability threshold.
            full: Not used, but can be set to any value since it samples from the full distribution.
        Args:
            method (str): Sampling method to use. Options are 'top_k', 'top_p', or 'full'.
            sample_val (int | float): The value for sampling. For 'top_k', it is an integer k; for 'top_p', it is a float p.
            temperature (float): Temperature scaling factor for logits. Scaled logits are given as logits/temperature.
        """
        assert method in ["top_k", "top_p", "full"], (
            "Method must be one of 'top_k', 'top_p', or 'full'."
        )
        self.method = method
        self.sample_val = sample_val
        self.temperature = temperature

    def _get_top_k_sample_batched(
        self, k_val: int | float, character_probabilities: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Generates the next character using top-k sampling scheme.

        In top-k sampling, the probability mass is redistributed among the
        top-k next tokens, where k is a hyperparameter. Once redistributed,
        the next token is sampled from the top-k tokens.
        """
        if character_probabilities.ndim == 1:
            # Character probabilities are 1D, so we need to add a batch dimension
            character_probabilities = character_probabilities.unsqueeze(0)  # (1, E)
        top_values, top_indices = torch.topk(
            character_probabilities, k_val, sorted=True
        )
        # Take the sum of the top probabilities and renormalize
        tot_probs = top_values / torch.sum(top_values, dim=-1).reshape(-1, 1)
        # Sample from the top k probabilities. This represents a multinomial distribution
        try:
            assert torch.allclose(torch.sum(tot_probs, dim=-1), torch.tensor(1.0))
        except Exception:
            print("Probabilities did not pass allclose check!")
            print(f"Sum of probs is {torch.sum(tot_probs)}")
        selected_index = torch.multinomial(tot_probs, 1)
        # For gather to work, both tensors have to have the same number of dimensions:
        if len(top_indices.shape) != len(selected_index.shape):
            top_indices = top_indices.reshape(selected_index.shape[0], -1)
        output = torch.gather(top_indices, -1, selected_index)
        output_token_probs = torch.gather(tot_probs, -1, selected_index)
        return output, output_token_probs

    def _get_top_p_sample(self, p_val, character_probabilities):
        """
        Generates the next character using top-p sampling scheme

        In top-p sampling, the next character is sampled from the smallest set of
        characters whose cumulative probability exceeds p_val.
        """
        if character_probabilities.ndim == 1:
            # Character probabilities are 1D, so we need to add a batch dimension
            character_probabilities = character_probabilities.unsqueeze(0)  # (1, E)
        sorted_probs, sorted_indices = torch.sort(
            character_probabilities, descending=True
        )
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        # Find the smallest set of indices whose cumulative probability exceeds p_val
        #   and sample from that
        mask = cum_probs > p_val
        # Shift the mask to the right by one position to include the first token
        #   that exceeds the p_val. This is because we want to include the first token
        mask = torch.roll(mask, shifts=1, dims=-1)
        mask[:, 0] = False  # Ensure the first token is not masked out
        sorted_probs[mask] = 0
        renormed_probs = sorted_probs / torch.sum(sorted_probs, dim=-1, keepdim=True)
        try:
            assert torch.allclose(torch.sum(renormed_probs), torch.tensor(1.0))
        except Exception:
            print("Probabilities did not pass allclose check!")
            print(f"Sum of probs is {torch.sum(renormed_probs)}")
        selected_index = torch.multinomial(renormed_probs, 1)
        output = torch.gather(sorted_indices, -1, selected_index)
        output_token_probs = torch.gather(renormed_probs, -1, selected_index)
        return output, output_token_probs

    def _get_full_sample(self, character_probabilities):
        """
        Generates the next character using full sampling scheme

        The next character is chosen randomly based on the full distribution
        of next character probabilities.
        """
        # import ipdb; ipdb.set_trace()
        # Always sum across the last dimension when checking probability mass sums to 1
        try:
            tot_prob = torch.sum(character_probabilities, dim=-1)
            # assert torch.allclose(torch.sum(tot_prob, dim=-1), torch.tensor(1.0)) # Wrong, shouldn't sum over batch size
            assert torch.allclose(tot_prob, torch.tensor(1.0))
        except Exception:
            print("Probabilities did not pass allclose check! (full sample)")
            print(f"Sum of probs is {torch.sum(tot_prob)}")

        selected_index = torch.multinomial(character_probabilities, 1)
        selected_probs = torch.gather(character_probabilities, -1, selected_index)
        return selected_index, selected_probs

    def __call__(self, logits):
        """Takes in model logits and returns sampled indices using the specified sampling method.
        Args:
            logits (torch.Tensor): The raw logits from the model OF THE NEXT POSITION, of shape (batch_size, vocab_size).
        Returns:
            torch.Tensor: Indices of the sampled tokens, of shape (batch_size,).
        """
        # Scale logits by temperature
        scaled_logits = logits / self.temperature  # (N, E)
        # Convert logits to probabilities
        character_probabilities = torch.softmax(scaled_logits, dim=-1)  # (N, E)

        if self.method == "top_k":
            return self._get_top_k_sample_batched(
                self.sample_val, character_probabilities
            )
        elif self.method == "top_p":
            return self._get_top_p_sample(self.sample_val, character_probabilities)
        elif self.method == "full":
            return self._get_full_sample(character_probabilities)
        else:
            raise ValueError(f"Unknown sampling method: {self.method}")
