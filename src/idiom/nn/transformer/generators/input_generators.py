import numpy as np
from idiom.nn.transformer.utils.tokenizer import CharTokenizer


def look_ahead_residues(residues: list[str], tokenizer: CharTokenizer) -> int:
    """Determines the maximum length of the residue sequences in numbers of tokens

    Args:
        residues: list[str]
            List of residue sequences
        tokenizer: CharTokenizer
            Instance of the CharTokenizer class for tokenizing the residue sequences

    Returns:
        int: The maximum length of the tokenized residue sequences
    """
    max_len = 0
    for i in range(len(residues)):
        tokens = tokenizer.tokenize(residues[i])
        max_len = max(max_len, len(tokens))
    # To account for additional stop token
    return max_len + 1


# Base class for input generators, to be inherited by others
class InputGeneratorBase:
    # Getters have concrete implementations, but constructor and transform are not implemented
    def __init__(
        self, residues: np.ndarray, tokenizer: CharTokenizer, alphabet: np.ndarray
    ) -> None:
        self.alphabet_size = -100
        self.max_len = -100
        self.tokens = {
            "TOK_PAD": -100,
            "TOK_START": -100,
            "TOK_STOP": -100,
            "TOK_MASK": -100,
        }

    def transform(self, residues: str) -> np.ndarray:
        pass

    def get_size(self) -> int:
        return self.alphabet_size

    def get_ctrl_tokens(self) -> dict[str, int]:
        return self.tokens

    def get_max_seq_len(self) -> int:
        return self.max_len


class ResiduesInputBasic(InputGeneratorBase):
    """Process residue sequences into a tokenized array with padding"""

    def __init__(
        self,
        residues: np.ndarray,
        tokenizer: CharTokenizer,
        alphabet: np.ndarray,
        apply_start: bool = True,
        apply_stop: bool = True,
    ) -> None:
        """
        Args:
            residues: np.ndarray
                Array of residue sequences
            tokenizer: CharTokenizer
                Tokenizer for separating residue sequences into tokens
            alphabet: np.ndarray
                Array of SORTED unique tokens
            apply_start: bool
                Shift the tokenized sequence by one position to the right
                    using a start token
            apply_stop: bool
                Add the stop token to the tokenized sequence

        Notes:
            Converts a residue sequence into a right-padded sequence of tokens. The padding token
            is taken as the length of the alphabet.

            Shifting example:
            Given a sequence of tokens with padding token 0:
                [A, B, C, 0, 0, 0]
            Shifting adds a start token and shifts the sequence to the right:
                [<start>, A, B, C, 0, 0]
            The corresponding target for this sequence will be:
                [A, B, C, <EOS>, 0, 0]
            Note that the lengths of both sequences are the same. The EOS token is only used in the target
                generator. For consistency between the two, the tokens are:
                    pad: len(alphabet)
                    start: len(alphabet) + 1
                    stop: len(alphabet) + 2
                    mask: len(alphabet) + 3
        """
        self.tokenizer = tokenizer
        self.max_len = look_ahead_residues(residues, self.tokenizer) + 10  # buffer
        self.index_map = {char: i for i, char in enumerate(alphabet)}
        self.apply_start = apply_start
        self.apply_stop = apply_stop

        self.pad_token = len(alphabet)
        self.start_token = len(alphabet) + 1
        self.stop_token = len(alphabet) + 2
        self.mask_token = len(alphabet) + 3
        self.alphabet_size = len(alphabet) + 4  # Accounting for all tokens

        # Dictionary for keeping track of all tokens
        self.tokens = {
            "TOK_PAD": self.pad_token,
            "TOK_START": self.start_token,
            "TOK_STOP": self.stop_token,
            "TOK_MASK": self.mask_token,
        }

    def transform(self, residues: str) -> np.ndarray:
        residues = str(residues)  # Type cast for safety
        tokenized_res = self.tokenizer.tokenize(residues)
        tokenized_res = [self.index_map[char] for char in tokenized_res]
        if self.apply_start:
            tokenized_res = [self.start_token] + tokenized_res
        if self.apply_stop:
            tokenized_res = tokenized_res + [self.stop_token]
        # Pad to the maximum length
        tokenized_res = tokenized_res + [self.pad_token] * (
            self.max_len - len(tokenized_res)
        )
        return np.array(tokenized_res)
