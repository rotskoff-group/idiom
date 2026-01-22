import numpy as np
from .tokenizer import BasicSmilesTokenizer
from mole.utils.structure import compute_dihedrals


def look_ahead_smiles(smiles: list[str], tokenizer: BasicSmilesTokenizer) -> int:
    """Determines the maximum length of the smiles strings in numbers of tokens

    Args:
        smiles: list[str]
            List of SMILES strings
        tokenizer: BasicSmilesTokenizer
            Instance of the BasicSmilesTokenizer class for tokenizing the SMILES strings

    Returns:
        int: The maximum length of the tokenized SMILES strings
    """
    max_len = 0
    for i in range(len(smiles)):
        tokens = tokenizer.tokenize(smiles[i])
        max_len = max(max_len, len(tokens))
    # To account for additional stop token
    return max_len + 1


def look_ahead_dihedral(smiles: list[str]) -> int:
    """Determines the maximum number of dihedral angles computable over the SMILES strings

    Args:
        smiles: list[str]
            The list of SMILES strings to compute the dihedral angles over

    Returns:
        int: The maximum number of dihedral angles that can be computed over the SMILES strings
    """
    max_len = 0
    for i in range(len(smiles)):
        angles, _ = compute_dihedrals(smiles[i])
        max_len = max(max_len, len(angles))
    return max_len + 1


# Base class for input generators, to be inherited by others
class InputGeneratorBase:
    # Getters have concrete implementations, but constructor and transform are not implemented
    def __init__(
        self, smiles: np.ndarray, tokenizer: BasicSmilesTokenizer, alphabet: np.ndarray
    ) -> None:
        self.alphabet_size = -100
        self.max_len = -100
        self.tokens = {
            "TOK_PAD": -100,
            "TOK_START": -100,
            "TOK_STOP": -100,
            "TOK_MASK": -100,
        }

    def transform(self, smiles: str) -> np.ndarray:
        pass

    def get_size(self) -> int:
        return self.alphabet_size

    def get_ctrl_tokens(self) -> dict[str, int]:
        return self.tokens

    def get_max_seq_len(self) -> int:
        return self.max_len


class SMILESInputBasic(InputGeneratorBase):
    """Process SMILES strings into a tokenized array with padding"""

    def __init__(
        self,
        smiles: np.ndarray,
        tokenizer: BasicSmilesTokenizer,
        alphabet: np.ndarray,
        apply_start: bool = True,
        apply_stop: bool = True,
    ) -> None:
        """
        Args:
            smiles: np.ndarray
                Array of SMILES strings
            tokenizer: BasicSmilesTokenizer
                Tokenizer for separating SMILES strings into tokens
            alphabet: np.ndarray
                Array of SORTED unique tokens
            apply_start: bool
                Shift the tokenized sequence by one position to the right
                    using a start token
            apply_stop: bool
                Add the stop token to the tokenized sequence

        Notes:
            Converts a SMILES string into a right-padded sequence of tokens. The padding token
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
        self.max_len = look_ahead_smiles(smiles, self.tokenizer) + 10  # buffer
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

    def transform(self, smiles: str) -> np.ndarray:
        smiles = str(smiles)  # Type cast for safety
        tokenized_smiles = self.tokenizer.tokenize(smiles)
        tokenized_smiles = [self.index_map[char] for char in tokenized_smiles]
        if self.apply_start:
            tokenized_smiles = [self.start_token] + tokenized_smiles
        if self.apply_stop:
            tokenized_smiles = tokenized_smiles + [self.stop_token]
        # Pad to the maximum length
        tokenized_smiles = tokenized_smiles + [self.pad_token] * (
            self.max_len - len(tokenized_smiles)
        )
        return np.array(tokenized_smiles)


class SMILESAndStructureInput(InputGeneratorBase):
    def __init__(
        self, smiles: np.ndarray, tokenizer: BasicSmilesTokenizer, alphabet: np.ndarray
    ) -> None:
        """
        See SMILESInputBasic for details on the arguments

        In addition to generating the tokenized smiles, this generator class also generates the
        dihedral angles of the molecule over the CANONICALIZED SMILES string. The canonicalization
        is critical because it determines the atomic indices that the dihedral maps on to in the
        molecule.

        This method also has two additional tokens for encoding structural portions of the input,
        which are added to the class's token dictionary

        This is intended to be used with the transfusion model architecture, as described in
        https://www.arxiv.org/abs/2408.11039

        The format of the input is as follows:
        <SMILES> <STRUCT_START> <STRUCT> <STRUCT_END>

        One needs to be careful that here, <STURUCT_START> takes the place of the STOP token
        for the SMILES sequence and shifting needs to be carefully done to account for this.
        """
        self.tokenizer = tokenizer
        # Track lengths separately for padding separate information
        self.max_tok_len = look_ahead_smiles(smiles, self.tokenizer)
        self.max_dihedral_len = look_ahead_dihedral(smiles)
        # Maximum length of the combined input sequence
        self.max_len = self.max_tok_len + self.max_dihedral_len + 10  # buffer
        self.index_map = {char: i for i, char in enumerate(alphabet)}

        self.pad_token = len(alphabet)
        self.start_token = len(alphabet) + 1
        self.stop_token = len(alphabet) + 2
        self.mask_token = len(alphabet) + 3
        self.struct_token = len(alphabet) + 4
        self.struct_start_token = len(alphabet) + 5
        self.struct_stop_token = len(alphabet) + 6
        self.alphabet_size = len(alphabet) + 7  # Accounting for all tokens

        self.tokens = {
            "TOK_PAD": self.pad_token,
            "TOK_START": self.start_token,
            "TOK_STOP": self.stop_token,
            "TOK_MASK": self.mask_token,
            "STRUCT": self.struct_token,
            "STRUCT_START": self.struct_start_token,
            "STRUCT_END": self.struct_stop_token,
            "STRUCT_PAD": -100,
        }

    # OLD TRANSFORM METHOD
    # def transform(self, smiles: str) -> tuple[np.ndarray, list[float], np.ndarray]:
    #     """
    #     Returns both the tokenized SMILES string with the dihedral spacer tokens and the
    #     dihedral angles over the canonicalized indices of the molecule
    #     """
    #     smiles = str(smiles) #Type cast for safety
    #     tokenized_input = self.tokenizer.tokenize(smiles)
    #     tokenized_input = [self.index_map[char] for char in tokenized_input]
    #     #Compute the dihedral information
    #     dihedral_angles, dihedral_indices = compute_dihedrals(smiles)
    #     #Add in a spacing of padding tokens to account for dihedral elements
    #     structure_block = [self.struct_token] * len(dihedral_angles) + [self.struct_stop_token]
    #     #Target tokens shifted by one position, but only for the smiles sequence block
    #     tokens_for_input = [self.start_token] + tokenized_input + structure_block
    #     tokens_for_target = tokenized_input + [self.struct_start_token] + structure_block
    #     assert len(tokens_for_input) == len(tokens_for_target)
    #     # if self.apply_start:
    #     #     tokenized_input = [self.start_token] + tokenized_input
    #     # if self.apply_stop:
    #     #     tokenized_input = tokenized_input + [self.stop_token]
    #     tokens_for_input = tokens_for_input + [self.pad_token] * (self.max_len - len(tokens_for_input))
    #     tokens_for_target = tokens_for_target + [self.pad_token] * (self.max_len - len(tokens_for_target))
    #     #Pad the dihedral angles too with -100
    #     dihedral_angles = dihedral_angles + [-100] * (self.max_dihedral_len - len(dihedral_angles))
    #     #Pad + process the dihedral indices as well
    #     if dihedral_indices == []:
    #         dihedral_indices = np.ones((self.max_dihedral_len, 4)) * -100
    #     else:
    #         dihedral_indices = np.array(dihedral_indices)
    #         assert dihedral_indices.shape[1] == 4
    #         assert self.max_dihedral_len >= dihedral_indices.shape[0]
    #         dihedral_indices = np.pad(dihedral_indices,
    #                                 ((0, self.max_dihedral_len - dihedral_indices.shape[0]), (0, 0)),
    #                                 mode='constant',
    #                                 constant_values=-100)
    #     return (tokens_for_input, tokens_for_target, dihedral_angles, dihedral_indices)

    def transform(self, smiles: str) -> tuple[np.ndarray, list[float], np.ndarray]:
        """
        Returns both the tokenized SMILES string with the dihedral spacer tokens and the
        dihedral angles over the canonicalized indices of the molecule
        """
        smiles = str(smiles)  # Type cast for safety
        tokenized_input = self.tokenizer.tokenize(smiles)
        tokenized_input = [self.index_map[char] for char in tokenized_input]
        # Compute the dihedral information
        dihedral_angles, dihedral_indices = compute_dihedrals(smiles)
        # Add in a spacing of padding tokens to account for dihedral elements
        structure_block = [self.struct_token] * len(dihedral_angles) + [
            self.struct_stop_token
        ]
        # Add both the sequence start and structure start tokens into the sequence,
        #   will be careful to mask out later
        # <TOK_START><SMILES><STRUCT_START><STRUCT><STRUCT_END>
        tokenized_input = (
            [self.start_token]
            + tokenized_input
            + [self.struct_start_token]
            + structure_block
        )
        # Pad to the appropriate length
        tokenized_input = tokenized_input + [self.pad_token] * (
            self.max_len - len(tokenized_input)
        )

        # Pad the dihedral angles too with -100
        dihedral_angles = dihedral_angles + [-100] * (
            self.max_dihedral_len - len(dihedral_angles)
        )
        # Pad + process the dihedral indices as well
        if dihedral_indices == []:
            dihedral_indices = np.ones((self.max_dihedral_len, 4)) * -100
        else:
            dihedral_indices = np.array(dihedral_indices)
            assert dihedral_indices.shape[1] == 4
            assert self.max_dihedral_len >= dihedral_indices.shape[0]
            dihedral_indices = np.pad(
                dihedral_indices,
                ((0, self.max_dihedral_len - dihedral_indices.shape[0]), (0, 0)),
                mode="constant",
                constant_values=-100,
            )
        return (tokenized_input, dihedral_angles, dihedral_indices)


class SMILESAndTokenizedStructureInput(InputGeneratorBase):
    def __init__(
        self,
        smiles: np.ndarray,
        tokenizer: BasicSmilesTokenizer,
        alphabet: np.ndarray,
        struct_token_info: dict,
    ) -> None:
        assert "struct_pad" in struct_token_info
        assert "struct_mask" in struct_token_info
        assert "struct_stop" in struct_token_info
        assert "struct_start" in struct_token_info
        self.struct_token_info = struct_token_info
        self.tokenizer = tokenizer
        self.max_tok_len = look_ahead_smiles(smiles, self.tokenizer)
        # FH: A bit of a hack, but assume that there are at most twice as many structural tokens as there are SMILES tokens
        #   for any given molecule
        self.max_len = self.max_tok_len * 3
        self.index_map = {char: i for i, char in enumerate(alphabet)}

        # Control tokens for SMILES sequence
        self.pad_token = len(alphabet)
        self.start_token = len(alphabet) + 1
        self.stop_token = len(alphabet) + 2
        self.mask_token = len(alphabet) + 3
        self.struct_token = len(alphabet) + 4
        self.struct_start_token = len(alphabet) + 5
        self.struct_stop_token = len(alphabet) + 6
        self.struct_mask_token = len(alphabet) + 7
        self.alphabet_size = len(alphabet) + 8  # Accounting for all tokens

        self.tokens = {
            "TOK_PAD": self.pad_token,
            "TOK_START": self.start_token,
            "TOK_STOP": self.stop_token,
            "TOK_MASK": self.mask_token,
            "TOK_MAX_SIZE": self.alphabet_size,  # Add some additional tokens to prevent index out of bounds
            "STRUCT": self.struct_token,
            "STRUCT_START": self.struct_token_info[
                "struct_start"
            ],  # Start token for the structure tokens
            "STRUCT_STOP": self.struct_token_info[
                "struct_stop"
            ],  # Stop token for the structure tokens
            "STRUCT_MASK": self.struct_token_info[
                "struct_mask"
            ],  # Mask token for the structure tokens
            "STRUCT_PAD": self.struct_token_info[
                "struct_pad"
            ],  # Pad token for the structure tokens
            "STRUCT_MAX_SIZE": max(self.struct_token_info.values()),
        }

    def transform(
        self, smiles: str, struct_tokens: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # All this does is create a SMILES token sequence with placeholders for the structure tokens
        smiles = str(smiles)  # Type cast for safety
        tokenized_input = self.tokenizer.tokenize(smiles)
        tokenized_input = [self.index_map[char] for char in tokenized_input]
        # Compute structure token block based on the structure padding token
        n_struct_tokens = (struct_tokens != self.struct_token_info["struct_pad"]).sum()
        structure_block = [self.struct_token] * n_struct_tokens
        tokenized_input = (
            [self.start_token] + tokenized_input + [self.stop_token] + structure_block
        )
        # Pad to the appropriate length
        tokenized_input = tokenized_input + [self.pad_token] * (
            self.max_len - len(tokenized_input)
        )
        return (tokenized_input, struct_tokens)


class SMILESStructureAVHGenerator(InputGeneratorBase):
    def __init__(
        self,
        smiles: np.ndarray,
        tokenizer: BasicSmilesTokenizer,
        alphabet: np.ndarray,
        struct_token_info: dict,
    ) -> None:
        assert "struct_pad" in struct_token_info
        assert "struct_mask" in struct_token_info
        assert "struct_stop" in struct_token_info
        assert "struct_start" in struct_token_info
        assert "struct_max_length" in struct_token_info
        assert "atom_padding_idx" in struct_token_info
        assert "valency_padding_idx" in struct_token_info
        assert "hybrid_padding_idx" in struct_token_info

        self.struct_token_info = struct_token_info
        self.tokenizer = tokenizer
        self.max_tok_len = look_ahead_smiles(smiles, self.tokenizer)
        self.max_len = (
            self.max_tok_len + (2 * self.struct_token_info["struct_max_length"]) + 10
        )
        self.index_map = {char: i for i, char in enumerate(alphabet)}

        # Control tokens for SMILES sequence
        self.pad_token = len(alphabet)
        self.start_token = len(alphabet) + 1
        self.stop_token = len(alphabet) + 2
        self.mask_token = len(alphabet) + 3
        self.struct_token = len(alphabet) + 4
        self.struct_start_token = len(alphabet) + 5
        self.struct_stop_token = len(alphabet) + 6
        self.struct_mask_token = len(alphabet) + 7
        self.alphabet_size = len(alphabet) + 8  # Accounting for all tokens

        self.tokens = {
            "TOK_PAD": self.pad_token,
            "TOK_START": self.start_token,
            "TOK_STOP": self.stop_token,
            "TOK_MASK": self.mask_token,
            "TOK_MAX_SIZE": self.alphabet_size,  # Add some additional tokens to prevent index out of bounds
            "STRUCT": self.struct_token,
            "STRUCT_START": self.struct_token_info[
                "struct_start"
            ],  # Start token for the structure tokens
            "STRUCT_STOP": self.struct_token_info[
                "struct_stop"
            ],  # Stop token for the structure tokens
            "STRUCT_MASK": self.struct_token_info[
                "struct_mask"
            ],  # Mask token for the structure tokens
            "STRUCT_PAD": self.struct_token_info[
                "struct_pad"
            ],  # Pad token for the structure tokens
            "STRUCT_MAX_SIZE": max(self.struct_token_info.values()),
        }

    def transform(
        self, smiles: str, struct_tokens: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # Here, the structure token is a 4xT array, where T is the number of HEAVY atoms in the molecule.
        # It is expected that the order of the 4 rows is as follows:
        #  1. Structure token
        #  2. Atom class token (0 to n_atom_classes)
        #  3. Valency class token (0 to n_valency_classes)
        #  4. Hybridization class token (0 to n_hybrid_classes)
        # This transformation interleaves each token sequence with that sequence's associated padding token.
        # For example, assume you have 4 atoms and their associated elements, valencies, and hybridizations.
        # Then, the final structure token sequence looks like:
        #  [ P, S0,   P, S1,  P, S2,  P, S3]
        #  [A0,  P,  A1,  P, A2,  P, A3,  P]
        #  [V0,  P,  V1,  P, V2,  P, V3,  P]
        #  [H0,  P,  H1,  P, H2,  P, H3,  P]
        # Where the P in each row is the padding token for that row and maps to a 0 vector upon embedding.
        # This 4xT array is then padded to the maximum length with the padding tokens for each row.
        # The structure padding token is defined, but the atom, valency, and hybridization padding tokens are
        #   assumed to be the number of classes for that token.

        # Determine the number of non-padding tokens in the structure token sequence
        #   This is used to determine how many structure tokens to add to the input sequence
        n_nonpad_struct_tokens = (
            struct_tokens[0] != self.struct_token_info["struct_pad"]
        ).sum()

        # Smiles tokenization:
        smiles = str(smiles)  # Type cast for safety
        tokenized_input = self.tokenizer.tokenize(smiles)
        tokenized_input = [self.index_map[char] for char in tokenized_input]

        n_struct_tokens = n_nonpad_struct_tokens * 2
        structure_block = [self.struct_token] * n_struct_tokens
        tokenized_input = (
            [self.start_token] + tokenized_input + [self.stop_token] + structure_block
        )

        # Pad to the appropriate length
        tokenized_input = tokenized_input + [self.pad_token] * (
            self.max_len - len(tokenized_input)
        )

        # Structure tokenization:
        structure_tokens = struct_tokens[0, :n_nonpad_struct_tokens]
        atom_tokens = struct_tokens[1, :n_nonpad_struct_tokens]
        valency_tokens = struct_tokens[2, :n_nonpad_struct_tokens]
        hybridization_tokens = struct_tokens[3, :n_nonpad_struct_tokens]

        # Interleave each one correctly
        # We are going to use masking tokens for interleaving as those are easier to deal with later in the
        #   training process
        s_new = (
            np.ones(len(structure_tokens) * 2) * self.struct_token_info["struct_mask"]
        )
        s_new[1::2] = structure_tokens
        a_new = (
            np.ones(len(structure_tokens) * 2)
            * self.struct_token_info["atom_padding_idx"]
        )
        a_new[0::2] = atom_tokens
        v_new = (
            np.ones(len(structure_tokens) * 2)
            * self.struct_token_info["valency_padding_idx"]
        )
        v_new[0::2] = valency_tokens
        h_new = (
            np.ones(len(structure_tokens) * 2)
            * self.struct_token_info["hybrid_padding_idx"]
        )
        h_new[0::2] = hybridization_tokens

        # Pad to the appropriate length
        s_new = np.pad(
            s_new,
            (0, self.max_len - len(s_new)),
            "constant",
            constant_values=self.struct_token_info["struct_pad"],
        )
        a_new = np.pad(
            a_new,
            (0, self.max_len - len(a_new)),
            "constant",
            constant_values=self.struct_token_info["atom_padding_idx"],
        )
        v_new = np.pad(
            v_new,
            (0, self.max_len - len(v_new)),
            "constant",
            constant_values=self.struct_token_info["valency_padding_idx"],
        )
        h_new = np.pad(
            h_new,
            (0, self.max_len - len(h_new)),
            "constant",
            constant_values=self.struct_token_info["hybrid_padding_idx"],
        )

        struct_total = np.vstack((s_new, a_new, v_new, h_new))
        return (tokenized_input, struct_total)


class IDAInputGenerator(InputGeneratorBase):
    def __init__(
        self,
        smiles: np.ndarray,
        tokenizer: BasicSmilesTokenizer,
        alphabet: np.ndarray,
        struct_token_info: dict,
    ) -> None:
        assert "struct_pad" in struct_token_info
        assert "struct_mask" in struct_token_info
        assert "struct_stop" in struct_token_info
        assert "struct_start" in struct_token_info
        assert "struct_max_length" in struct_token_info
        assert "atom_padding_idx" in struct_token_info
        assert "valency_padding_idx" in struct_token_info
        assert "hybrid_padding_idx" in struct_token_info

        self.struct_token_info = struct_token_info
        self.tokenizer = tokenizer
        self.max_tok_len = look_ahead_smiles(smiles, self.tokenizer)
        self.max_len = (
            self.max_tok_len + (2 * self.struct_token_info["struct_max_length"]) + 10
        )
        self.index_map = {char: i for i, char in enumerate(alphabet)}

        # Control tokens for SMILES sequence
        self.pad_token = len(alphabet)
        self.start_token = len(alphabet) + 1
        self.stop_token = len(alphabet) + 2
        self.mask_token = len(alphabet) + 3
        self.struct_token = len(alphabet) + 4
        self.struct_start_token = len(alphabet) + 5
        self.struct_stop_token = len(alphabet) + 6
        self.struct_mask_token = len(alphabet) + 7
        self.alphabet_size = len(alphabet) + 12  # Accounting for all tokens

        self.tokens = {
            "TOK_PAD": self.pad_token,
            "TOK_START": self.start_token,
            "TOK_STOP": self.stop_token,
            "TOK_MASK": self.mask_token,
            "TOK_MAX_SIZE": self.alphabet_size,  # Add some additional tokens to prevent index out of bounds
            "STRUCT": self.struct_token,
            "STRUCT_START": self.struct_token_info[
                "struct_start"
            ],  # Start token for the structure tokens
            "STRUCT_STOP": self.struct_token_info[
                "struct_stop"
            ],  # Stop token for the structure tokens
            "STRUCT_MASK": self.struct_token_info[
                "struct_mask"
            ],  # Mask token for the structure tokens
            "STRUCT_PAD": self.struct_token_info[
                "struct_pad"
            ],  # Pad token for the structure tokens
            "STRUCT_MAX_SIZE": max(self.struct_token_info.values()),
        }

    def transform(
        self, smiles: str, struct_tokens: np.ndarray, coords: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # Here, the structure token is a 4xT array, where T is the number of HEAVY atoms in the molecule.
        # It is expected that the order of the 4 rows is as follows:
        #  1. Structure token
        #  2. Atom class token (0 to n_atom_classes)
        #  3. Valency class token (0 to n_valency_classes)
        #  4. Hybridization class token (0 to n_hybrid_classes)
        # This transformation interleaves each token sequence with that sequence's associated padding token.
        # For example, assume you have 4 atoms and their associated elements, valencies, and hybridizations.
        # Then, the final structure token sequence looks like:
        #  [ P, S0,   P, S1,  P, S2,  P, S3]
        #  [A0,  P,  A1,  P, A2,  P, A3,  P]
        #  [V0,  P,  V1,  P, V2,  P, V3,  P]
        #  [H0,  P,  H1,  P, H2,  P, H3,  P]
        # Where the P in each row is the padding token for that row and maps to a 0 vector upon embedding.
        # This 4xT array is then padded to the maximum length with the padding tokens for each row.
        # The structure padding token is defined, but the atom, valency, and hybridization padding tokens are
        #   assumed to be the number of classes for that token.
        # The coordinates are assumed to have shape (T, 3) where T is the number of heavy atoms

        # Determine the number of non-padding tokens in the structure token sequence
        #   This is used to determine how many structure tokens to add to the input sequence
        n_nonpad_struct_tokens = (
            struct_tokens[0] != self.struct_token_info["struct_pad"]
        ).sum()

        # Smiles tokenization:
        smiles = str(smiles)  # Type cast for safety
        tokenized_input = self.tokenizer.tokenize(smiles)
        tokenized_input = [self.index_map[char] for char in tokenized_input]

        # coordinate entries should be padded to the length of the tokenized input
        assert np.sum(~np.all(coords == -np.inf, axis=1)) == n_nonpad_struct_tokens, (
            "The number of non-padding coordinate rows must match the number of non-padding structure tokens"
        )

        # FH: These two variables, n_struct_tokens and tokenized_input_len, control the
        # number of positions in the total sequence, as the full sequence length is
        # 2 + len(tokenized_input) + n_struct_tokens
        _n_struct_tokens = n_nonpad_struct_tokens * 2
        # FH: Add 2 for start, stop tokens
        _tokenized_input_len = 2 + len(tokenized_input)
        structure_block = [self.struct_token] * _n_struct_tokens
        tokenized_input = (
            [self.start_token] + tokenized_input + [self.stop_token] + structure_block
        )
        assert len(tokenized_input) == _tokenized_input_len + _n_struct_tokens

        # Pad to the appropriate length
        tokenized_input = tokenized_input + [self.pad_token] * (
            self.max_len - len(tokenized_input)
        )
        _padded_total_length = len(tokenized_input)

        # Structure tokenization:
        structure_tokens = struct_tokens[0, :n_nonpad_struct_tokens]
        atom_tokens = struct_tokens[1, :n_nonpad_struct_tokens]
        valency_tokens = struct_tokens[2, :n_nonpad_struct_tokens]
        hybridization_tokens = struct_tokens[3, :n_nonpad_struct_tokens]

        # Interleave each one correctly
        # We are going to use masking tokens for interleaving as those are easier to deal with later in the
        #   training process
        s_new = (
            np.ones(len(structure_tokens) * 2) * self.struct_token_info["struct_mask"]
        )
        s_new[1::2] = structure_tokens
        a_new = (
            np.ones(len(structure_tokens) * 2)
            * self.struct_token_info["atom_padding_idx"]
        )
        a_new[0::2] = atom_tokens
        v_new = (
            np.ones(len(structure_tokens) * 2)
            * self.struct_token_info["valency_padding_idx"]
        )
        v_new[0::2] = valency_tokens
        h_new = (
            np.ones(len(structure_tokens) * 2)
            * self.struct_token_info["hybrid_padding_idx"]
        )
        h_new[0::2] = hybridization_tokens

        # Pad to the appropriate length
        s_new = np.pad(
            s_new,
            (0, self.max_len - len(s_new)),
            "constant",
            constant_values=self.struct_token_info["struct_pad"],
        )
        a_new = np.pad(
            a_new,
            (0, self.max_len - len(a_new)),
            "constant",
            constant_values=self.struct_token_info["atom_padding_idx"],
        )
        v_new = np.pad(
            v_new,
            (0, self.max_len - len(v_new)),
            "constant",
            constant_values=self.struct_token_info["valency_padding_idx"],
        )
        h_new = np.pad(
            h_new,
            (0, self.max_len - len(h_new)),
            "constant",
            constant_values=self.struct_token_info["hybrid_padding_idx"],
        )
        struct_total = np.vstack((s_new, a_new, v_new, h_new))

        nonpad_coords = coords[np.all(np.isfinite(coords), axis=1)]

        # FH: Pad the coordinates here on both ends with -inf
        coords_padded = np.empty([_padded_total_length, 3])
        coords_padded.fill(-np.inf)
        coords_padded[
            _tokenized_input_len + 1 : _tokenized_input_len + _n_struct_tokens : 2,
            :,
        ] = nonpad_coords

        return (tokenized_input, struct_total, coords_padded)
