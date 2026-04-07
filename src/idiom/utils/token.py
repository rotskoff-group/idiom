def aggregate_tokens_hdf5(hdf5_ptr):
    """
    Combines all the tokens contained within an hdf5 file into one consistent object
    to be used throughout
    """
    token_dict = {}
    if "input_metadata" in hdf5_ptr.keys():
        token_dict["input"] = {"TOK": {}, "STRUCT": {}}
        for k in hdf5_ptr["input_metadata"]["ctrl_tokens"].keys():
            if "TOK" in k:
                token_dict["input"]["TOK"][k] = hdf5_ptr["input_metadata"][
                    "ctrl_tokens"
                ][k][()]
            elif "STRUCT" in k:
                token_dict["input"]["STRUCT"][k] = hdf5_ptr["input_metadata"][
                    "ctrl_tokens"
                ][k][()]
        if (len(token_dict["input"]["TOK"].keys()) > 0) and (
            "TOK_MAX_SIZE" not in token_dict["input"]["TOK"].keys()
        ):
            token_dict["input"]["TOK"]["TOK_MAX_SIZE"] = max(
                token_dict["input"]["TOK"].values()
            )
        if (len(token_dict["input"]["STRUCT"].keys()) > 0) and (
            "STRUCT_MAX_SIZE" not in token_dict["input"]["STRUCT"].keys()
        ):
            token_dict["input"]["STRUCT"]["STRUCT_MAX_SIZE"] = max(
                token_dict["input"]["STRUCT"].values()
            )
        inp_size = hdf5_ptr["input_metadata"]["source_size"][()]
    else:
        inp_size = -1
    if "target_metadata" in hdf5_ptr.keys():
        token_dict["target"] = {"TOK": {}, "STRUCT": {}}
        for k in hdf5_ptr["target_metadata"]["ctrl_tokens"].keys():
            if "TOK" in k:
                token_dict["target"]["TOK"][k] = hdf5_ptr["target_metadata"][
                    "ctrl_tokens"
                ][k][()]
            elif "STRUCT" in k:
                token_dict["target"]["STRUCT"][k] = hdf5_ptr["target_metadata"][
                    "ctrl_tokens"
                ][k][()]
        if (len(token_dict["target"]["TOK"].keys()) > 0) and (
            "TOK_MAX_SIZE" not in token_dict["target"]["TOK"].keys()
        ):
            token_dict["target"]["TOK"]["TOK_MAX_SIZE"] = max(
                token_dict["target"]["TOK"].values()
            )
        if (len(token_dict["target"]["STRUCT"].keys()) > 0) and (
            "STRUCT_MAX_SIZE" not in token_dict["target"]["STRUCT"].keys()
        ):
            token_dict["target"]["STRUCT"]["STRUCT_MAX_SIZE"] = max(
                token_dict["target"]["STRUCT"].values()
            )
        tgt_size = hdf5_ptr["target_metadata"]["target_size"][()]
    else:
        tgt_size = -1
    if "alphabet" in hdf5_ptr.keys():
        token_dict["alphabet"] = hdf5_ptr["alphabet"][()]
    # Add information on total number of tokens
    token_dict["TOTAL"] = max(inp_size, tgt_size)
    return token_dict
