def shared_eval_autoreg(lightning_module, batch, batch_idx, prefix):
    """Autoregressive shared_eval has no loss masking

    Args:
        lightning_module: LightningModule instance with model and loss_fn
        batch: Tuple of: (see transformer_sharded_autoreg_collate_fn() in dataset.py)
            struct: Tensor of shape (B, L_struct) with structure tokens
            src: Tensor of shape (B, L_src) with input tokens (L_struct == L_src)
            src_key_pad_mask: Bool tensor of shape (B, L_src), where True marks padding
            tgt: Tensor of shape (B, L_tgt) with target tokens
            tgt_key_pad_mask: Bool tensor of shape (B, L_tgt), where True marks padding
            seq_id: Tensor of shape (B, L_seq) with sequence identifiers / positions
        batch_idx: Batch index
        prefix: Logging prefix ("train", "validation", or "test")

    Returns:
        loss: Mean loss for the batch
    """
    struct, src, src_key_pad_mask, tgt, tgt_key_pad_mask, seq_id = batch
    out = lightning_module.model(
        src, struct, seq_id
    )  # (B, L_src, C) where C = vocab size, and L_src == L_struct
    out = out.permute(
        0, 2, 1
    )  # (B, L_src, C) -> (B, C, L_src) for CrossEntropyLoss which expects shape (N, C, ...)
    loss = lightning_module.loss_fn(out, tgt)
    loss = loss.mean()
    metrics = {f"{prefix}/loss": loss}
    lightning_module.log_dict(metrics, on_step=True, on_epoch=True, sync_dist=True)
    return loss
