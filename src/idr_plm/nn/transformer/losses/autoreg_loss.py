def shared_eval_autoreg(lightning_module, batch, batch_idx, prefix):
    """Autoregressive shared_eval has no loss masking

    Args:
        lightning_module: LightningModule instance with model and loss_fn
        batch: Tuple of (struct, src, src_key_pad_mask, tgt, tgt_key_pad_mask, seq_id)
        batch_idx: Batch index
        prefix: Logging prefix ("train", "validation", or "test")

    Returns:
        loss: Mean loss for the batch
    """
    struct, src, src_key_pad_mask, tgt, tgt_key_pad_mask, seq_id = batch
    out = lightning_module.model(src, struct, seq_id)
    out = out.permute(0, 2, 1)
    loss = lightning_module.loss_fn(out, tgt)
    loss = loss.mean()
    metrics = {f"{prefix}/loss": loss}
    lightning_module.log_dict(metrics, on_step=True, on_epoch=True, sync_dist=True)
    return loss
