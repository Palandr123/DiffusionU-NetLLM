import torch
import numpy as np


def position(attn: torch.Tensor,
    i: int,
    box_orig: list[int],
    shift: tuple[int, int],
    idxs: np.ndarray[np.int64] | None = None,
    tgt: torch.Tensor | None = None,
    ):
    attn = attn[i]
    if idxs is not None:
        attn = attn[..., idxs]

    x1, y1, x2, y2 = box_orig
    x, y = shift
    x1_target = x1 + x
    x2_target = x2 + x
    y1_target = y1 + y
    y2_target = y2 + y
    attn_orig = attn[:,y1:y2, x1:x2, :]
    attn_target = attn[:,y1_target:y2_target, x1_target:x2_target, :]
    return (attn_orig**2).mean()-(attn_target**2).mean()


def preserve(
    aux: torch.Tensor,
    i: int,
    target_aux: torch.Tensor,
    box_orig: list[int],
    idxs: np.ndarray[np.int64] | None = None,
    ):
    attn = aux[i]
    target_attn = target_aux[i].to(attn.device)

    if idxs is not None:
        attn = attn[..., idxs]
        target_attn = target_attn[..., idxs]
    x1, y1, x2, y2 = box_orig
    return ((attn[:,:, y1:y2, x1:x2]-target_attn[:,:,y1:y2, x1:x2])**2).mean()


def preserve_background(
    aux: torch.Tensor,
    i: int,
    target_aux: torch.Tensor,
    mask: torch.Tensor,
    idxs: np.ndarray[np.int64] | None = None,
    ):
    attn = aux[i]
    target_attn = target_aux[i].to(attn.device)

    if idxs is not None:
        attn = attn[..., idxs]
        target_attn = target_attn[..., idxs]
    return ((attn * mask.to(attn.device) - target_attn * mask.to(attn.device))**2).mean()