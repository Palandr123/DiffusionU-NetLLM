import torch
import torchvision
import numpy as np
import torchvision.transforms.functional


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
    mask: torch.Tensor,
    idxs: np.ndarray[np.int64] | None = None,
    ):
    attn = aux[i]
    target_attn = target_aux[i].to(attn.device)

    if idxs is not None:
        attn = attn[..., idxs]
        target_attn = target_attn[..., idxs]
    x1, y1, x2, y2 = box_orig
    mask_cropped = torchvision.transforms.functional.resize(mask, attn.shape[-1])[..., y1:y2, x1:x2].to(attn.device)
    mask_cropped = (mask_cropped >= 0.4).float()
    return (((attn[:,:,y1:y2, x1:x2]-target_attn[:,:,y1:y2, x1:x2])*mask_cropped)**2).mean()