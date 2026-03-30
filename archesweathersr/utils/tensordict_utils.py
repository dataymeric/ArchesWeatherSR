import torch
import torch.nn.functional as F
from tensordict.tensordict import TensorDict


def tensordict_apply(f, *args, **kwargs):
    tdicts = [a for a in args if isinstance(a, TensorDict)]
    tdicts += [v for v in kwargs.values() if isinstance(v, TensorDict)]
    # check that all found tdicts have same keys
    tdict_keys = set(tdicts[0].keys())
    for tdict in tdicts[1:]:
        assert tdict_keys == set(tdict.keys()), (
            "All TensorDicts must have the same keys"
        )
    return TensorDict(
        {
            k: f(
                *[(a[k] if isinstance(a, TensorDict) else a) for a in args],
                **{
                    ki: (vi[k] if isinstance(vi, TensorDict) else vi)
                    for ki, vi in kwargs.items()
                },
            )
            for k in tdict_keys
        },
        device=tdicts[0].device,
    ).auto_batch_size_()


def tensordict_cat(tdict_list, dim=0, **kwargs):
    """
    weirdly, the tensordict library requires a strict condition for batch size,
    whereas we just need to concat tensors one by one without needing them to have exact same dimensions.
    """
    return TensorDict(
        dict(
            {
                k: torch.cat([tdict[k] for tdict in tdict_list], dim=dim, **kwargs)
                for k in tdict_list[0].keys()
            }
        ),
        device=tdict_list[0].device,
    ).auto_batch_size_()


def tensordict_interp(tdict, target, mode="bicubic", align_corners=False):
    """Upsample a TensorDict to a desired target shape."""
    if mode == "trilinear":
        # supports 5D tensor only
        return TensorDict(
            {
                k: F.interpolate(
                    tdict[k],
                    size=target[k].shape[-3:],
                    mode=mode,
                    align_corners=align_corners,
                )
                for k in tdict.keys()
            },
            device=tdict.device,
        ).auto_batch_size_()
    elif mode in ["bilinear", "bicubic"]:
        # supports 4D tensor only
        out = {}
        for k in tdict.keys():
            b, c, pl, lat, lon = tdict[k].shape
            out[k] = (
                F.interpolate(
                    tdict[k].permute(0, 2, 1, 3, 4).reshape(b * pl, c, lat, lon),
                    size=target[k].shape[-2:],
                    mode=mode,
                    align_corners=align_corners,
                )
                .reshape(b, pl, c, target[k].shape[-2], target[k].shape[-1])
                .permute(0, 2, 1, 3, 4)
            )
        return TensorDict(out, device=tdict.device).auto_batch_size_()
    else:
        raise ValueError(
            f"Unsupported mode: {mode}. Supported modes are 'trilinear', 'bilinear', 'bicubic'."
        )
