import torch

def aggregate(base, states, sizes, defend=False):
    keys=list(base)
    updates=torch.stack([torch.cat([(s[k].to(base[k].device)-base[k]).flatten() for k in keys]) for s in states])
    finite=torch.isfinite(updates).all(dim=1)
    valid=updates[finite]
    if not len(valid): raise ValueError('No finite client updates')
    ref=valid.median(dim=0).values
    norms=valid.norm(dim=1); med=norms.median()
    cosine=torch.nn.functional.cosine_similarity(valid,ref.expand_as(valid),dim=1)
    def mad(x): return (x-x.median()).abs().median()
    bad=(norms>med+torch.maximum(3.5*1.4826*mad(norms),0.35*med+1e-8)) | (cosine<cosine.median()-torch.maximum(3.5*1.4826*mad(cosine),torch.tensor(.12,device=cosine.device)))
    flags=(~finite).clone(); flags[finite]=bad
    if defend:
        accepted=~flags
        if accepted.sum()<6:
            delta=ref
        else:
            weights=torch.tensor(sizes,device=updates.device,dtype=updates.dtype)[accepted]
            selected=updates[accepted]; scale=(med*1.5/selected.norm(dim=1).clamp_min(1e-8)).clamp(max=1)
            delta=((selected*scale[:,None])*weights[:,None]).sum(0)/weights.sum()
    else:
        weights=torch.tensor(sizes,device=updates.device,dtype=updates.dtype)[finite]
        delta=(valid*weights[:,None]).sum(0)/weights.sum()
    out={}; pos=0
    for k in keys:
        n=base[k].numel(); out[k]=base[k]+delta[pos:pos+n].reshape_as(base[k]); pos+=n
    return out, flags.nonzero().flatten().tolist() if defend else []
