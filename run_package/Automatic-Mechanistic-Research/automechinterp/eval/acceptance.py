"""Real-checkpoint acceptance checks, separate from scientific outcome scores."""
from __future__ import annotations

import torch

from ..tools import adapter
from .causal_measurements import ActivationBank, ablate_heads, score_contrast, ForwardCounter


def run_acceptance(handle, tolerance=1e-4):
    prompt='When John and Mary went to the store, John gave a drink to'
    corrupted='When Mary and John went to the store, Mary gave a drink to'
    positive,negative='Mary','John'
    checks=[]
    def record(name, passed, **evidence):
        checks.append({'name':name,'passed':bool(passed),**evidence})
    with ForwardCounter(handle.model) as counter:
        baseline=score_contrast(handle,prompt,positive,negative)
        record('finite_full_continuation_metric',True,metric=baseline)
        bank=ActivationBank.capture(handle,prompt)
        for kind in ('residual','block_update','mlp','head'):
            for layer in (0,handle.n_layers//2,handle.n_layers-1):
                patched=score_contrast(handle,prompt,positive,negative,
                          lambda:bank.patch(kind,layer,bank.prompt_length-1,head=0))
                err=abs(patched['margin']-baseline['margin'])
                record(f'self_patch_{kind}_L{layer}',err<=tolerance,error=err,tolerance=tolerance)
        other=handle.tokenizer.encode(corrupted,add_special_tokens=True)
        cross=score_contrast(handle,corrupted,positive,negative,
                            lambda:bank.patch('head',handle.n_layers-1,len(other)-1,head=0))
        record('changed_prompt_patch',torch.isfinite(torch.tensor(cross['margin'])),metric=cross)
        changed=score_contrast(handle,prompt,positive,negative,
                               lambda:ablate_heads(handle,[(0,0),(handle.n_layers-1,0)]))
        restored=score_contrast(handle,prompt,positive,negative)
        record('joint_ablation_and_restoration',abs(restored['margin']-baseline['margin'])<=tolerance,
               ablated_margin=changed['margin'],restored_margin=restored['margin'])
        multi=score_contrast(handle,'The name of the city is','New York','New Jersey')
        record('multi_token_candidates',multi['positive_tokens']>1 and multi['negative_tokens']>1,metric=multi)
        projection=adapter._find_attn_out_proj(handle.layers[handle.n_layers//2])
        leaf={}
        def gradient_hook(module,args,kwargs):
            x=(args[0] if args else kwargs['input']).detach().clone().requires_grad_(True)
            leaf['x']=x
            return ((x,)+args[1:],kwargs) if args else (args,{**kwargs,'input':x})
        h=projection.register_forward_pre_hook(gradient_hook,with_kwargs=True)
        try:
            batch=handle.tokenizer([prompt],return_tensors='pt').to(handle.device)
            output=handle.model(**batch,use_cache=False).logits
            target=handle.tokenizer.encode(' Mary',add_special_tokens=False)[0]
            grad,=torch.autograd.grad(output[0,-1,target],leaf['x'])
            record('head_gradient',torch.isfinite(grad).all() and grad.abs().sum()>0,
                   absolute_gradient_sum=float(grad.abs().sum()))
        finally:
            h.remove()
        all_modules=list(handle.model.modules())
        # ForwardCounter's one model-level hook is expected until context exit.
        remaining=sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in all_modules)-1
        record('all_intervention_hooks_removed',remaining==0,remaining=remaining)
    return {'status':'passed' if all(c['passed'] for c in checks) else 'failed',
            'model':adapter.profile_network(handle),'checks':checks,'telemetry':counter.as_dict(),
            'scope':'real-checkpoint engineering acceptance, not circuit ground-truth validation'}
