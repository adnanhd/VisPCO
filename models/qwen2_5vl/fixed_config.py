import json
import types

import torch


def _bisect(cost, target, lo, hi, steps=40):
    """Return x in [lo, hi] with cost(x) = target for cost increasing in x."""
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if cost(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fix_configuration(module, fixed_k=None, fixed_r_file=None):
    """Replace the predictor's output by a fixed configuration that meets each sample's budget.

    fixed_k: FastV configuration from the paper's kernel, k fixed, r solved per sample.
    fixed_r_file: per-layer keep ratios p_l, scaled to min(1, c * p_l) with c solved per sample.
    """
    assert (fixed_k is None) != (fixed_r_file is None)
    L = module.num_hidden_layers
    profile = None
    if fixed_r_file is not None:
        with open(fixed_r_file) as f:
            profile = torch.tensor(json.load(f), dtype=torch.float64)
        assert profile.numel() == L, f"{fixed_r_file}: {profile.numel()} ratios for {L} layers"
        profile = torch.cummin(profile.clamp(0, 1), dim=0).values
    module.fixed_log = []

    def layer_r(x):
        if profile is None:
            return module.kernel_func(torch.tensor([float(fixed_k)], dtype=torch.float64),
                                      torch.tensor([x], dtype=torch.float64))[0]
        return (profile * x).clamp(max=1.0)

    def forward(self, hidden_states, budget, visual_mask, attention_mask):
        B = hidden_states.size(0)
        max_b, min_b = self.compute_budget_range(visual_mask, attention_mask)
        rows = []
        for i in range(B):
            n = attention_mask[i].sum().item()
            n_v = visual_mask[i].sum().item()
            n_t = n - n_v
            target = float(min_b[i]) + (float(max_b[i]) - float(min_b[i])) * float(budget[i])

            def cost(x):
                return sum(float(self.compute_flops(n_t + n_v * float(r))) for r in layer_r(x))

            hi = 1.0 if profile is None else 1.0 / max(float(profile[profile > 0].min()), 1e-6)
            infeasible = cost(0.0) > target
            x = 0.0 if infeasible else _bisect(cost, target, 0.0, hi)
            rows.append(layer_r(x))
            self.fixed_log.append({'x': x, 'infeasible': infeasible, 'target': target, 'cost': cost(x)})
        r = torch.stack(rows).to(device=hidden_states.device, dtype=hidden_states.dtype)
        return r, torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype)

    module.forward = types.MethodType(forward, module)
