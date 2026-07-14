#!/usr/bin/env python3
"""
model.py  --  Group B (2/2) of the GNN / PC-SAFT density factory.

The Figure-2 GNN that maps a molecular graph to PC-SAFT pure-component parameters:

    atom/bond encode  ->  N message-passing layers (hidden dim)  ->  global sum pool
                       ->  [optionally concat the molecule's vdW]  ->  3-layer "halving" MLP head(s)

Two heads (the paper's two-model design, folded into one shared body here):
  - core head      -> (m, sigma, epsilon_k)              -- every molecule has these
  - association head -> (kappa_ab, epsilon_k_ab)          -- only when predict_association=True

Optional extra input (use_vdw=True): the molecule's van der Waals volume [cubic angstrom] is
standardized with the training mean/std (stored as buffers) and concatenated to the pooled graph
embedding before the heads, so the head input grows from `hidden` to `hidden + 1`. This lets the
network use a global descriptor that the graph alone may not capture -- most importantly the
composition of multi-fragment mixtures (e.g. deep eutectic solvents), where the same SMILES can
correspond to several vdW values. vdW is passed to forward() at call time, not stored on the
graph. When use_vdw=False the model behaves exactly as before.

Conv factory covers the three Table-4 candidates: PNA, GATv2, TransformerConv. PNA needs a
degree histogram of the training graphs (compute_degree_histogram), because its scalers
normalize against the typical node degree.

Positivity: PC-SAFT parameters must be positive, and feos-torch returns NaN on non-physical
inputs. Each head output is therefore passed through softplus(raw + bias), where bias is the
inverse-softplus of a physical prior, so parameters start near sane magnitudes and stay > 0.
Set positive=False to recover the bare linear+bias head used in the learning modules.
Optionally (use_bounds=True) each output is instead mapped through a sigmoid squash into a
physical box [lo, hi] (see BOUNDS below), with the bias set so the head starts at the prior.
Bounds stop the predicted parameters from wandering into regions where FeOs cannot converge
(a source of NaN density and training collapse); this mode takes precedence over `positive`.

Output: a [B, 3] (core only) or [B, 5] (core + association) tensor, columns ordered
(m, sigma, epsilon_k[, kappa_ab, epsilon_k_ab]) -- exactly what pcsaft.assemble_pcsaft_params
consumes. The GNN runs once per molecule (per (molecule, vdW) when vdW is on); train.py expands
these per-molecule parameters to the per-(T,P) experimental points before calling the density
bridge.

Run `python model.py` for the self-check (PyG + OGB required).
"""

from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import PNAConv, GATv2Conv, TransformerConv, global_add_pool
from torch_geometric.utils import degree
from torch_geometric.data import Data
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from ogb.utils.mol import smiles2graph

DEFAULT_PRIORS = {"m": 3.0, "sigma": 3.5, "epsilon_k": 250.0,
                  "kappa_ab": 0.02, "epsilon_k_ab": 2000.0}
# Physical box for the (optional) bounded prediction head. When PCSAFTGNN is built with
# use_bounds=True, each parameter is mapped through squash() into its [lo, hi] instead of through
# softplus, so the GNN can never emit a non-physical value that pushes FeOs out of its convergence
# region (a cause of NaN density / training collapse). Ranges follow the PCP-SAFT regression
# bounds in the literature (e.g. Felton et al. 2024, Table 1), adjusted slightly.
# BOUNDS = {"m": (1.0, 15.0), "sigma": (2.5, 5.5), "epsilon_k": (150.0, 600.0), # AC
        #   "kappa_ab": (1e-4, 0.15), "epsilon_k_ab": (500.0, 5000.0)}
BOUNDS = {"m": (1.0, 15.0), "sigma": (2.5, 4.5), "epsilon_k": (150.0, 500.0),
          "kappa_ab": (1e-4, 0.15), "epsilon_k_ab": (500.0, 5000.0)}  # DEFAULT
# NOTE: the two association entries are required only when predict_association=True AND
# use_bounds=True. The paper runs association OFF, so they are never read by any reported
# result; without them that combination raised KeyError("kappa_ab").
# BOUNDS = {"m": (1.0, 10.0), "sigma": (2.5, 5.0), "epsilon_k": (100.0, 1000.0),} # Felton et al. 2024
# BOUNDS = {"m": (1.0, 23.32), "sigma": (1.9, 4.5), "epsilon_k": (50.0, 550.0),} # Esper et al. 2023


def squash(raw: torch.Tensor, lo, hi) -> torch.Tensor:
    """Smoothly map an unbounded pre-activation into (lo, hi) via a sigmoid."""
    return lo + (hi - lo) * torch.sigmoid(raw)


def inv_softplus(y: float) -> float:
    """Inverse of softplus; linear approximation for large y to avoid float overflow."""
    return y if y > 20.0 else math.log(math.expm1(y))


def _bounds_bias(prior: float, lo: float, hi: float) -> float:
    """Bias b with squash(b, lo, hi) == prior, so a bounded head starts at the physical prior."""
    frac = (prior - lo) / (hi - lo)
    frac = min(max(frac, 1e-6), 1.0 - 1e-6)     # keep the logit finite if a prior sits on a bound
    return math.log(frac / (1.0 - frac))


def smiles_to_data(smiles: str) -> Data:
    """SMILES -> PyG Data with OGB integer atom/bond features (handles dot-separated ion pairs)."""
    g = smiles2graph(smiles)
    return Data(
        x=torch.from_numpy(g["node_feat"]).long(),
        edge_index=torch.from_numpy(g["edge_index"]).long(),
        edge_attr=torch.from_numpy(g["edge_feat"]).long(),
        num_nodes=int(g["num_nodes"]),
    )


def compute_degree_histogram(data_list) -> torch.Tensor:
    """Histogram of node in-degrees over a list of Data objects (the `deg` arg for PNAConv)."""
    max_deg = 0
    degs = []
    for d in data_list:
        if d.edge_index.numel() == 0:
            continue
        dd = degree(d.edge_index[1], num_nodes=d.num_nodes, dtype=torch.long)
        degs.append(dd)
        max_deg = max(max_deg, int(dd.max()))
    hist = torch.zeros(max_deg + 1, dtype=torch.long)
    for dd in degs:
        hist += torch.bincount(dd, minlength=max_deg + 1)
    return hist


class PCSAFTGNN(nn.Module):
    def __init__(self, hidden: int = 256, depth: int = 6, conv_type: str = "PNA",
                 predict_association: bool = False, use_vdw: bool = False,
                 vdw_mean: float = 0.0, vdw_std: float = 1.0,
                 towers: int = 4, pre_layers: int = 1,
                 post_layers: int = 2, heads: int = 4, deg: torch.Tensor = None,
                 positive: bool = True, priors: dict = None,
                 use_bounds: bool = False, bounds: dict = None):
        super().__init__()
        self.conv_type = conv_type
        self.predict_association = predict_association
        self.use_vdw = use_vdw
        self.positive = positive
        self.use_bounds = use_bounds
        priors = priors or DEFAULT_PRIORS

        self.atom_encoder = AtomEncoder(hidden)
        self.bond_encoder = BondEncoder(hidden)
        self.convs = nn.ModuleList(
            self._make_conv(conv_type, hidden, towers, pre_layers, post_layers, heads, deg)
            for _ in range(depth))

        # When vdW is used it is concatenated to the pooled graph embedding, so the heads take
        # one extra input. vdW is standardized inside forward() with these (training) stats.
        head_in = hidden + (1 if use_vdw else 0)
        self.core_head = self._halving_head(head_in, hidden, 3)
        self.assoc_head = self._halving_head(head_in, hidden, 2) if predict_association else None

        if use_vdw:
            self.register_buffer("vdw_mean", torch.tensor(float(vdw_mean)))
            self.register_buffer("vdw_std", torch.tensor(max(float(vdw_std), 1e-8)))

        # Output-activation buffers. Bounded mode registers the per-parameter box [lo, hi] and a
        # logit-bias (so the head starts at the prior); the plain positive mode registers the
        # inverse-softplus bias. Both are saved in the state_dict, so a reloaded model reproduces
        # the exact activation it trained with regardless of the current BOUNDS/priors constants.
        core_keys = ("m", "sigma", "epsilon_k")
        assoc_keys = ("kappa_ab", "epsilon_k_ab")
        if use_bounds:
            bounds = bounds or BOUNDS
            self.register_buffer("core_lo", torch.tensor([bounds[k][0] for k in core_keys]))
            self.register_buffer("core_hi", torch.tensor([bounds[k][1] for k in core_keys]))
            self.register_buffer(
                "core_bias", torch.tensor([_bounds_bias(priors[k], *bounds[k]) for k in core_keys]))
            if predict_association:
                self.register_buffer("assoc_lo", torch.tensor([bounds[k][0] for k in assoc_keys]))
                self.register_buffer("assoc_hi", torch.tensor([bounds[k][1] for k in assoc_keys]))
                self.register_buffer(
                    "assoc_bias", torch.tensor([_bounds_bias(priors[k], *bounds[k]) for k in assoc_keys]))
        else:
            self.register_buffer(
                "core_bias", torch.tensor([inv_softplus(priors[k]) for k in core_keys]))
            if predict_association:
                self.register_buffer(
                    "assoc_bias", torch.tensor([inv_softplus(priors[k]) for k in assoc_keys]))

    @staticmethod
    def _make_conv(conv_type, hidden, towers, pre_layers, post_layers, heads, deg):
        if conv_type == "PNA":
            if deg is None:
                raise ValueError("conv_type='PNA' requires a degree histogram (deg=...).")
            if hidden % towers != 0:
                raise ValueError(f"hidden ({hidden}) must be divisible by towers ({towers}).")
            return PNAConv(hidden, hidden,
                           aggregators=["mean", "min", "max", "std"],
                           scalers=["identity", "amplification", "attenuation"],
                           deg=deg, edge_dim=hidden, towers=towers,
                           pre_layers=pre_layers, post_layers=post_layers)
        if conv_type in ("GATv2", "Transformer"):
            if hidden % heads != 0:
                raise ValueError(f"hidden ({hidden}) must be divisible by heads ({heads}).")
            cls = GATv2Conv if conv_type == "GATv2" else TransformerConv
            return cls(hidden, hidden // heads, heads=heads, edge_dim=hidden)
        raise ValueError(f"unknown conv_type {conv_type!r} (use PNA / GATv2 / Transformer)")

    @staticmethod
    def _halving_head(in_dim, hidden, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, hidden // 4), nn.ReLU(),
            nn.Linear(hidden // 4, out_dim))

    def forward(self, data, vdw=None) -> torch.Tensor:
        h = self.atom_encoder(data.x)
        e = self.bond_encoder(data.edge_attr)
        for conv in self.convs:
            h = F.relu(conv(h, data.edge_index, e))
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        g = global_add_pool(h, batch)                      # [B, hidden]

        if self.use_vdw:
            if vdw is None:
                raise ValueError("use_vdw=True but forward() was called without a vdW tensor.")
            v = torch.as_tensor(vdw, dtype=g.dtype, device=g.device).view(-1, 1)
            v = (v - self.vdw_mean) / self.vdw_std         # standardize with training stats
            g = torch.cat([g, v], dim=1)                   # [B, hidden + 1]

        core = self.core_head(g)
        if self.use_bounds:                                # sigmoid squash into [lo, hi]
            core = squash(core + self.core_bias, self.core_lo, self.core_hi)
        elif self.positive:                                # softplus (unbounded above)
            core = F.softplus(core + self.core_bias)
        if self.assoc_head is None:
            return core
        assoc = self.assoc_head(g)
        if self.use_bounds:
            assoc = squash(assoc + self.assoc_bias, self.assoc_lo, self.assoc_hi)
        elif self.positive:
            assoc = F.softplus(assoc + self.assoc_bias)
        return torch.cat([core, assoc], dim=1)


# ---------------------------------------------------------------------------
# self-check
# ---------------------------------------------------------------------------
def _self_check():
    from torch_geometric.loader import DataLoader
    torch.manual_seed(0)
    print("== model.py self-check ==")

    smiles = [
        "CCCC[n+]1ccn(C)c1.F[B-](F)(F)F",                                   # [BMIM][BF4]
        "CCCCCC[n+]1ccn(C)c1.O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F",         # [HMIM][NTf2]
        "CCO",                                                               # ethanol (single fragment)
    ]
    datas = [smiles_to_data(s) for s in smiles]
    deg = compute_degree_histogram(datas)
    print(f"  built {len(datas)} graphs; degree histogram (len {len(deg)}): {deg.tolist()}")
    batch = next(iter(DataLoader(datas, batch_size=len(datas))))

    for conv in ("PNA", "GATv2", "Transformer"):
        for assoc in (False, True):
            for use_vdw in (False, True):
                for use_bounds in (False, True):
                    model = PCSAFTGNN(hidden=64, depth=3, conv_type=conv, predict_association=assoc,
                                      use_vdw=use_vdw, vdw_mean=175.0, vdw_std=80.0,
                                      use_bounds=use_bounds, towers=4, heads=4, deg=deg)
                    vdw = torch.rand(len(datas), dtype=torch.float64) * 300.0 if use_vdw else None
                    out = model(batch, vdw=vdw)
                    tag = f"{conv} assoc={assoc} vdw={use_vdw} bounds={use_bounds}"
                    expected = (len(datas), 5 if assoc else 3)
                    assert out.shape == expected, f"{tag}: got {tuple(out.shape)}"
                    assert torch.all(out > 0), f"{tag}: parameters must be positive"
                    if use_bounds:
                        lo = torch.cat([model.core_lo, model.assoc_lo]) if assoc else model.core_lo
                        hi = torch.cat([model.core_hi, model.assoc_hi]) if assoc else model.core_hi
                        assert torch.all(out >= lo - 1e-4) and torch.all(out <= hi + 1e-4), \
                            f"{tag}: output landed outside the physical box"
                    (out.sum()).backward()
                    gnorm = sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None)
                    assert torch.isfinite(gnorm) and gnorm > 0, f"{tag}: bad gradient"
                    m, sig, eps = (round(float(x), 2) for x in out[0, :3].detach())
                    print(f"  {conv:11s} assoc={str(assoc):5s} vdw={str(use_vdw):5s} "
                          f"bounds={str(use_bounds):5s} out{tuple(out.shape)}  "
                          f"(m,sigma,eps)[0]=({m},{sig},{eps})  grad ok")

    # bias-at-priors: a freshly built bounded head at zero pre-activation must return the physical
    # priors (so training starts at sane magnitudes), and the squash keeps outputs inside [lo, hi].
    mb = PCSAFTGNN(hidden=64, depth=3, conv_type="GATv2", predict_association=True,
                   use_bounds=True, heads=4, deg=deg)
    core_prior = squash(mb.core_bias, mb.core_lo, mb.core_hi)
    assoc_prior = squash(mb.assoc_bias, mb.assoc_lo, mb.assoc_hi)
    exp_core = torch.tensor([DEFAULT_PRIORS[k] for k in ("m", "sigma", "epsilon_k")])
    exp_assoc = torch.tensor([DEFAULT_PRIORS[k] for k in ("kappa_ab", "epsilon_k_ab")])
    assert torch.allclose(core_prior, exp_core, rtol=1e-4, atol=1e-4), "bias-at-priors (core) off"
    assert torch.allclose(assoc_prior, exp_assoc, rtol=1e-4, atol=1e-4), "bias-at-priors (assoc) off"
    print(f"  bias-at-priors OK: core->{[round(float(x), 2) for x in core_prior]}  "
          f"assoc->{[round(float(x), 3) for x in assoc_prior]}")
    print("model.py self-check passed.")


if __name__ == "__main__":
    _self_check()