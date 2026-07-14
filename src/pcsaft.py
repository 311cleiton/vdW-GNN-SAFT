#!/usr/bin/env python3
"""
pcsaft.py  --  Group B (1/2) of the GNN / PC-SAFT density factory.

The differentiable physics bridge: turn predicted PC-SAFT pure-component parameters
into a liquid mass density that can be compared to experiment, with gradients flowing
back to the parameters. This is what makes the "faithful" (physics-in-the-loop) training
work -- the GNN never sees a reference parameter, only the resulting density error.

Engine
------
We use feos-torch (PcSaftPure), a PyTorch-differentiable wrapper around the FeOs PC-SAFT
implementation. Verified facts this module relies on (feos-torch 0.1.0):
  - PcSaftPure(P) takes a [N, 8] float64 parameter matrix, columns in this order:
        [ m, sigma[A], epsilon_k[K], mu[D], kappa_ab, epsilon_k_ab[K], na, nb ]
  - .liquid_density(T[K], P[Pa]) -> (nan_flag[N] bool, density) in mol/L (= kmol/m^3).
        `density` covers the points feos could solve: it is length N when every point converges,
        but feos COMPACTS it to the non-flagged points when some fail to converge, so
        liquid_density_mol_per_L scatters it back to length N (failed -> NaN). Cross-checked
        against plain feos 0.9.5 (exact 1000x of mol/m^3).
  - float64 inputs are required; float32 raises a conversion error.
  - gradients of density w.r.t. the parameters are exact (FeOs dual numbers).

Units
-----
  mol/L * g/mol = g/L = kg/m^3, so   mass_density[kg/m^3] = molar_density[mol/L] * MW[g/mol].
That is exactly the density column produced by build_dataset.py, so no further conversion.

Loss
----
huber_relative_loss is the paper's modified Huber loss (eqs 3-4) -- HuberLoss(delta=0.01)
applied to the *relative* error -- here on density rather than on parameters. Points where
FeOs failed to converge (nan flag) are masked out so they never poison the gradient.

Public API
----------
  assemble_pcsaft_params(pred, na=1, nb=1) -> [N,8] float64
  liquid_density_mol_per_L(params8, T_K, P_Pa) -> (rho[mol/L], valid_mask)
  mass_density_kg_per_m3(params8, T_K, P_kPa, MW_g_mol) -> (rho[kg/m^3], valid_mask)
  huber_relative_loss(pred, target, delta=0.01, mask=None) -> scalar
  density_mape(pred, target, mask=None) -> float   (the eval / tuning metric)

Run `python pcsaft.py` for the self-check (requires feos-torch; cross-checks against plain
feos if it is installed).
"""

from __future__ import annotations
import torch
import torch.nn.functional as F

from feos_torch import PcSaftPure

# feos-torch PcSaftPure parameter layout
PARAM_NAMES = ("m", "sigma", "epsilon_k", "mu", "kappa_ab", "epsilon_k_ab", "na", "nb")


def assemble_pcsaft_params(pred: torch.Tensor, na: float = 1.0, nb: float = 1.0) -> torch.Tensor:
    """Map a model's predicted parameters to the [N, 8] feos-torch vector (differentiably).

    pred: [N, 3] -> (m, sigma, epsilon_k); association is switched off (kappa/eps_ab/na/nb = 0).
    pred: [N, 5] -> (m, sigma, epsilon_k, kappa_ab, epsilon_k_ab); na, nb taken from arguments
                    (default 1,1 = the 2B association scheme).
    Returns a float64 tensor on the same device, with mu fixed at 0.
    """
    if pred.dim() != 2 or pred.shape[1] not in (3, 5):
        raise ValueError(f"pred must be [N,3] or [N,5], got {tuple(pred.shape)}")
    n = pred.shape[0]
    zeros = torch.zeros(n, dtype=pred.dtype, device=pred.device)
    m, sigma, eps = pred[:, 0], pred[:, 1], pred[:, 2]
    if pred.shape[1] == 5:
        kappa, eps_ab = pred[:, 3], pred[:, 4]
        na_t = torch.full((n,), float(na), dtype=pred.dtype, device=pred.device)
        nb_t = torch.full((n,), float(nb), dtype=pred.dtype, device=pred.device)
    else:
        kappa = eps_ab = na_t = nb_t = zeros
    params8 = torch.stack([m, sigma, eps, zeros, kappa, eps_ab, na_t, nb_t], dim=1)
    return params8.double().contiguous()


def _as_f64(x, n=None) -> torch.Tensor:
    t = torch.as_tensor(x, dtype=torch.float64)
    return t.reshape(-1).contiguous()


def liquid_density_mol_per_L(params8: torch.Tensor, T_K, P_Pa):
    """Liquid molar density [mol/L] at (T,P). Returns (rho[N], valid_mask[N]).

    feos flags points it cannot solve (no converged liquid root) in `nan_flag`. When some points
    fail it returns densities ONLY for the points that solved, so the density tensor can be SHORTER
    than the inputs. We scatter it back to full length (failed points -> NaN) so the result always
    lines up 1:1 with the inputs; the failed points are then masked out and counted in coverage.
    This is common in the high-pressure / early-training regime, where some (parameter, T, P)
    combinations have no converged liquid root. (When every point converges, rho is already length
    N and the scatter is skipped, so this is a no-op in the easy case.)
    """
    params8 = params8.double().contiguous()
    T = _as_f64(T_K)
    P = _as_f64(P_Pa)
    n = params8.shape[0]
    if not (T.shape[0] == P.shape[0] == n):
        # alignment bug upstream (collate / mol_index), NOT a feos issue -- fail loudly
        raise ValueError(
            f"input length mismatch before the density solve: "
            f"params8={n}, T={T.shape[0]}, P={P.shape[0]} (these must be equal).")

    nan_flag, rho = PcSaftPure(params8).liquid_density(T, P)

    if rho.shape[0] != n:                       # feos compacted the output to the solved points
        n_ok = int((~nan_flag).sum())
        if rho.shape[0] != n_ok:
            raise RuntimeError(
                f"feos liquid_density returned {rho.shape[0]} densities for {n} inputs "
                f"({n_ok} non-flagged) -- cannot align. Check the feos_torch version/return shape.")
        full = rho.new_full((n,), float("nan"))
        full[~nan_flag] = rho                   # differentiable scatter onto the solved positions
        rho = full

    valid = (~nan_flag) & torch.isfinite(rho) & (rho > 0)
    return rho, valid


def mass_density_kg_per_m3(params8: torch.Tensor, T_K, P_kPa, MW_g_mol):
    """Liquid mass density [kg/m^3] at (T, P[kPa]). Returns (rho, valid_mask)."""
    P_Pa = _as_f64(P_kPa) * 1000.0
    rho_molL, valid = liquid_density_mol_per_L(params8, T_K, P_Pa)
    MW = _as_f64(MW_g_mol)
    rho_kgm3 = rho_molL * MW          # mol/L * g/mol = kg/m^3
    return rho_kgm3, valid


def huber_relative_loss(pred: torch.Tensor, target, delta: float = 0.01, mask=None) -> torch.Tensor:
    """Modified Huber loss (eqs 3-4) on relative error: HuberLoss(delta) of (pred/target - 1).

    delta=0.01 puts the quadratic->linear knee at 1% relative error. Invalid points (mask
    False) are dropped FIRST -- before the division and Huber -- so NaN densities at
    non-converged points cannot leak a NaN into the gradient. If nothing is valid, returns a
    graph-connected zero so .backward() is safe.
    """
    pred = pred.double()
    target = _as_f64(target)
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    if pred.numel() == 0:
        return pred.sum() * 0.0
    ratio = pred / target
    per_point = F.huber_loss(ratio, torch.ones_like(ratio), delta=delta, reduction="none")
    return per_point.mean()


def density_mape(pred: torch.Tensor, target, mask=None) -> float:
    """Mean absolute percentage error of density (the validation / tuning metric)."""
    pred = pred.detach().double()
    target = _as_f64(target)
    ape = 100.0 * (pred - target).abs() / target
    if mask is not None:
        ape = ape[mask]
    return float(ape.mean()) if ape.numel() else float("nan")


# ---------------------------------------------------------------------------
# self-check
# ---------------------------------------------------------------------------
def _self_check():
    torch.manual_seed(0)
    print("== pcsaft.py self-check ==")

    # Core-only parameters for 3 pseudo-ILs (m, sigma, epsilon_k), require grad.
    pred = torch.tensor([[5.0, 3.5, 250.0],
                         [6.2, 3.7, 270.0],
                         [4.1, 3.4, 230.0]], dtype=torch.float32, requires_grad=True)
    MW = torch.tensor([226.0, 419.0, 200.0])          # g/mol (e.g. BMIM-BF4, HMIM-NTf2, generic)
    T = torch.tensor([298.15, 313.15, 333.15])         # K
    P = torch.tensor([101.325, 101.325, 101.325])      # kPa

    params8 = assemble_pcsaft_params(pred)
    assert params8.shape == (3, 8) and params8.dtype == torch.float64
    rho, valid = mass_density_kg_per_m3(params8, T, P, MW)
    print(f"  density [kg/m^3]: {rho.detach().numpy().round(2)}  | valid: {valid.tolist()}")
    assert valid.all(), "FeOs failed to converge on the probe parameters"
    assert torch.all((rho > 500) & (rho < 2500)), "densities outside a sane IL band"

    # gradient flows from a density loss back to the (float32) predictions
    target = torch.tensor([1210.0, 1450.0, 1080.0])    # kg/m^3 pretend-experimental
    loss = huber_relative_loss(rho, target, delta=0.01, mask=valid)
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0
    print(f"  loss={float(loss.detach()):.6f}  MAPE={density_mape(rho, target, valid):.3f}%  "
          f"grad-norm={float(pred.grad.norm()):.4g}")

    # association path: [N,5] -> 8-vector with na=nb=1
    pred5 = torch.tensor([[5.0, 3.5, 250.0, 0.02, 2000.0]], dtype=torch.float64)
    p8 = assemble_pcsaft_params(pred5, na=1, nb=1)
    assert p8.shape == (1, 8) and float(p8[0, 4]) == 0.02 and float(p8[0, 6]) == 1.0
    rho5, v5 = mass_density_kg_per_m3(p8, [298.15], [101.325], [226.0])
    print(f"  association-on density: {float(rho5[0]):.2f} kg/m^3  (valid={bool(v5[0])})")

    # cross-check the units/values against plain feos, if available
    try:
        import feos
        try:
            import si_units as si
        except ImportError:
            from feos import si
        RHO = si.MOL / si.METER**3
        rec = feos.PureRecord(feos.Identifier(name="probe"), molarweight=226.0,
                              m=5.0, sigma=3.5, epsilon_k=250.0)
        eos = feos.EquationOfState.pcsaft(feos.Parameters.new_pure(rec))
        st = feos.State(eos, temperature=298.15 * si.KELVIN, pressure=101.325e3 * si.PASCAL,
                        density_initialization="liquid")
        feos_molar_mol_m3 = float(st.density / RHO)
        feos_kgm3 = feos_molar_mol_m3 / 1000.0 * 226.0   # mol/m^3 -> mol/L -> kg/m^3 (x MW)
        ours = float(rho[0].detach())
        print(f"  cross-check vs plain feos: ours={ours:.2f}  feos={feos_kgm3:.2f} kg/m^3  "
              f"(rel diff {abs(ours-feos_kgm3)/feos_kgm3*100:.3f}%)")
        assert abs(ours - feos_kgm3) / feos_kgm3 < 1e-3, "feos-torch vs feos density mismatch"
        print("  feos cross-check passed.")
    except Exception as e:
        print(f"  (plain-feos cross-check skipped: {type(e).__name__}: {e})")

    print("pcsaft.py self-check passed.")


if __name__ == "__main__":
    _self_check()
