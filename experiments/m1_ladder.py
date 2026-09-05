"""The measurement ladder of `docs/STATUS.md` section 5.12, one stage per rung.

The point of the ladder is that a diagnostic GPU point costs 1.64 h, so every
suspect that can be eliminated from a cached `(W, H)` gets eliminated first.
Rung 1 is `m0_lever_audit.py --build`, which writes that cache; everything here
reads it.  Build it at the DRIVER's token count, not the default:

    python -u experiments/m0_lever_audit.py --build --build-only --seqs 128 \
        --cache results/block0_problems_262k

Stages, and the question each one answers:

    mask         is the mask semi-structured, or has it collapsed to
                 structural channel pruning?           (rung 2a)
    prune        what does the pruning cost on its own? (rung 2b)
    hessian      is `hessian_block=512` the lever the synthetic measurement
                 said it was?                           (rung 3)
    quantizer    with the pruning removed entirely, does the error collapse or
                 stay?                                  (rung 4)
    sequential   how much do the records hide by collecting every Hessian on a
                 fully dense block?                     (rung 5)
    precision    is any of this float32 running out of room?
    ablate       of bare E8P / + rotation / + LDLQ, which part pays?
    granularity  a tile is 16 rows sharing one index set, one rotation and one
                 scale, and the rotation mixes the index axis only -- so the
                 per-row spread survives it.  Hold density at 1.0 and move the
                 tile size and nothing changes but that sharing.
    spectrum     tr(E H E^T) = sum_i lambda_i ||E v_i||^2 splits the error over
                 H's eigendirections; the same sum for W says where the SIGNAL
                 is.  Error more concentrated than signal means LDLQ is losing
                 exactly where the objective looks.

`sequential` is the one stage that loads the model: it has to re-run block 0
with an upstream layer already compressed.  The rest are seconds to minutes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "experiments", _ROOT / "eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import accounting as A                                             # noqa: E402
import calibrate as Cal                                            # noqa: E402
import compact as C                                                # noqa: E402
import hf_llama as HF                                              # noqa: E402
import prune as P                                                  # noqa: E402
import quantize as Qz                                              # noqa: E402
import rotation as R                                               # noqa: E402
import tiling as Tl                                                # noqa: E402
from m0_lever_audit import ALL_LAYERS, load_problem                 # noqa: E402
from m1_gates import PIPELINE_ROTATE_KRON, run_config, tile_hessian_stream  # noqa: E402

DEFAULT_CACHE = _ROOT / "results" / "block0_problems_262k"
DEFAULT_MODEL = "NousResearch/Llama-2-7b-hf"
# The two seams inside a block: each target is scored against a Hessian its
# upstream layers were still dense for.
SEAMS = [("self_attn.o_proj", ["self_attn.v_proj"]),
         ("mlp.down_proj", ["mlp.gate_proj", "mlp.up_proj"])]


def _scheme(tile_size) -> str:
    return {1: "unstructured", Tl.MAX_TILE: "structured"}.get(tile_size, "tile")


def _cond(M: torch.Tensor) -> float:
    ev = torch.linalg.eigvalsh(M.double())
    return float(ev.max() / ev.clamp_min(0).min().clamp_min(1e-30))


# --------------------------------------------------------------------------- #
# Stages that read one cached problem
# --------------------------------------------------------------------------- #

def stage_mask(prob, args) -> dict:
    """Rung 2a: how much do the tiles' index sets actually differ?"""
    d = A.density_for_budget(_scheme(args.tile), args.budget, None, prob.n_in,
                             tile_size=args.tile, vq_bits=Qz.E8P_BITS_PER_WEIGHT)
    mask = P.prune(prob.W, axis="B", tile_size=args.tile, density=d,
                   metric="wanda", act_norm=prob.act_norm,
                   align=Qz.E8P_DIM).mask
    S = mask.support
    Sf = S.float()
    k = Sf.sum(1)
    inter = Sf @ Sf.T
    jac = inter / (k[:, None] + k[None, :] - inter)
    off = ~torch.eye(len(k), dtype=torch.bool, device=jac.device)
    return {"k": int(k[0]), "n_tiles": int(S.shape[0]),
            "density": mask.density(),
            "jaccard_mean": float(jac[off].mean()),
            "jaccard_min": float(jac[off].min()),
            "in_all_tiles": int(S.all(0).sum()),
            "in_any_tile": int(S.any(0).sum())}


def stage_prune(prob, args) -> dict:
    """Rung 2b: the mask on its own, against the whole pipeline."""
    kw = dict(budget_bits=args.budget, tile_size=args.tile)
    return {"prune_only": run_config(prob, quantize=False, align=Qz.E8P_DIM,
                                     **kw)["rel_output_error"],
            "full": run_config(prob, **kw)["rel_output_error"]}


def stage_hessian(prob, args) -> dict:
    """Rung 3: 512-wide sub-Hessian blocks against the unconstrained feedback."""
    out = {}
    for tag, hb in (("h512", 512), ("hfull", None)):
        t0 = time.perf_counter()
        out[tag] = run_config(prob, budget_bits=args.budget,
                              tile_size=args.tile,
                              hessian_block=hb)["rel_output_error"]
        out[f"{tag}_sec"] = round(time.perf_counter() - t0, 1)
    out["full_over_512"] = out["hfull"] / out["h512"]
    return out


def stage_quantizer(prob, args) -> dict:
    """Rung 4: the quantizer with no pruning to hide behind.

    `dense_budget` is whatever buys density 1.0 at this tile size, so the arms
    differ in sparsity and in nothing else.
    """
    dense = A.bits_per_position(_scheme(args.tile), 1.0, None, prob.n_in,
                                tile_size=args.tile,
                                vq_bits=Qz.E8P_BITS_PER_WEIGHT)
    out = {}
    for tag, bits in (("sparse", args.budget), ("dense", dense)):
        r = run_config(prob, budget_bits=bits, tile_size=args.tile)
        out[f"{tag}_bits"] = round(bits, 4)
        out[f"{tag}_d"] = r["density_realized"]
        out[f"{tag}_err"] = r["rel_output_error"]
    out["gain_from_dropping_sparsity"] = 1 - out["dense_err"] / out["sparse_err"]
    return out


def stage_precision(prob, args) -> dict:
    """float32 is what the driver runs.  Does float64 move anything?"""
    return {"float32": run_config(prob, budget_bits=args.budget,
                                  tile_size=args.tile)["rel_output_error"]}


def stage_ablate(prob, args) -> dict:
    """Bare E8P, then the rotation, then LDLQ -- at equal density and alignment."""
    arms = {"bare": dict(rotate_axis=None, ldlq=False),
            "rot": dict(rotate_axis="index", ldlq=False),
            "rot_ldlq": dict(rotate_axis="index", ldlq=True)}
    out = {}
    for tag, kw in arms.items():
        out[tag] = run_config(prob, budget_bits=args.budget,
                              tile_size=args.tile, align=Qz.E8P_DIM,
                              **kw)["rel_output_error"]
    out["rot_gain"] = 1 - out["rot"] / out["bare"]
    out["ldlq_gain"] = 1 - out["rot_ldlq"] / out["rot"]
    return out


def stage_granularity(prob, args) -> dict:
    """Density pinned at 1.0, tile size swept: only the sharing changes.

    Small tile sizes are the expensive arm, not the cheap one: at d=1.0 every
    tile carries the full index axis, so T=1 is `n_out` separate rotations and
    LDLQ passes over a k x k Hessian.  T=1 on a 4096-wide layer runs for hours;
    `--gran-tiles` is there so the sweep can start where it is affordable.
    """
    out = {}
    for t in args.gran_tiles:
        bits = A.bits_per_position(_scheme(t), 1.0, None, prob.n_in,
                                   tile_size=t, vq_bits=Qz.E8P_BITS_PER_WEIGHT)
        r = run_config(prob, budget_bits=bits, tile_size=t)
        out[str(t)] = (r["skipped"] if "skipped" in r else
                       {"bits": round(bits, 4), "d": r["density_realized"],
                        "err": r["rel_output_error"]})
    return out


def stage_spectrum(prob, args) -> dict:
    """Split the error and the signal over H's eigendirections."""
    r = run_config(prob, budget_bits=args.budget, tile_size=args.tile,
                   return_weight=True)
    E = prob.W - r["W_hat"]
    lam, V = torch.linalg.eigh(prob.H.double())
    order = torch.argsort(lam, descending=True)
    lam, V = lam[order].clamp_min(0), V[:, order]
    err = lam * (E.double() @ V).pow(2).sum(0)
    sig = lam * (prob.W.double() @ V).pow(2).sum(0)
    out = {"rel_output_error": r["rel_output_error"], "cond": float(
        lam.max() / lam.clamp_min(0).min().clamp_min(1e-30))}
    for frac in (0.01, 0.10, 0.50):
        n = max(1, int(frac * len(lam)))
        tag = int(frac * 100)
        out[f"err_top{tag}"] = float(err[:n].sum() / err.sum())
        out[f"sig_top{tag}"] = float(sig[:n].sum() / sig.sum())
        out[f"ratio_top{tag}"] = out[f"err_top{tag}"] / out[f"sig_top{tag}"]
    return out


def stage_conditioning(prob, args) -> dict:
    """Rung 2c: tile 0's sub-Hessian, before and after the rotation.

    An orthogonal map cannot move eigenvalues, so this is a check that the
    rotation is what it claims -- and a reading of how much room float32 has.
    """
    d = A.density_for_budget(_scheme(args.tile), args.budget, None, prob.n_in,
                             tile_size=args.tile, vq_bits=Qz.E8P_BITS_PER_WEIGHT)
    pruned = P.prune(prob.W, axis="B", tile_size=args.tile, density=d,
                     metric="wanda", act_norm=prob.act_norm, align=Qz.E8P_DIM)
    cw = C.compact(pruned.W, pruned.mask)
    rot, Qm = R.rotate(cw, axis="index", seed=0)
    fac = (R.kronecker_factors(cw.k, 0, rot.blocks.dtype, rot.blocks.device)
           if PIPELINE_ROTATE_KRON else None)
    return {"cond_raw": _cond(tile_hessian_stream(prob, cw, None)(0)),
            "cond_rot": _cond(tile_hessian_stream(prob, cw, Qm, factors=fac)(0)),
            "float32_headroom": 1.0 / torch.finfo(torch.float32).eps}


PER_LAYER = {"mask": stage_mask, "prune": stage_prune, "hessian": stage_hessian,
             "quantizer": stage_quantizer, "precision": stage_precision,
             "ablate": stage_ablate, "granularity": stage_granularity,
             "spectrum": stage_spectrum, "conditioning": stage_conditioning}


# --------------------------------------------------------------------------- #
# Rung 5, which needs the model
# --------------------------------------------------------------------------- #

def run_sequential(args, emit) -> list[dict]:
    """Three numbers per seam, all at the driver's token count.

    recorded    compress with the stale H, score against the stale H  (today)
    honest      compress with the stale H, score against the fresh H  (the cost)
    sequential  compress with the fresh H, score against the fresh H  (the fix)
    """
    emit(f"loading {args.model} ...")
    harness = HF.load_llama(args.model, dtype=torch.float16)
    tokens = Cal.load_calibration_tokens(harness.tokenizer, n_samples=args.seqs,
                                         seqlen=args.seqlen, seed=0,
                                         dataset="wikitext2")
    batches = [tokens[i:i + args.batch]
               for i in range(0, tokens.shape[0], args.batch)]
    hidden, bkw = HF.capture_block_inputs(harness.model, batches)
    block = harness.blocks[0].to(args.device)
    hidden = [h.to(args.device) for h in hidden]
    bkw = HF.to_device(bkw, args.device)
    linears = Cal.find_linears(block)
    pristine = {k: v.weight.detach().clone() for k, v in linears.items()}

    def problem(W, H, name):
        return Cal.LayerProblem.from_statistics(W, H, name=name)

    rows = []
    for target, upstream in SEAMS:
        for k, v in linears.items():        # every seam starts from the real block
            v.weight.data.copy_(pristine[k])
        accs = Cal.collect_block_statistics(
            block, hidden, block_kwargs=bkw, names=upstream + [target],
            dtype=torch.float32)
        H_stale = accs[target].H.clone()
        W_t = pristine[target].to(args.device, torch.float32)

        for up in upstream:
            r = run_config(problem(pristine[up].to(args.device, torch.float32),
                                   accs[up].H, up),
                           budget_bits=args.budget, tile_size=args.tile,
                           return_weight=True)
            linears[up].weight.data.copy_(r["W_hat"].to(torch.float16))
            emit(f"  compressed {up}, err {r['rel_output_error']:.4f}")

        H_fresh = Cal.collect_block_statistics(
            block, hidden, block_kwargs=bkw, names=[target],
            dtype=torch.float32)[target].H
        p_stale = problem(W_t, H_stale, target)
        p_fresh = problem(W_t, H_fresh, target)
        kw = dict(budget_bits=args.budget, tile_size=args.tile,
                  return_weight=True)
        from_stale = run_config(p_stale, **kw)
        from_fresh = run_config(p_fresh, **kw)

        row = {"layer": target, "upstream": upstream,
               "n_tokens": int(accs[target].n_tokens),
               "H_drift": float((H_fresh - H_stale).norm() / H_stale.norm()),
               "recorded": from_stale["rel_output_error"],
               "honest": p_fresh.output_error(from_stale["W_hat"]),
               "sequential": p_fresh.output_error(from_fresh["W_hat"])}
        row["optimism"] = row["honest"] / row["recorded"]
        row["sequential_gain"] = 1 - row["sequential"] / row["honest"]
        emit(json.dumps(row))
        rows.append(row)
        del accs, H_stale, H_fresh, W_t
        torch.cuda.empty_cache()
    return rows


# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="*", default=["mask", "prune", "hessian",
                                                    "quantizer", "ablate"],
                    help=f"any of {sorted(PER_LAYER)} plus 'sequential'")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--layers", nargs="*", default=list(ALL_LAYERS))
    ap.add_argument("--budget", type=float, default=1.5)
    ap.add_argument("--tile", type=int, default=16)
    ap.add_argument("--gran-tiles", nargs="*", default=[16, 64, Tl.MAX_TILE],
                    type=lambda s: s if s == Tl.MAX_TILE else int(s),
                    help="tile sizes for the granularity sweep; small ones are "
                         "the expensive arm (T=1 runs for hours at d=1.0)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--model", default=DEFAULT_MODEL, help="sequential only")
    ap.add_argument("--seqs", type=int, default=128, help="sequential only")
    ap.add_argument("--seqlen", type=int, default=2048, help="sequential only")
    ap.add_argument("--batch", type=int, default=4, help="sequential only")
    ap.add_argument("--out", type=Path, default=_ROOT / "results" / "m1_ladder.json")
    args = ap.parse_args(argv)

    unknown = set(args.stages) - set(PER_LAYER) - {"sequential"}
    if unknown:
        ap.error(f"unknown stage(s) {sorted(unknown)}; "
                 f"pick from {sorted(PER_LAYER) + ['sequential']}")

    def emit(msg):
        print(msg, flush=True)

    out: dict = {"cache": str(args.cache), "budget_bits": args.budget,
                 "tile_size": args.tile, "stages": {}}

    per_layer = [s for s in args.stages if s in PER_LAYER]
    if per_layer:
        dtype = getattr(torch, args.dtype)
        for name in args.layers:
            prob = load_problem(args.cache, name, device=args.device, dtype=dtype)
            row = {"layer": name, "n_tokens": prob.n_tokens,
                   "n_in": prob.n_in, "n_out": prob.n_out}
            for stage in per_layer:
                row[stage] = PER_LAYER[stage](prob, args)
            emit(json.dumps(row))
            out["stages"].setdefault("per_layer", []).append(row)
            del prob
            if args.device == "cuda":
                torch.cuda.empty_cache()

    if "sequential" in args.stages:
        out["stages"]["sequential"] = run_sequential(args, emit)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    emit(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
