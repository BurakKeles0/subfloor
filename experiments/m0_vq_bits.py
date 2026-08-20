"""M0 -- what a 2-bit lattice checkpoint actually costs per weight.

Spec v7 section 3.2 takes `vq_bits = idx_bits / dim` for a structured codebook
and sets the amortization to zero, giving E8P exactly 2.000 bits/weight.  The
whole live band (1.40 - 1.80) hangs off that number, so the spec requires it to
be measured from a released checkpoint before it anchors anything.  This does
that, against QuIP#'s and QTIP's own Llama-2-7B releases -- the second is Gate
A's actual competitor, so the pre-registration asks for its cost separately.

Nothing is downloaded but a header.  A safetensors file opens with a JSON
manifest naming every tensor, its dtype, shape and byte range, so two range
requests give the exact size of every tensor in a 2.1 GB file.  A handful of the
small tensors are then fetched to see what the side info *is*, not just how big
it is; that turns out to matter more than the size does.

Three numbers, deliberately redundant:

  payload   the codeword tensors against config-derived shapes -- must land on
            2.000 exactly
  stored    payload plus everything else the checkpoint keeps per linear
  by_size   the same quantity derived from the total file size, which does not
            depend on classifying the tensors correctly and so catches anything
            the classification missed

Then the part that concerns us.  They rotate a linear once; we rotate once per
tile, because each tile owns a different column set.  Their side info and ours
do not scale alike, and `pipeline_side_bits` works out what our design pays --
which is where the SU/SV probe earns its keep.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FILENAME = "model.safetensors"

DTYPE_BITS = {"F16": 16, "BF16": 16, "F32": 32, "I64": 64, "I32": 32,
              "I16": 16, "I8": 8, "U8": 8, "BOOL": 8}


@dataclass(frozen=True)
class Layout:
    """How one release names things, and the shapes it is quantizing.

    Shapes come from the Llama-2-7B config (hidden 4096, intermediate 11008),
    never from the stored tensors: back-computing n_in out of the index would
    assume the 2 bits under test.
    """
    payload: str                       # tensor suffix holding the codewords
    shapes: dict                       # linear name -> (n_out, n_in)


#: QuIP# fuses q/k/v and up/gate into single linears; QTIP keeps them apart.
_UNFUSED = {
    "self_attn.q_proj": (4096, 4096), "self_attn.k_proj": (4096, 4096),
    "self_attn.v_proj": (4096, 4096), "self_attn.o_proj": (4096, 4096),
    "mlp.gate_proj": (11008, 4096), "mlp.up_proj": (11008, 4096),
    "mlp.down_proj": (4096, 11008),
}

LAYOUTS = {
    "relaxml/Llama-2-7b-E8P-2Bit": Layout("Qidxs", {
        "self_attn.qkv_proj": (12288, 4096),
        "self_attn.o_proj": (4096, 4096),
        "mlp.upgate_proj": (22016, 4096),
        "mlp.down_proj": (4096, 11008),
    }),
    "relaxml/Llama-2-7b-QTIP-2Bit": Layout("trellis", _UNFUSED),
}

REPO = "relaxml/Llama-2-7b-E8P-2Bit"

#: Survivors in one tile at the middle of the live band: d = 0.72 of 11008.
#: Only used to put the pipeline side-info table on concrete numbers.
SURVIVORS_PER_TILE = round(0.72 * 11008)


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------

def _auth() -> dict:
    from huggingface_hub import get_token
    token = get_token()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _url(repo: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def fetch_manifest(repo: str = REPO, filename: str = FILENAME,
                   *, cache: Path | None = None) -> dict:
    """Return the safetensors header plus the file's total size.

    Cached: the arithmetic below is worth re-running offline, and the header is
    the only part that needs the network.
    """
    if cache is not None and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    import requests
    from huggingface_hub import HfApi

    auth, url = _auth(), _url(repo, filename)
    size = next(s.size for s in HfApi().model_info(repo, files_metadata=True).siblings
                if s.rfilename == filename)

    head = requests.get(url, headers={**auth, "Range": "bytes=0-7"}, timeout=60)
    n = struct.unpack("<Q", head.content[:8])[0]
    body = requests.get(url, headers={**auth, "Range": f"bytes=8-{8 + n - 1}"},
                        timeout=180).content
    header = json.loads(body)
    header.pop("__metadata__", None)

    out = {"repo": repo, "filename": filename,
           "header_bytes": int(n), "file_bytes": int(size), "header": header}
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def fetch_tensor(name: str, manifest: dict):
    """One tensor, by range request.  Only used on the small side-info ones."""
    import numpy as np
    import requests

    entry = manifest["header"][name]
    base = 8 + manifest["header_bytes"]
    start, end = entry["data_offsets"]
    raw = requests.get(
        _url(manifest["repo"], manifest["filename"]),
        headers={**_auth(), "Range": f"bytes={base + start}-{base + end - 1}"},
        timeout=180).content
    dt = {"F16": np.float16, "F32": np.float32, "I64": np.int64}[entry["dtype"]]
    return np.frombuffer(raw, dtype=dt).reshape(entry["shape"] or [1])


# --------------------------------------------------------------------------
# accounting
# --------------------------------------------------------------------------

def linear_key(name: str, layout: Layout) -> str | None:
    """'model.layers.7.mlp.down_proj.SU' -> 'mlp.down_proj'."""
    for key in layout.shapes:
        if f".{key}." in name:
            return key
    return None


def tensor_bits(entry: dict) -> int:
    n = 1
    for s in entry["shape"]:
        n *= s
    return n * DTYPE_BITS[entry["dtype"]]


@dataclass
class Accounting:
    n_weights: int = 0              # dense-equivalent weights that were quantized
    payload_bits: int = 0           # Qidxs
    side_bits: int = 0              # SU, SV, Wscale, codebook_id, fuse_scales
    unquantized_bits: int = 0       # embeddings, lm_head, norms
    side_breakdown: dict = field(default_factory=dict)
    packing_exact: bool = True      # every Qidxs is exactly 2 bits/weight
    packing_detail: dict = field(default_factory=dict)

    @property
    def payload_per_weight(self) -> float:
        return self.payload_bits / self.n_weights

    @property
    def stored_per_weight(self) -> float:
        return (self.payload_bits + self.side_bits) / self.n_weights


def account(manifest: dict, layout: Layout) -> Accounting:
    """Classify every tensor and add up the bits.

    A tensor is one of three things: the codeword payload, per-linear side info,
    or a part of the model the release leaves in fp16.  Anything fitting none of
    those raises -- an unclassified tensor would silently drop out of the total,
    which is the failure this measurement exists to rule out.
    """
    acc = Accounting()
    for name, entry in manifest["header"].items():
        bits = tensor_bits(entry)
        key = linear_key(name, layout)
        if key is None:
            if name.endswith(".weight") and ("norm" in name or "embed" in name
                                             or name.startswith("lm_head")):
                acc.unquantized_bits += bits
                continue
            raise ValueError(f"unclassified tensor: {name} "
                             f"{entry['dtype']} {entry['shape']}")
        if name.endswith("." + layout.payload):
            n_out, n_in = layout.shapes[key]
            weights = n_out * n_in
            acc.n_weights += weights
            acc.payload_bits += bits
            # The claim under test, per linear: the codeword tensor holds
            # exactly two bits for every weight of the dense layer it replaces.
            per_weight = bits / weights
            acc.packing_detail[key] = {
                "n_out": n_out, "n_in": n_in,
                "payload_shape": entry["shape"], "payload_dtype": entry["dtype"],
                "bits_per_weight": per_weight,
            }
            if per_weight != 2.0:
                acc.packing_exact = False
        else:
            acc.side_bits += bits
            suffix = name.rsplit(".", 1)[1]
            acc.side_breakdown[suffix] = acc.side_breakdown.get(suffix, 0) + bits
    if acc.n_weights == 0:
        raise ValueError(f"no {layout.payload!r} tensors -- wrong repo or format")
    return acc


def by_file_size(manifest: dict, acc: Accounting) -> dict:
    """The same bits/weight, from the file size rather than the manifest.

    This is the number spec v7 asks for, and it is worth having separately
    because it cannot be fooled by a tensor left out of the classification:
    everything that is not the header and not an fp16 model parameter is charged
    to the quantized layers, padding between tensors included.
    """
    header_overhead = 8 + manifest["header_bytes"]
    counted_bytes = sum(tensor_bits(e) for e in manifest["header"].values()) // 8
    gap = manifest["file_bytes"] - header_overhead - counted_bytes
    quantized_bits = (manifest["file_bytes"] * 8
                      - header_overhead * 8
                      - acc.unquantized_bits)
    return {
        "file_bytes": manifest["file_bytes"],
        "header_bytes": header_overhead,
        "unquantized_bytes": acc.unquantized_bits // 8,
        "unattributed_bytes": gap,
        "bits_per_weight": quantized_bits / acc.n_weights,
    }


def inspect_side_info(manifest: dict, layout: Layout, layer: int = 0) -> dict:
    """Are SU/SV sign vectors, or learned scales?

    This decides whether the term is storable as a seed.  QuIP# describes SU/SV
    as random signs; if the released values are exactly +-1 they carry no
    information and cost nothing, because a generator reproduces them.  If they
    are not, they were fine-tuned and every entry has to be kept.
    """
    import numpy as np

    out = {}
    for key in ("mlp.down_proj", "self_attn.o_proj"):
        for part in ("SU", "SV"):
            name = f"model.layers.{layer}.{key}.{part}"
            if name not in manifest["header"]:
                continue
            a = fetch_tensor(name, manifest).astype(np.float32)
            mag = np.abs(a)
            out[f"{key}.{part}"] = {
                "n": int(a.size),
                "all_pm1": bool(np.all(mag == 1.0)),
                "n_unique": int(np.unique(a).size),
                "mag_min": float(mag.min()), "mag_max": float(mag.max()),
                "mag_mean": float(mag.mean()),
                "frac_within_1pct": float((np.abs(mag - 1.0) <= 0.01).mean()),
            }
    return out


# --------------------------------------------------------------------------
# what our pipeline pays instead
# --------------------------------------------------------------------------

def pipeline_side_bits(tile_size: int, survivors_per_tile: int, *,
                       n_in: int = 11008, n_out: int = 4096,
                       entry_bits: int = 16, seed_bits: int = 32) -> dict:
    """Rotation side info per *surviving* weight, for compacted survivors.

    QuIP# rotates a whole linear once, so its vectors amortize over the whole
    matrix.  We rotate each tile separately -- that is the point, each tile owns
    a different column set -- and the naive reading is that the column-side
    vector is then paid `n_out / T` times over.  Per survivor, with `k`
    survivors in a tile, that would be `b / T`: at `b = 16` an unaffordable
    1.0 bits at T=16, at `b = 1` still a quarter of the index itself.

    The measurement says we do not have to pay it, because QuIP#'s SU and SV are
    not the same kind of object.  SU (input side) is a sign vector barely moved
    off +-1 by fine-tuning; SV (output side) carries real per-channel scale.  So
    the transform separates:

        diagonal on input channels    global, n_in entries, shared by all tiles
                                      -- a diagonal commutes with the gather, so
                                      it applies before compaction
        diagonal on output channels   global, n_out entries
        rotation proper               per tile, but a random orthogonal drawn
                                      from a seed carries no payload

    Only the rotation is per-tile, and only the diagonals hold information.  The
    `1/T` term never forms.
    """
    if tile_size <= 0 or survivors_per_tile <= 0:
        raise ValueError("tile_size and survivors_per_tile must be positive")
    k, n_survivors = survivors_per_tile, n_out * survivors_per_tile
    return {
        # what a per-tile learned column vector would cost -- the term we avoid
        "per_tile_learned_fp16": entry_bits / tile_size,
        "per_tile_packed_sign": 1.0 / tile_size,
        # what the separated design costs
        "seeded_rotation": seed_bits * (n_out / tile_size) / n_survivors,
        "input_diagonal": entry_bits * n_in / n_survivors,
        "output_diagonal": entry_bits * n_out / n_survivors,
        "adopted_total": (seed_bits * (n_out / tile_size)
                          + entry_bits * n_in
                          + entry_bits * n_out) / n_survivors,
    }


def budget_impact(vq_assumed: float, vq_measured: float, budget: float,
                  tile_size: int, n_idx: int = 11008) -> dict:
    """What the correction does to the density a budget buys."""
    import accounting as A

    kw = dict(scheme="tile", budget_bits=budget, weight_bits=None,
              n_idx=n_idx, tile_size=tile_size)
    d_a = A.density_for_budget(**kw, vq_bits=vq_assumed)
    d_m = A.density_for_budget(**kw, vq_bits=vq_measured)
    return {"budget": budget, "tile_size": tile_size,
            "d_assumed": d_a, "d_measured": d_m, "delta": d_m - d_a,
            "relative": (d_m - d_a) / d_a}


# --------------------------------------------------------------------------

def run(repo: str = REPO, *, cache: Path | None = None,
        probe: bool = True) -> dict:
    if repo not in LAYOUTS:
        raise ValueError(f"no layout for {repo!r}; known: {sorted(LAYOUTS)}")
    layout = LAYOUTS[repo]
    manifest = fetch_manifest(repo, cache=cache)
    acc = account(manifest, layout)

    out = {
        "meta": {
            "utc": datetime.now(timezone.utc).isoformat(),
            "repo": repo,
            "payload_tensor": layout.payload,
            "claim_under_test": "vq_bits = idx_bits/dim = 16/8 = 2.000 exactly",
        },
        "n_quantized_weights": acc.n_weights,
        "packing_exact": acc.packing_exact,
        "packing_detail": acc.packing_detail,
        "payload_bits_per_weight": acc.payload_per_weight,
        "stored_bits_per_weight": acc.stored_per_weight,
        "side_bits_per_weight": acc.side_bits / acc.n_weights,
        "side_breakdown_bits": acc.side_breakdown,
        "unquantized_bytes": acc.unquantized_bits // 8,
        "by_file_size": by_file_size(manifest, acc),
    }
    if probe:
        out["side_info_nature"] = inspect_side_info(manifest, layout)

    out["survivors_per_tile"] = SURVIVORS_PER_TILE
    out["pipeline_side_bits"] = {
        str(t): pipeline_side_bits(t, SURVIVORS_PER_TILE)
        for t in (4, 8, 16, 32, 64)
    }
    out["budget_impact"] = [
        budget_impact(2.0, out["stored_bits_per_weight"], b, 16)
        for b in (1.75, 1.60, 1.50)
    ]
    return out


def _verdict(out: dict) -> None:
    p = out["payload_bits_per_weight"]
    s = out["stored_bits_per_weight"]
    f = out["by_file_size"]["bits_per_weight"]
    print("\n" + "=" * 70)
    print(f"  {out['meta']['repo']}")
    print(f"  quantized weights      {out['n_quantized_weights']:,}")
    print(f"  payload ({out['meta']['payload_tensor']:<8s})     {p:.6f} bits/weight    "
          f"{'EXACT' if out['packing_exact'] else 'NOT 2.0'}")
    print(f"  + side info as stored  {s:.6f} bits/weight")
    print(f"  from total file size   {f:.6f} bits/weight    "
          f"(independent, agrees to {abs(f - s):.1e})")
    print()
    print(f"  spec v7 assumes 2.000000 -> understated by "
          f"{(s - 2.0) * 1000:.3f} millibits/weight ({(s / 2.0 - 1) * 100:.3f}%)")

    nat = out.get("side_info_nature")
    if nat:
        print()
        if all(v["all_pm1"] for v in nat.values()):
            print("  SU/SV are exactly +-1 -> seed-regenerable, cost ~0")
        else:
            print("  SU/SV are NOT +-1: fine-tuned scales, every entry is real.")
            for k, v in nat.items():
                print(f"    {k:22s} |x| in [{v['mag_min']:.4f}, {v['mag_max']:.4f}], "
                      f"{v['n_unique']:>4} distinct of {v['n']:>5}, "
                      f"{v['frac_within_1pct'] * 100:.1f}% within 1% of 1")

    k = out["survivors_per_tile"]
    print()
    print("  our rotation is per tile.  bits per surviving weight, "
          f"k = {k} survivors per tile:")
    print(f"    {'T':>5}  {'per-tile fp16':>14}  {'per-tile sign':>14}"
          f"  {'separated':>11}")
    for t, v in out["pipeline_side_bits"].items():
        print(f"    {t:>5}  {v['per_tile_learned_fp16']:>14.4f}  "
              f"{v['per_tile_packed_sign']:>14.4f}  {v['adopted_total']:>11.6f}")
    print("  a per-tile learned column vector is unaffordable; a per-tile sign")
    print("  vector still costs a quarter of the index at T=16.  Neither is")
    print("  needed: the diagonals are global and only the rotation is per tile,")
    print("  and a seeded rotation carries no payload.  No 1/T term forms.")

    print()
    print("  effect of the correction on the density a budget buys (T=16):")
    for b in out["budget_impact"]:
        print(f"    B={b['budget']:.2f}   d {b['d_assumed']:.6f} -> "
              f"{b['d_measured']:.6f}   ({b['relative'] * 100:+.3f}%)")
    print("=" * 70)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=REPO, choices=sorted(LAYOUTS))
    ap.add_argument("--all", action="store_true",
                    help="measure every known release")
    ap.add_argument("--no-probe", action="store_true",
                    help="skip fetching SU/SV (the part that needs real bytes)")
    ap.add_argument("--cache-dir", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results/m0_vq_bits.json"))
    args = ap.parse_args(argv)

    repos = sorted(LAYOUTS) if args.all else [args.repo]
    results = {}
    for repo in repos:
        cache = args.cache_dir / f"manifest_{repo.split('/')[-1]}.json"
        out = run(repo, cache=cache, probe=not args.no_probe)
        _verdict(out)
        results[repo] = out

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
