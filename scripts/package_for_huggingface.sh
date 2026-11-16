#!/usr/bin/env bash
# Stage the files hosted on Hugging Face (parameter samples, POD bases, final RBNO checkpoints)
# in <out_dir> under their repository paths, together with manifest.tsv (path, sha256, bytes).
#   bash scripts/package_for_huggingface.sh <out_dir> [source_dir]
# source_dir holds the generated data/, checkpoints/ and results/ (default: this repository).
# Upload <out_dir> afterwards, e.g.
#   hf upload <user>/<dataset> <out_dir> . --repo-type dataset
set -euo pipefail
repo_path="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${1:?usage: bash scripts/package_for_huggingface.sh <out_dir> [source_dir]}"
mkdir -p "$out"; out="$(cd "$out" && pwd)"
cd "${2:-$repo_path}"

files=()
for family in semilinear quasilinear hyperelastic; do
  files+=(data/$family/parameters/*)
done
for exp in semilinear_k0 semilinear_k1000 quasilinear hyperelastic; do
  files+=(data/$exp/pod/{basis.npy,eigenvalues.npy,meta.json,snapshot_rows.npy,projection_errors.npz})
  files+=(checkpoints/$exp/rbno/best_val.pt)
  files+=(results/$exp/rbno/{config.yaml,loss_history.jsonl,train_summary.json})
done

mkdir -p "$out"
: > "$out/manifest.tsv"
for f in "${files[@]}"; do
  [[ -f "$f" ]] || { echo "missing: $f" >&2; exit 1; }
  mkdir -p "$out/$(dirname "$f")"
  cp -p "$f" "$out/$f"
  printf '%s\t%s\t%s\n' "$f" "$(sha256sum "$f" | awk '{print $1}')" "$(stat -c %s "$f")" >> "$out/manifest.tsv"
done
echo "staged ${#files[@]} files in $out"
