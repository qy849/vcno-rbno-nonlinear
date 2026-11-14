#!/usr/bin/env bash
# Download the parameter samples, POD bases and final RBNO checkpoints into the repository.
#   bash scripts/download_data.sh <base_url> [semilinear|quasilinear|hyperelastic ...]
# e.g.
#   bash scripts/download_data.sh https://huggingface.co/datasets/<user>/<dataset>/resolve/main
#   bash scripts/download_data.sh https://huggingface.co/datasets/<user>/<dataset>/resolve/main quasilinear
# Files already present with the right checksum are skipped.
set -euo pipefail
repo_path="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
all_problems=(semilinear quasilinear hyperelastic)

usage() { sed -n '2,7p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

fetch() {  # url destination
  if command -v curl >/dev/null 2>&1; then curl -L --fail --retry 3 -o "$2" "$1"
  elif command -v wget >/dev/null 2>&1; then wget -O "$2" "$1"
  else echo "Either curl or wget is required." >&2; exit 1; fi
}

[[ $# -ge 1 ]] || { usage >&2; exit 1; }
base_url="${1%/}"; shift
if [[ $# -gt 0 ]]; then selected=("$@"); else selected=("${all_problems[@]}"); fi
for p in "${selected[@]}"; do
  [[ " ${all_problems[*]} " == *" $p "* ]] || { echo "Unknown problem: $p" >&2; usage >&2; exit 1; }
done

cd "$repo_path"
manifest="$(mktemp)"; trap 'rm -f "$manifest"' EXIT
fetch "$base_url/manifest.tsv" "$manifest"

while IFS=$'\t' read -r path checksum _bytes; do
  [[ -n "$path" ]] || continue
  # data/<family>/..., data/<exp>/..., checkpoints/<exp>/..., results/<exp>/...; <exp> starts with <family>
  name="$(echo "$path" | cut -d/ -f2)"
  keep=0; for p in "${selected[@]}"; do [[ "$name" == "$p"* ]] && keep=1; done
  (( keep )) || continue
  if [[ -f "$path" && "$(sha256sum "$path" | awk '{print $1}')" == "$checksum" ]]; then
    echo "up to date: $path"; continue
  fi
  mkdir -p "$(dirname "$path")"
  fetch "$base_url/$path" "$path.part"
  [[ "$(sha256sum "$path.part" | awk '{print $1}')" == "$checksum" ]] || { echo "Checksum mismatch: $path" >&2; exit 1; }
  mv "$path.part" "$path"
done < "$manifest"
echo "Downloaded into $repo_path/{data,checkpoints,results}"
