#!/usr/bin/env bash

# Load JSON preset values into shell variables only if they are currently empty.
# Usage:
#   load_json_preset_if_unset PRESET_FILE PRESET_NAME
load_json_preset_if_unset() {
  local preset_file="$1"
  local preset_name="$2"

  if [[ -z "$preset_name" ]]; then
    return 0
  fi
  if [[ ! -f "$preset_file" ]]; then
    echo "Error: preset file not found: $preset_file" >&2
    return 1
  fi
  if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is not installed." >&2
    return 1
  fi

  local preset_lines
  preset_lines="$(uv run python - "$preset_file" "$preset_name" <<'PY'
import json
import sys

path, name = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

if name not in data:
    print(f"ERROR: preset '{name}' not found in {path}")
    sys.exit(2)

preset = data[name]
keys = ["dtype", "compile", "batch_size", "block_size", "grad_accum", "n_layer", "n_head", "n_embd"]
for key in keys:
    if key in preset and preset[key] is not None:
        print(f"{key.upper()}={preset[key]}")
PY
)"

  if [[ "${preset_lines:-}" == ERROR:* ]]; then
    echo "$preset_lines" >&2
    return 1
  fi

  while IFS='=' read -r key value; do
    case "$key" in
      DTYPE) [[ -z "${DTYPE:-}" ]] && DTYPE="$value" ;;
      COMPILE) [[ -z "${COMPILE:-}" ]] && COMPILE="$value" ;;
      BATCH_SIZE) [[ -z "${BATCH_SIZE:-}" ]] && BATCH_SIZE="$value" ;;
      BLOCK_SIZE) [[ -z "${BLOCK_SIZE:-}" ]] && BLOCK_SIZE="$value" ;;
      GRAD_ACCUM) [[ -z "${GRAD_ACCUM:-}" ]] && GRAD_ACCUM="$value" ;;
      N_LAYER) [[ -z "${N_LAYER:-}" ]] && N_LAYER="$value" ;;
      N_HEAD) [[ -z "${N_HEAD:-}" ]] && N_HEAD="$value" ;;
      N_EMBD) [[ -z "${N_EMBD:-}" ]] && N_EMBD="$value" ;;
    esac
  done <<< "$preset_lines"
}
