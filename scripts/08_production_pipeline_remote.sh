#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$PROJECT_ROOT/scripts/activate_remote.sh"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
LOG_DIR="$PROJECT_ROOT/outputs/logs"
STATE_PATH="$LOG_DIR/production_state.json"
CURRENT_STAGE="pipeline"
mkdir -p "$LOG_DIR" configs/generated reports/generated outputs/evaluation

write_state() {
  "$PYTHON" - "$STATE_PATH" "$1" "$2" "${3:-}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path, stage, status, detail = sys.argv[1:]
Path(path).write_text(json.dumps({
    "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
    "stage": stage,
    "status": status,
    "detail": detail,
    "process_id": os.getppid(),
}, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

is_complete() {
  "$PYTHON" - "$1/run_metadata.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
raise SystemExit(0 if path.exists() and json.loads(path.read_text(encoding="utf-8")).get("status") == "complete" else 1)
PY
}

run_training() {
  local stage="$1"
  local config="$2"
  local output="$3"
  CURRENT_STAGE="$stage"
  if is_complete "$output"; then
    write_state "$stage" skipped "A completed run already exists."
    return
  fi
  write_state "$stage" running "$config"
  local train_stage="${stage%%-*}"
  set +e
  "$PYTHON" -m classical_llm.cli train "$train_stage" --config "$config" 2>&1 | tee "$LOG_DIR/$stage.log"
  local exit_code="${PIPESTATUS[0]}"
  set -e
  if [[ "$exit_code" -ne 0 ]]; then
    write_state "$stage" failed "Exit code $exit_code; see outputs/logs/$stage.log"
    return "$exit_code"
  fi
  write_state "$stage" complete "$output"
}

fail_pipeline() {
  local exit_code=$?
  write_state "$CURRENT_STAGE" failed "Exit code $exit_code; inspect outputs/logs/$CURRENT_STAGE.log."
  exit "$exit_code"
}
trap fail_pipeline ERR

run_training cpt-05 configs/remote/cpt_05_4090d.yaml outputs/cpt/qwen2.5-0.5b-classical05
run_training cpt-20 configs/remote/cpt_20_4090d.yaml outputs/cpt/qwen2.5-0.5b-classical20
run_training cpt-50 configs/remote/cpt_50_4090d.yaml outputs/cpt/qwen2.5-0.5b-classical50

"$PYTHON" scripts/select_cpt.py | tee "$LOG_DIR/cpt-selection.log"
run_training sft configs/generated/sft_selected_4090d.yaml outputs/sft/qwen2.5-0.5b-classical
run_training reward configs/remote/reward_4090d.yaml outputs/reward/qwen2.5-0.5b-classical-rm
run_training dpo configs/remote/dpo_4090d.yaml outputs/dpo/qwen2.5-0.5b-classical
run_training grpo configs/remote/grpo_4090d.yaml outputs/grpo/qwen2.5-0.5b-classical

CURRENT_STAGE="evaluation"
write_state evaluation running "Generating five models x 400 domain responses plus perplexities."
declare -A MODELS=(
  [base]=".cache/huggingface/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
  [cpt]="$("$PYTHON" -c 'import json; r=json.load(open("reports/generated/cpt_selection.json", encoding="utf-8")); print(r["selected"]["output"] + "/best")')"
  [sft]="outputs/sft/qwen2.5-0.5b-classical/best"
  [dpo]="outputs/dpo/qwen2.5-0.5b-classical/best"
  [grpo]="outputs/grpo/qwen2.5-0.5b-classical/best"
)

for name in base cpt sft dpo grpo; do
  generation="outputs/evaluation/$name.jsonl"
  scored="outputs/evaluation/$name.scored.jsonl"
  if [[ ! -s "$generation" ]]; then
    "$PYTHON" -m classical_llm.cli generate --model "${MODELS[$name]}" \
      --dataset data/final/evaluation/domain_400.jsonl \
      --output "$generation" --max-new-tokens 384
  fi
  "$PYTHON" -m classical_llm.cli score --input "$generation" --output "$scored"
  for category in classical general_zh general_en; do
    "$PYTHON" -m classical_llm.cli perplexity --model "${MODELS[$name]}" \
      --dataset "data/final/evaluation/perplexity/$category.jsonl" \
      --output "outputs/evaluation/$name.$category.ppl.json" \
      --max-length 1024 --max-documents 200
  done
done

generations='{"base":"outputs/evaluation/base.jsonl","cpt":"outputs/evaluation/cpt.jsonl","sft":"outputs/evaluation/sft.jsonl","dpo":"outputs/evaluation/dpo.jsonl","grpo":"outputs/evaluation/grpo.jsonl"}'
"$PYTHON" -m classical_llm.cli blind-review --generations "$generations" \
  --output outputs/evaluation/blinded_review.jsonl
"$PYTHON" -m classical_llm.cli report --output reports/generated/results.json
"$PYTHON" scripts/update_readme_results.py
write_state pipeline complete "Training and automatic evaluation complete; human blind scores remain."
trap - ERR
