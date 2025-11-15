#!/usr/bin/env bash
set -euo pipefail

# ===============================
# 基本配置
# ===============================
VENV_NAME="dela"            # 虚拟环境名
GPU_ID="0"                  # GPU id
TRAIN_SCRIPT="train.py"     # 训练脚本
PYTHON_BIN="python"         # python 命令
JOBS_FILE=""                # 可选 jobs 文件

# 默认任务（model,config,cur_id）
JOBS=(
  "glipe200,config,200.0"
  "glipe200,config,200.1"
)

# ===============================
# 激活虚拟环境（只激活 dela）
# ===============================
echo "Activating virtual environment: $VENV_NAME"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$VENV_NAME"
elif [[ -f "$HOME/.virtualenvs/$VENV_NAME/bin/activate" ]]; then
  source "$HOME/.virtualenvs/$VENV_NAME/bin/activate"
elif [[ -f "./venv/bin/activate" ]]; then
  source "./venv/bin/activate"
else
  echo "❌ 无法找到虚拟环境 $VENV_NAME，请检查 conda 或 venv 路径"
  exit 1
fi
echo "✅ 虚拟环境已激活: $(which python)"

# ===============================
# 设置 GPU
# ===============================
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
echo "Using GPU (CUDA_VISIBLE_DEVICES) = ${CUDA_VISIBLE_DEVICES}"

# ===============================
# 检查训练脚本
# ===============================
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "❌ ERROR: train script not found: $TRAIN_SCRIPT"
  exit 2
fi

# ===============================
# 从 jobs.txt 读取任务（可选）
# ===============================
if [[ -n "$JOBS_FILE" ]]; then
  if [[ ! -f "$JOBS_FILE" ]]; then
    echo "❌ ERROR: jobs file not found: $JOBS_FILE"
    exit 3
  fi
  mapfile -t JOBS < <(grep -v '^\s*$' "$JOBS_FILE" | grep -v '^\s*#' | sed 's/\r$//')
fi

# ===============================
# 解析任务
# ===============================
parse_job() {
  local line="$1"
  IFS=',' read -r m c id <<<"$line"
  echo "$m" "$c" "$id"
}

# ===============================
# 检测 train.py 是否支持 CLI 参数
# ===============================
supports_cli_args=false
if grep -qE -- "--model|argparse" "$TRAIN_SCRIPT"; then
  supports_cli_args=true
fi

echo "Train script: $TRAIN_SCRIPT"
echo "Python: $PYTHON_BIN"
echo "Jobs to run: ${#JOBS[@]}"
echo "train.py supports CLI args (--model/...)? -> $supports_cli_args"
echo

# ===============================
# 运行任务
# ===============================
run_with_args() {
  local m="$1"; local c="$2"; local id="$3"
  mkdir -p "output/log/${id}" "output/model/${id}"
  STDOUT_LOG="output/log/${id}/out.log"
  STDERR_LOG="output/log/${id}/err.log"
  echo "=== Running model=$m config=$c cur_id=$id ==="
  echo "Logs -> $STDOUT_LOG , $STDERR_LOG"
  "$PYTHON_BIN" -u "$TRAIN_SCRIPT" --model "$m" --config "$c" --cur_id "$id" >"$STDOUT_LOG" 2>"$STDERR_LOG"
  echo "=== Finished job $id ==="
}

# ===============================
# 主循环
# ===============================
for jobline in "${JOBS[@]}"; do
  [[ -z "${jobline// }" ]] && continue
  read -r model config cur_id <<<"$(parse_job "$jobline")"
  echo "→ Running job: model=$model, config=$config, cur_id=$cur_id"
  run_with_args "$model" "$config" "$cur_id"
done

echo "✅ All jobs finished successfully."
