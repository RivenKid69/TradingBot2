#!/usr/bin/env bash
set -euo pipefail

LOGDIR="wsl_diag_logs"
mkdir -p "$LOGDIR"

echo "==[ ENV / OS ]=="        | tee "$LOGDIR/env.txt"
uname -a                      | tee -a "$LOGDIR/env.txt"
cat /etc/os-release           | tee -a "$LOGDIR/env.txt" || true
echo "==[ WSL hint ]=="       | tee -a "$LOGDIR/env.txt"
grep -E 'Microsoft|WSL' /proc/version || true | tee -a "$LOGDIR/env.txt"

# Активируем venv, если есть
if [ -d ".venv" ]; then
  echo "Activating .venv" | tee -a "$LOGDIR/env.txt"
  source .venv/bin/activate
fi

echo "==[ PYTHON ]=="          | tee -a "$LOGDIR/env.txt"
which python3                 | tee -a "$LOGDIR/env.txt"
python3 -V                   | tee -a "$LOGDIR/env.txt"
python3 -c "import sys;print(sys.executable)" | tee -a "$LOGDIR/env.txt"
echo "==[ pip pkgs ]=="        | tee -a "$LOGDIR/env.txt"
python3 -m pip freeze | grep -Ei '^(optuna|pandas|pyarrow|numpy|pyyaml|lightgbm|xgboost|catboost)=' || true | tee -a "$LOGDIR/env.txt"

echo "==[ System resources ]=="| tee -a "$LOGDIR/env.txt"
locale                        | tee -a "$LOGDIR/env.txt" || true
ulimit -n                     | tee -a "$LOGDIR/env.txt" || true
df -h .                       | tee -a "$LOGDIR/env.txt"
free -h                       | tee -a "$LOGDIR/env.txt"

# Проверка, что мы в корне проекта
if [ ! -f "train_model_multi_patch.py" ]; then
  echo "❌ Запусти скрипт из корня репозитория (где train_model_multi_patch.py)." | tee -a "$LOGDIR/env.txt"
  exit 1
fi

# Поиск конфигов и грубая проверка обратных слэшей (Windows-пути) в yaml
echo "==[ YAML sanity ]=="     | tee "$LOGDIR/yaml_sanity.txt"
grep -RIn --include="*.yaml" "\\\\" configs || true | tee -a "$LOGDIR/yaml_sanity.txt"

# Запуск Python-диагностики
python3 diag_val_split.py \
  --config configs/config_train_spot_bar.yaml \
  --logdir "$LOGDIR" || true

# Упакуем бандл
BUNDLE="support_bundle_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$BUNDLE" "$LOGDIR"
echo "✅ Готово. Support bundle: $BUNDLE"
echo "   Внутри: env.txt, yaml_sanity.txt, diag_val_split.json/csv/txt"
