#!/usr/bin/env bash
set -euo pipefail

# Master installer: installs Python deps for MAGIC/AutoClass + runs the R dependency installer.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"
autoclass_dir="${AUTOCLASS_DIR:-${script_dir}/AutoClass}"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "Python executable not found: ${python_bin}" >&2
  exit 1
fi

packages=(
  "rds2py:rds2py"
  "magic:magic-impute"
  "numpy:numpy"
  "pandas:pandas"
  "sklearn:scikit-learn"
  "matplotlib:matplotlib"
  "tensorflow:tensorflow"
)

missing=()
for entry in "${packages[@]}"; do
  import_name="${entry%%:*}"
  pip_name="${entry#*:}"

  if [[ "${import_name}" == "magic" ]]; then
    if ! "${python_bin}" - <<'PY'
import sys
try:
    import magic
    ok = hasattr(magic, "MAGIC")
except Exception:
    ok = False
sys.exit(0 if ok else 1)
PY
    then
      missing+=("${pip_name}")
    fi
  else
    if ! "${python_bin}" - <<PY
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("${import_name}") is not None else 1)
PY
    then
      missing+=("${pip_name}")
    fi
  fi
done

if ((${#missing[@]})); then
  echo "Installing missing Python packages: ${missing[*]}"
  "${python_bin}" -m pip install --upgrade "${missing[@]}"
else
  echo "Python dependencies already satisfied."
fi

if ! command -v git >/dev/null 2>&1; then
  echo "git is required to clone AutoClass. Please install git and rerun." >&2
  exit 1
fi

if [[ -d "${autoclass_dir}/.git" ]]; then
  echo "AutoClass already cloned at ${autoclass_dir}"
else
  echo "Cloning AutoClass into ${autoclass_dir}"
  git clone https://github.com/datapplab/AutoClass.git "${autoclass_dir}"
fi

Rscript "${script_dir}/install_imputation_libs.R"
