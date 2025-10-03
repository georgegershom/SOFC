#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

HR="${1:-hr4}"
JOB="sofc_${HR}"

if ! command -v abaqus >/dev/null 2>&1; then
  echo "[ERROR] Abaqus CLI 'abaqus' not found in PATH."
  echo "To run locally:"
  echo "  cd $(pwd) && abaqus cae noGUI=build_model.py -- --hr ${HR} --job ${JOB}"
  exit 1
fi

echo "Running Abaqus noGUI build for job ${JOB} with ${HR} ..."
abaqus cae noGUI=build_model.py -- --hr "${HR}" --job "${JOB}"

echo "Done. Outputs under $(pwd)/jobs/${JOB}/ and in the working directory."

