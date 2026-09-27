#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python run_main.py  > main.log  2>&1; echo "MAIN  exit=$?"
python run_abl.py   > abl.log   2>&1; echo "ABL   exit=$?"
python run_bench.py > bench.log 2>&1; echo "BENCH exit=$?"
python run_sens.py  > sens.log  2>&1; echo "SENS  exit=$?"
echo ALL_DONE
