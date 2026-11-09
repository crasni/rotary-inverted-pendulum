#!/bin/bash
# Simulated curriculum pilot: collect (stochastic) -> update -> deterministic eval, repeated.
set -e
R=runs/curriculum_sim/pilot_$1; STAGE=$2; ROUNDS=$3; LR=${4:-3e-5}; UTD=${5:-2}
mkdir -p $R
PY="env PYTHONPATH=src:scripts .venv/bin/python scripts/rig_curriculum.py"
CK=runs/sac_finetune/V1/policy.zip
$PY collect --checkpoint $CK --output $R/eval_v0 --stage $STAGE --episodes 8 --deterministic --sim --seed 100 | tail -0
$PY collect --checkpoint $CK --output $R/eval_v0_hang --stage hanging --episodes 4 --deterministic --sim --seed 200 | tail -0
BATCHES=""
for i in $(seq 1 $ROUNDS); do
  $PY collect --checkpoint $CK --output $R/train_$i --stage $STAGE --episodes 10 --sim --seed $i | tail -0
  BATCHES="$BATCHES $R/train_$i"
  WARM=0; [ $i = 1 ] && WARM=2000
  N=$(ls $R/train_$i/transitions_*.npz | wc -l)
  $PY update --checkpoint $CK --batches $BATCHES --output $R/v$i --gradient-steps $((N*300*UTD)) --learning-rate $LR --critic-warmup $WARM --seed $i | tail -1
  CK=$R/v$i/policy.zip
  $PY collect --checkpoint $CK --output $R/eval_v$i --stage $STAGE --episodes 8 --deterministic --sim --seed 100 | tail -0
  $PY collect --checkpoint $CK --output $R/eval_v${i}_hang --stage hanging --episodes 4 --deterministic --sim --seed 200 | tail -0
done
$PY report $R/eval_v* 
