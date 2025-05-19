#!/usr/bin/env bash

SEEDS=(0 1 2 3 4 5 6 7 8 9)


# Iterate and call Python script
for SEED in "${SEEDS[@]}"; do
  echo "Running script with seed=$SEED"
  python3 src/scripts/few_shot.py --seed="$SEED" \
 --checkpoint_baseline="dump/devout-rain-112_val.ckpt" --checkpoint_model="dump/wobbly-disco-103_val.ckpt"\
 --batch_size=16\
 --nonlinear_classifier=true --dropout=0.\
 --lr=5e-05 --weight_decay=0.
done

