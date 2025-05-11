#!/usr/bin/env bash

BETAS=(0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0)

# Iterate and call Python script
for BETA in "${BETAS[@]}"; do
  echo "Running script with beta=$BETA"
  python3 src/scripts/train.py --num_epochs=4 --batch_size=16 \
    --nonlinear_classifier=false --dropout=0.1 \
    --lr=1e-5 --weight_decay=0 --beta="$BETA"
done