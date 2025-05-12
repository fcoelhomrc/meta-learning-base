#!/usr/bin/env bash

# PACS -> ACPS
# Art, Cartoon, Photo, Sketch

# Sweep ordering
# Sketch -> Photo -> Cartoon -> Art
# (Reverse of dataset internal ordering, which is alphabetical)

# Current settings: run-20250512_050600-lzmqa6ic

BATCH_SIZE=4
echo "BATCH_SIZE = $BATCH_SIZE"

echo "TARGET = SKETCH"
python3 src/scripts/model_selection.py --baseline --source_domains 0 1 2 --num_epochs=4 --lr=4.925e-05 --beta=0.438 --weight_decay=8.7e-05 --nonlinear_classifier=true --dropout=0 --batch_size=$BATCH_SIZE

echo "TARGET = PHOTO"
python3 src/scripts/model_selection.py --baseline --source_domains 0 1 3 --num_epochs=4 --lr=4.925e-05 --beta=0.438 --weight_decay=8.7e-05 --nonlinear_classifier=true --dropout=0 --batch_size=$BATCH_SIZE

echo "TARGET = CARTOON"
python3 src/scripts/model_selection.py --baseline --source_domains 0 2 3 --num_epochs=4 --lr=4.925e-05 --beta=0.438 --weight_decay=8.7e-05 --nonlinear_classifier=true --dropout=0 --batch_size=$BATCH_SIZE

echo "TARGET = ART_PAINTING"
python3 src/scripts/model_selection.py --baseline --source_domains 1 2 3 --num_epochs=4 --lr=4.925e-05 --beta=0.438 --weight_decay=8.7e-05 --nonlinear_classifier=true --dropout=0 --batch_size=$BATCH_SIZE
