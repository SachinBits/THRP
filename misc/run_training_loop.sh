#!/bin/bash

while true; do
  echo "Starting generalist training..."
  python3 scripts/train_generalist_kaggle.py \
    --epochs 5 --device 0 --batch-size 32

  echo "Generalist finished. Restarting in 10 seconds..."
  sleep 10

  echo "Starting specialist training..."
  python3 scripts/train_specialist_kaggle.py \
    --all --epochs 5 --device 0 --batch-size 32

  echo "Specialist finished. Restarting in 10 seconds..."
  sleep 10
done