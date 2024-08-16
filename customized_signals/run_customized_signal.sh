#!/bin/bash

# List of arguments
# py customized_signal.py --stock_name MCD --train_window_size 10 --train_start 2023-01-01 --train_end 2023-09-10 
# --test_window_size 10 --learn_iteration 100 --model_name A2C


stock_name="MCD"
train_window_sizes=("10" "50" "100")
train_start="2023-01-01"
train_end="2023-09-10 "
test_window_size="10"
learn_iteration="100"
model_name="A2C"

# Loop through the arguments
for train_window_size in "${train_window_sizes[@]}"; do
    python customized_signal.py --stock_name "$stock_name" --train_window_size "$train_window_size" --train_start "$train_start" --train_end "$train_end" --test_window_size "$test_window_size" --learn_iteration "$learn_iteration" --model_name "$model_name"
done