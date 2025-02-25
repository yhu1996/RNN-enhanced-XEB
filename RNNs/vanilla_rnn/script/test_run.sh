
cd RNNs

# Define the values you want to pass
M=1000
batch_size=100
val_ratio=0.2
N_h=10
N_epochs=100
dropout_flag=True
dropout_rate=0.1

python3 vanilla_rnn/script/example_train.py "$M" "$batch_size" "$val_ratio" "$N_h" "$N_epochs" "$dropout_flag" "$dropout_rate"
