N=600
S=2
noise=0.0
data=COVID
r=100
fea_dim=200
mdl=LeNet5
num=4
gpu=2
CUDA_VISIBLE_DEVICES=$gpu python runC.py --n_clients $num --dataset $data --model $mdl --feature_dim $fea_dim --rounds $r --seed $S --preset IL --data_size $N --noise $noise
CUDA_VISIBLE_DEVICES=$gpu python runC.py --n_clients $num --dataset $data --model $mdl --feature_dim $fea_dim --rounds $r --seed $S --preset FD --data_size $N --noise $noise --kd_type feature
# CUDA_VISIBLE_DEVICES=$gpu python runC.py --n_clients $num --dataset $data --model $mdl --feature_dim $fea_dim --rounds $r --seed $S --preset FD --data_size $N --noise $noise --kd_type output

echo "----------------------------------------------------"

noise=0.2
CUDA_VISIBLE_DEVICES=$gpu python runC.py --n_clients $num --dataset $data --model $mdl --feature_dim $fea_dim --rounds $r --seed $S --preset IL --data_size $N --noise $noise
CUDA_VISIBLE_DEVICES=$gpu python runC.py --n_clients $num --dataset $data --model $mdl --feature_dim $fea_dim --rounds $r --seed $S --preset FD --data_size $N --noise $noise --kd_type feature
# CUDA_VISIBLE_DEVICES=$gpu python runC.py --n_clients $num --dataset $data --model $mdl --feature_dim $fea_dim --rounds $r --seed $S --preset FD --data_size $N --noise $noise --kd_type output

echo "----------------------------------------------------"

noise=0.4
CUDA_VISIBLE_DEVICES=$gpu python runC.py --n_clients $num --dataset $data --model $mdl --feature_dim $fea_dim --rounds $r --seed $S --preset IL --data_size $N --noise $noise
CUDA_VISIBLE_DEVICES=$gpu python runC.py --n_clients $num --dataset $data --model $mdl --feature_dim $fea_dim --rounds $r --seed $S --preset FD --data_size $N --noise $noise --kd_type feature
# CUDA_VISIBLE_DEVICES=$gpu python runC.py --n_clients $num --dataset $data --model $mdl --feature_dim $fea_dim --rounds $r --seed $S --preset FD --data_size $N --noise $noise --kd_type output

echo "----------------------------------------------------"

noise=0.6
CUDA_VISIBLE_DEVICES=$gpu python runC.py --n_clients $num --dataset $data --model $mdl --feature_dim $fea_dim --rounds $r --seed $S --preset IL --data_size $N --noise $noise
CUDA_VISIBLE_DEVICES=$gpu python runC.py --n_clients $num --dataset $data --model $mdl --feature_dim $fea_dim --rounds $r --seed $S --preset FD --data_size $N --noise $noise --kd_type feature
# CUDA_VISIBLE_DEVICES=$gpu python runC.py --n_clients $num --dataset $data --model $mdl --feature_dim $fea_dim --rounds $r --seed $S --preset FD --data_size $N --noise $noise --kd_type output

echo "----------------------------------------------------"