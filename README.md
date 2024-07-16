# surrogate-search

python train.py --arch resnet18 --dataset CIFAR10 --use_dynamic_surrogate 0 --epochs 300 --model_learning_rate 0.01 --dist_learning_rate 0.0001 --timesteps 5 --batch_size 64 --initial_temp 0.5