# ABAE: Auxiliary Balanced AutoEncoder for Class-imbalanced Semi-supervised Learning

Code for the paper entitled "ABAE: Auxiliary Balanced AutoEncoder for Class-imbalanced Semi-supervised Learning"
-------------------------------------------------------------------------------------------------------------------------------------------------------
Dependencies:
python 3.8
torch 2.0.1+cu117 
torchvision 0.15.2+cu117
numpy 1.24.4
scipy 1.10.1
randAugment (pip install git+https://github.com/ildoonet/pytorch-randaugment),(if an error occurs, type apt-get install git)
tensorboardX (pip install tensorboad)
matplotlib (pip install matplotlib)
progress (pip install progress)
-------------------------------------------------------------------------------------------------------------------------------------------------------
#Train:
For example, if you want to run ABAE.py with 0th gpu, ratio of labeled data as 20%
for svhn dataset:
N1(number of data points belonging to first class = num_max) as 1000 , ratio of imbalance as 100, 500 epoch with each epoch 500 iteration, manualseed as 0, dataset as SVHN, imbalance type with long tailed imbalance:
python ABAE.py --gpu 1 --label_ratio 20 --num_max 1000 --imb_ratio 100 --epoch 500 --val-iteration 500 --manualSeed 0 --dataset svhn --imbalancetype long 
for cifar10 dataset:
N1(number of data points belonging to first class = num_max) as 1000 , ratio of imbalance as 100, 500 epoch with each epoch 500 iteration, manualseed as 0, dataset as cifar10, imbalance type with long tailed imbalance:
python ABAE.py --gpu 4 --label_ratio 20 --num_max 1000 --imb_ratio 100 --epoch 500 --val-iteration 500 --manualSeed 0 --dataset cifar10 --imbalancetype long 
for cifar100 dataset:
N1(number of data points belonging to first class = num_max) as 100, ratio of imbalance as 100, 500 epoch with each epoch 500 iteration, manualseed as 0, dataset as cifar100, imbalance type with long tailed imbalance:
python ABAE.py --gpu 5 --label_ratio 20 --num_max 100 --imb_ratio 20 --epoch 500 --val-iteration 500 --manualSeed 0 --dataset cifar100 --imbalancetype long 
-------------------------------------------------------------------------------------------------------------------------------------------------------

These codes validate peformance of algorithms on testset after each epoch of training

-------------------------------------------------------------------------------------------------------------------------------------------------------

Performance of algorithms are summarized in the paper.
