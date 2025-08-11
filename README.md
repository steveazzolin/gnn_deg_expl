# [WIP] Explanation Shortcuts


## Relevant Commands (OLD)

```shell

# BAColorGVIsolated (SMGNN - ACR)
goodtg --config_path final_configs/BAColorGVIsol/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test
# BAColorGVIsolated (SMGNN - ACR w less reg)
goodtg --config_path final_configs/BAColorGVIsol/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --l_norm_coeff 0.1
# BAColorGVIsolated (SMGNN - ACR w 1-layer CLF)
goodtg --config_path final_configs/BAColorGVIsol/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --gnn_clf_layer 1 --entr_coeff 1.0 --l_norm_coeff 0.4
# BAColorGVIsolated (SMGNN - ACR w 2-layer CLF)
# BAColorGVIsolated (SMGNN - GIN)
goodtg --config_path final_configs/BAColorGVIsol/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --backbone GIN --l_norm_coeff 0.2


# BAColorGVIsolated (GSAT - ACR)
 goodtg --config_path final_configs/BAColorGVIsol/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --gnn_clf_layer 0 --entr_coeff 0.1 --l_norm_coeff 0.4
 # BAColorGVIsolated (GSAT - ACR w 1-layer CLF)
 goodtg --config_path final_configs/BAColorGVIsol/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --gnn_clf_layer 1 --entr_coeff 0.1 --l_norm_coeff 0.4
 # BAColorGVIsolated (GSAT - ACR w 1-layer CLF - Pretrain Degenrate)
 goodtg --config_path final_configs/BAColorGVIsol/basis/no_shift/GSAT.yaml --seeds "2/3/4/5" --task hist --gnn_clf_layer 1 --pretrain_degenerate --entr_coeff 0.1 --l_norm_coeff 0.4


# BAColorGVIsolated (DIR - ACR k=0.1)
goodtg --config_path final_configs/BAColorGVIsol/basis/no_shift/DIR.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --ood_param 0.1
# BAColorGVIsolated (DIR - ACR k=0.5)
goodtg --config_path final_configs/BAColorGVIsol/basis/no_shift/DIR.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --ood_param 0.5
# BAColorGVIsolated (DIR - ACR k=0.1 w 1-layer CLF)
goodtg --config_path final_configs/BAColorGVIsol/basis/no_shift/DIR.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --ood_param 0.1 --gnn_clf_layer 1
# BAColorGVIsolated (DIR - ACR k=0.5 w 1-layer CLF)
goodtg --config_path final_configs/BAColorGVIsol/basis/no_shift/DIR.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --ood_param 0.1 --gnn_clf_layer 1
# BAColorGVIsolated (DIR - ACR k=0.1 w 1-layer CLF - Pretrain Degenerate)
goodtg --config_path final_configs/BAColorGVIsol/basis/no_shift/DIR.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --ood_param 0.1 --gnn_clf_layer 1 --pretrain_degenerate
# BAColorGVIsolated (DIR - ACR k=0.5 w 1-layer CLF - Pretrain Degenerate)
goodtg --config_path final_configs/BAColorGVIsol/basis/no_shift/DIR.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --ood_param 0.5 --gnn_clf_layer 1 --pretrain_degenerate








## Relevant Commands (NEW)

```shell

# BAColorGVIsolated (SMGNN - ACR) (1/2 no deg; 3 deg)
goodtg --config_path final_configs/BAColorGVIsol/basis/no_shift/SMGNN.yaml --seeds "1/2/3" --task test --l_norm_coeff 0.4
# BAColorGVIsolated (SMGNN - ACR w less reg)
# BAColorGVIsolated (SMGNN - ACR w 1-layer CLF)
# BAColorGVIsolated (SMGNN - ACR w 2-layer CLF)
# BAColorGVIsolated (SMGNN - GIN)


# BAColorGVIsolated (GSAT - ACR)
# BAColorGVIsolated (GSAT - ACR w 1-layer CLF)
# BAColorGVIsolated (GSAT - ACR - As per paper with lin. clf. - Pretrain Degenerate/Suff)
goodtg --config_path final_configs/BAColorGVIsol/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5" --task test --pretrain degenerate
goodtg --config_path final_configs/BAColorGVIsol/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5" --task test --pretrain suff
goodtg --config_path final_configs/BAColorGVIsol/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5" --task test --pretrain degenerate --backbone ACR2
goodtg --config_path final_configs/BAColorGVIsol/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5" --task test --pretrain suff --backbone ACR2
goodtg --config_path final_configs/BAColorRBIsol/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5" --task train --pretrain sub
goodtg --config_path final_configs/BAColorRBIsol/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5" --task train --pretrain sub --backbone ACR2

goodtg --config_path final_configs/BAColorGVIsol/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5" --task test --pretrain degenerate
goodtg --config_path final_configs/BAColorGVIsol/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5" --task test --pretrain suff
goodtg --config_path final_configs/BAColorRBIsol/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5" --task test --pretrain sub
goodtg --config_path final_configs/BAColorGVIsol/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5" --task test --pretrain degenerate --backbone ACR2
goodtg --config_path final_configs/BAColorGVIsol/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5" --task test --pretrain suff --backbone ACR2

# DIR pretrain experiments are run with both optimizing DIR loss and without
goodtg --config_path final_configs/BAColorGVIsol/basis/no_shift/DIR.yaml --seeds "1/2/3/4/5" --task test --pretrain degenerate --backbone ACR2 --ood_param 0.5
goodtg --config_path final_configs/BAColorGVIsol/basis/no_shift/DIR.yaml --seeds "1/2/3/4/5" --task test --pretrain suff --backbone ACR2  --ood_param 0.5
goodtg --config_path final_configs/BAColorGVIsol/basis/no_shift/DIR.yaml --seeds "1/2/3/4/5" --task test --pretrain degenerate --backbone ACR2 --ood_param 0.3
goodtg --config_path final_configs/BAColorGVIsol/basis/no_shift/DIR.yaml --seeds "1/2/3/4/5" --task test --pretrain suff --backbone ACR2  --ood_param 0.3
goodtg --config_path final_configs/BAColorRBIsol/basis/no_shift/DIR.yaml --seeds "1/2/3/4/5" --task test --pretrain sub --backbone ACR2 --ood_param 0.5
goodtg --config_path final_configs/BAColorRBIsol/basis/no_shift/DIR.yaml --seeds "1/2/3/4/5" --task test --pretrain sub --backbone ACR2 --ood_param 0.3


# Config to measure how often SEGNNs yield poor unfaithful explanations
goodtg --config_path final_configs/BAColorGVIsol/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5" --task test --entr_coeff 0.1 --l_norm_coeff 0.4 --backbone ACR2 --use_readout_norm none --gnn_clf_layer 2 # squashed expls
goodtg --config_path final_configs/BAColorGVIsol/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5" --task test --entr_coeff 1.0 --l_norm_coeff 0.4 --backbone ACR2 --use_readout_norm bn --gnn_clf_layer 2 # 3/5 deg
goodtg --config_path final_configs/BAColorGVIsol/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5" --task test --entr_coeff 1.0 --l_norm_coeff 0.4 --backbone ACR2 --use_readout_norm bn --gnn_clf_layer 1

# same as above, but by changing only ood_param similar to the analysis for MNIST
goodtg --config_path final_configs/BAColorGVIsol/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5" --task test --entr_coeff 1.0 --l_norm_coeff 0.4 --backbone ACR2 --use_readout_norm bn --gnn_clf_layer 2 # (default) --ood_param 1 # 3 over 5 are deg
goodtg --config_path final_configs/BAColorGVIsol/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5" --task test --entr_coeff 1.0 --l_norm_coeff 0.4 --backbone ACR2 --use_readout_norm bn --gnn_clf_layer 2 --ood_param 0.1
goodtg --config_path final_configs/BAColorGVIsol/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5" --task test --entr_coeff 1.0 --l_norm_coeff 0.4 --backbone ACR2 --use_readout_norm bn --gnn_clf_layer 2 --ood_param 0.01 #TODO

goodtg --config_path final_configs/BAColorGVIsol/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5" --task test --backbone ACR2



##
# MNIST
##
goodtg --config_path final_configs/MNIST/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5" --task test
goodtg --config_path final_configs/MNIST/basis/no_shift/GSAT.yaml --seeds "1" --task test --gpu_idx 1 --ood_param 0.5 #(r=0.5)
goodtg --config_path final_configs/MNIST/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --pretrain degenerate --backbone ACR
goodtg --config_path final_configs/MNIST/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --pretrain suff --backbone ACR
goodtg --config_path final_configs/MNIST/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --pretrain degenerate --backbone ACR2
goodtg --config_path final_configs/MNIST/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --pretrain suff  --backbone ACR2

goodtg --config_path final_configs/MNIST/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --pretrain degenerate --backbone ACR
goodtg --config_path final_configs/MNIST/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --pretrain suff --backbone ACR
goodtg --config_path final_configs/MNIST/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --pretrain degenerate --backbone ACR2
goodtg --config_path final_configs/MNIST/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --pretrain suff  --backbone ACR2

goodtg --config_path final_configs/MNIST/basis/no_shift/DIR.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --backbone ACR2 --ood_param 0.8 --use_readout_norm bn #(Acc0.5)
goodtg --config_path final_configs/MNIST/basis/no_shift/DIR.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --backbone ACR2 --ood_param 0.1 --use_readout_norm bn #(Acc0.2)
goodtg --config_path final_configs/MNIST/basis/no_shift/DIR.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --backbone ACR2 --ood_param 0.8  #(Acc0.42)
goodtg --config_path final_configs/MNIST/basis/no_shift/DIR.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --backbone ACR2 --ood_param 0.1  #(Acc0.2)
goodtg --config_path final_configs/MNIST/basis/no_shift/DIR.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --pretrain degenerate --backbone ACR2 --ood_param 0.8 --use_readout_norm bn # (Acc 0.95)
goodtg --config_path final_configs/MNIST/basis/no_shift/DIR.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --pretrain degenerate --backbone ACR2 --ood_param 0.1 --use_readout_norm bn # (Acc 89)
goodtg --config_path final_configs/MNIST/basis/no_shift/DIR.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --pretrain degenerate --backbone ACR2 --ood_param 0.8 --use_readout_norm none # (f1_pos does not reach ~1)
goodtg --config_path final_configs/MNIST/basis/no_shift/DIR.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --pretrain suff --backbone ACR2 --ood_param 0.8 --use_readout_norm bn #(Acc 0.5)
goodtg --config_path final_configs/MNIST/basis/no_shift/DIR.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --pretrain suff --backbone ACR2 --ood_param 0.8 --use_readout_norm none #(Acc 0.89)
goodtg --config_path final_configs/MNIST/basis/no_shift/DIR.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --pretrain suff --backbone ACR2 --ood_param 0.1 --use_readout_norm none #(Acc 0.47)
 


# --ood_param 0.1  --l_norm_coeff 1   --entr_coeff 5   (deg)   Test ACC=  (4 layers both DET and CLF)
# --ood_param 0.1  --l_norm_coeff 1   --entr_coeff 5   (deg)   Test ACC=0.89
# --ood_param 0.1  --l_norm_coeff 0.1 --entr_coeff 5   (deg)   Test ACC=0.87/0.91/0.84
# --ood_param 0.1  --l_norm_coeff 1   --entr_coeff 1.5 (plaus) Test ACC=0.91
# --ood_param 0.1  --l_norm_coeff 1   --entr_coeff 1   (plaus) Test ACC=0.93
# --ood_param 0.1  --l_norm_coeff 0.1 --entr_coeff 1   (plaus) Test ACC=0.93
# --ood_param 0.1  --l_norm_coeff 0.1 --entr_coeff 10  (plaus) Test ACC=0.88
# --ood_param 0.1  --l_norm_coeff 0.1 --entr_coeff 4   (plaus) Test ACC=0.91
# --ood_param 0.1  --l_norm_coeff 0.1 --entr_coeff 0.1 (plaus) 0.93
# --ood_param 0.1  --l_norm_coeff 0.0 --entr_coeff 5   (plaus) Test ACC=0.89
# TRAIN WITH 4 LAYERS AND STRONG REG.
goodtg --config_path final_configs/MNIST/basis/no_shift/SMGNN.yaml --seeds "1" --task test --gpu_idx 0


# MNIST with natural degenerate explanations
#goodtg --config_path final_configs/MNIST/basis/no_shift/SMGNN.yaml --seeds "1/2" --task test --gpu_idx 0 --ood_param 0.1  --l_norm_coeff 0.1 --entr_coeff 5 --model_layer 3 --gnn_clf_layer 3 # seed 1 deg. seed 2 all 1.0
# goodtg --config_path final_configs/MNIST/basis/no_shift/SMGNN.yaml --seeds "1/2" --task test --gpu_idx 1 --ood_param 0.1  --l_norm_coeff 0.1 --entr_coeff 5 --model_layer 3 --gnn_clf_layer 3 --backbone ACR2 # seed 1 very large explanation. seed 2 all 1.0
# goodtg --config_path final_configs/MNIST/basis/no_shift/SMGNN.yaml --seeds "1/2" --task test --gpu_idx 1 --ood_param 0.1  --l_norm_coeff 1 --entr_coeff 5 --backbone ACR2 # seed 1 very large explanation. seed 2 all 1.0

goodtg --config_path final_configs/MNIST/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --backbone ACR2 # checkme
goodtg --config_path final_configs/MNIST/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --ood_param 0.1 --backbone ACR2 # seed 1 some traces of deg. seed 2 not deg. seed 3/4 not deg. seed 5 some signs of deg
goodtg --config_path final_configs/MNIST/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --ood_param 1 --backbone ACR2 # seed 1/2/3 deg.

goodtg --config_path final_configs/MNIST/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --backbone ACR2 # all seed no deg
goodtg --config_path final_configs/MNIST/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --backbone ACR2 --extra_param False 10 0.3 # no deg
goodtg --config_path final_configs/MNIST/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --backbone ACR2 --mitigation_sampling raw # (NOTE: filename saved with tmp=no_r_annealing) no param sharing + no r annealing -> no deg
goodtg --config_path final_configs/MNIST/basis/no_shift/GSAT.yaml --seeds "1/2/3" --task test --gpu_idx 1 --backbone ACR2 --mitigation_sampling raw --ood_param 1 --extra_param False 10 0.5 # (NOTE: filename saved with tmp=no_r_annealing) no param sharing + no r annealing -> no deg
goodtg --config_path final_configs/MNIST/basis/no_shift/GSAT.yaml --seeds "1/2/3" --task test --gpu_idx 1 --backbone ACR2 --mitigation_sampling raw --ood_param 10 --extra_param False 10 0.5 # (NOTE: filename saved with tmp=no_r_annealing) no param sharing + no r annealing -> no deg
goodtg --config_path final_configs/MNIST/basis/no_shift/GSAT.yaml --seeds "1" --task test --gpu_idx 1 --backbone ACR2 --mitigation_sampling raw --ood_param 1 --extra_param False 10 0.5 --gnn_clf_layer 1 # (NOTE: filename saved with tmp=no_r_annealing) no param sharing + no r annealing -> no deg

# Trying to train a good performing DIR with k=0.1. If we manage, it would probably be a deg. explanation
goodtg --config_path final_configs/MNIST/basis/no_shift/DIR.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 0 --backbone ACR2 --ood_param 0.1 --use_readout_norm bn # 0.2
goodtg --config_path final_configs/MNIST/basis/no_shift/DIR.yaml --seeds "1" --task test --gpu_idx 0 --backbone ACR2 --ood_param 0.1 --use_readout_norm bn --model_layer 3 --gnn_clf_layer 3 # running



##
# CPatchMNIST
##
goodtg --config_path final_configs/CPatchMNIST/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --pretrain suff --backbone ACR2
goodtg --config_path final_configs/CPatchMNIST/basis/no_shift/GSAT.yaml --seeds "2" --task plot_explanations --gpu_idx 1 --pretrain suff --backbone ACR2



##
# CPatchMNIST2
##
goodtg --config_path final_configs/CPatchMNIST2/basis/no_shift/SMGNN.yaml --seeds "1" --task test --gpu_idx 1 --pretrain suff --backbone ACR2
goodtg --config_path final_configs/CPatchMNIST2/basis/no_shift/GSAT.yaml --seeds "1" --task test --gpu_idx 1 --pretrain suff --backbone ACR2







##
# Mutagenicity (--backbone ACR2 by default)
##
goodtg --config_path final_configs/MUTAG/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1
goodtg --config_path final_configs/MUTAG/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1
goodtg --config_path final_configs/MUTAG/basis/no_shift/DIR.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --ood_param 0.5
goodtg --config_path final_configs/MUTAG/basis/no_shift/DIR.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --ood_param 0.1

goodtg --config_path final_configs/MUTAG/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --pretrain degenerate
goodtg --config_path final_configs/MUTAG/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --pretrain degenerate
goodtg --config_path final_configs/MUTAG/basis/no_shift/DIR.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --ood_param 0.5 --pretrain degenerate


##
# GraphSST2Planted (--backbone ACR2 by default)
##

goodtg --config_path final_configs/SST2Planted/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1
goodtg --config_path final_configs/SST2Planted/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --pretrain degenerate

goodtg --config_path final_configs/SST2Planted/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1
goodtg --config_path final_configs/SST2Planted/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --pretrain degenerate
```