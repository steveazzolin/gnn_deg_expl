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

```