# set -e

echo "Time to train models!"

GPU=1
SEEDS="1/2/3"



# for DATASET in SST2/basis/no_shift; do
       # goodtg --config_path final_configs/${DATASET}/DIR.yaml \
       #        --seeds ${SEEDS} \
       #        --task train \
       #        --backbone ACR2 \
       #        --ood_param 0.8 \
       #        --use_readout_norm bn \
       #        --gpu_idx ${GPU}
       # echo "DONE TRAIN DIR 0.8 ${DATASET}"

       # goodtg --config_path final_configs/${DATASET}/DIR.yaml \
       #        --seeds ${SEEDS} \
       #        --task train \
       #        --backbone ACR2 \
       #        --ood_param 0.8 \
       #        --pretrain degenerate \
       #        --gpu_idx ${GPU}
       # echo "DONE TRAIN DIR 0.8 DEG ${DATASET}"

       # goodtg --config_path final_configs/${DATASET}/DIR.yaml \
       #        --seeds ${SEEDS} \
       #        --task train \
       #        --backbone ACR2 \
       #        --ood_param 0.1 \
       #        --gpu_idx ${GPU}
       # echo "DONE TRAIN DIR 0.1 ${DATASET}"

              
       # goodtg --config_path final_configs/${DATASET}/DIR.yaml \
       #        --seeds ${SEEDS} \
       #        --task train \
       #        --backbone ACR2 \
       #        --ood_param 0.1 \
       #        --pretrain degenerate \
       #        --gpu_idx ${GPU}
       # echo "DONE TRAIN DIR 0.1 DEG ${DATASET}"
# done


DATASET=SST2Planted/basis/no_shift
for ood_param in 1 10 100; do
       goodtg --config_path final_configs/${DATASET}/SMGNN.yaml \
              --seeds ${SEEDS} \
              --task train \
              --ood_param ${ood_param}  \
              --gpu_idx ${GPU}              
              # --l_norm_coeff: ${l_norm_coeff} \
              # --entr_coeff: ${entr_coeff} \
       echo "DONE TRAIN SMGNN ${ood_param}"
done




# for DATASET in GOODSST2/length/covariate; do
#        goodtg --config_path final_configs/${DATASET}/SMGNN.yaml \
#               --seeds ${SEEDS} \
#               --task train \
#               --average_edge_attn mean \
#               --gpu_idx ${GPU} \
#               --use_norm none \
#               --global_side_channel simple_product \
#               --mitigation_sampling raw              
#        echo "DONE TRAIN ${MODEL} ${DATASET} product"

#        goodtg --config_path final_configs/${DATASET}/SMGNN.yaml \
#               --seeds ${SEEDS} \
#               --task train \
#               --average_edge_attn mean \
#               --gpu_idx ${GPU} \
#               --use_norm none \
#               --global_side_channel simple_productscaled \
#               --mitigation_sampling raw              
#        echo "DONE TRAIN ${MODEL} ${DATASET} productscaled"

#        goodtg --config_path final_configs/${DATASET}/SMGNN.yaml \
#               --seeds ${SEEDS} \
#               --task train \
#               --average_edge_attn mean \
#               --gpu_idx ${GPU} \
#               --use_norm none \
#               --global_side_channel simple_godel \
#               --mitigation_sampling raw              
#        echo "DONE TRAIN ${MODEL} ${DATASET} godel"

#        goodtg --config_path final_configs/${DATASET}/SMGNN.yaml \
#               --seeds ${SEEDS} \
#               --task train \
#               --average_edge_attn mean \
#               --gpu_idx ${GPU} \
#               --use_norm none \
#               --global_side_channel simple_concept2 \
#               --mitigation_sampling raw              
#        echo "DONE TRAIN ${MODEL} ${DATASET} concept2"
# done

echo "DONE all :)"
