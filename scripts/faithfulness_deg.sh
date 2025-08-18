set -e

echo "I'm computing faithfulness :)"
echo "The PID of this script is: $$"

SPLITS="id_val"
SEEDS="1/2/3/4/5"
PRETRAIN="degenerate"
for DATASET in MNIST/basis/no_shift ; do  # BAColorGVIsol/basis/no_shift MUTAG/basis/no_shift SST2Planted/basis/no_shift
       for MODEL in DIR; do # SMGNN GSAT DIR
              for METRIC in suff_cause fidm fidp suff nec counter_fid rfidm rfidp; do

                     echo "DOING ${DATASET} ${MODEL} ${METRIC}"       
                     
                     goodtg --config_path final_configs/${DATASET}/${MODEL}.yaml \
                            --seeds ${SEEDS} \
                            --task eval_metric \
                            --metrics "${METRIC}" \
                            --splits ${SPLITS} \
                            --pretrain ${PRETRAIN} \
                            --backbone ACR2 \
                            --ood_param 0.8 \
                            --use_readout_norm bn \
                            --gpu_idx 1 \
                            --expval_budget 50

                     echo "DONE ${DATASET} ${MODEL} ${METRIC}"

              done
       done
done

echo "DONE all :)"
