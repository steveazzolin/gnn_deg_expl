set -e

echo "I'm computing faithfulness :)"
echo "The PID of this script is: $$"

SPLITS="id_val"
SEEDS="1/2/3/4/5"
PRETRAIN="degenerate"
for DATASET in MNIST/basis/no_shift; do
       for METRIC in fidm fidp rfidm rfidp suff nec counter_fid suff_cause; do

              goodtg --config_path final_configs/${DATASET}/SMGNN.yaml \
                     --seeds ${SEEDS} \
                     --task eval_metric \
                     --metrics "${METRIC}" \
                     --splits ${SPLITS} \
                     --pretrain ${PRETRAIN} \
                     --model_layer 2 \
                     --gnn_clf_layer 2
              echo "DONE SMGNN ${METRIC} ${PRETRAIN}"
              
       done
done

echo "DONE all :)"
