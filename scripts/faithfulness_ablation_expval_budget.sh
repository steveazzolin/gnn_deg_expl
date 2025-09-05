set -e

echo "I'm computing faithfulness :)"
echo "The PID of this script is: $$"

SPLITS="id_val"
SEEDS="1/2/3/4/5"
PRETRAIN="degenerate"
for DATASET in BAColorGVIsol/basis/no_shift; do
       for B in 10 20 50 80 100 150 200 300 500 1000; do

              goodtg --config_path final_configs/${DATASET}/GSAT.yaml \
                     --seeds ${SEEDS} \
                     --task eval_metric \
                     --metrics "rfidm/rfidp/suff_cause/suff/nec/counter_fid" \
                     --splits ${SPLITS} \
                     --pretrain ${PRETRAIN} \
                     --backbone ACR2 \
                     --expval_budget ${B}
              echo "DONE GSAT ${METRIC} ${PRETRAIN} ACR2 ablation expval_budget ${B}"
              
       done
done

echo "DONE all :)"
