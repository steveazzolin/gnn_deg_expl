r"""Kernel pipeline: main pipeline, initialization, task loading, etc.
"""
import json
from collections import defaultdict
from datetime import datetime

from torch_geometric import __version__ as __pyg_version__

from GOOD import config_summoner
from GOOD.kernel.pipeline_manager import load_pipeline
from GOOD.ood_algorithms.ood_manager import load_ood_alg
from GOOD.utils.args import args_parser
from GOOD.utils.logger import load_logger
from GOOD.utils.metric import assign_dict
from GOOD.definitions import OOM_CODE
import GOOD.utils.analysis as analysis
from GOOD.utils.loader import initialize_model_dataset


import warnings
import numpy as np
import matplotlib.pyplot as plt
import wandb
from scipy.stats import pearsonr

# if __pyg_version__ == "2.4.0":
#     torch.set_num_threads(6)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    args = args_parser()

    assert not args.seeds is None, args.seeds

    if args.task == 'eval_metric':
        analysis.evaluate_metric(args)
        exit(0)
    if args.task == 'plot_panel':
        analysis.generate_panel(args)
        exit(0)
    if args.task == 'permute_attention':
        analysis.permute_attention_scores(args)
        exit(0)
    if args.task == 'plot_sampling':
        analysis.generate_plot_sampling(args)
        exit(0)
    if args.task == 'stability_detector':
        # stability_detector_rebuttal(args)
        analysis.stability_detector_extended(args)
        exit(0)
    if args.task == 'plot_explanations':
        analysis.plot_explanation_examples(args)
        exit(0)
    if args.task == 'print_faith':
        analysis.print_faith(args)
        exit(0)
    if args.task == 'print_r_ge_b_hist':
        analysis.print_r_ge_b_hist(args)
        exit(0)
        

    run = None
    test_scores, test_losses = defaultdict(list), defaultdict(list)
    test_likelihoods_avg, test_likelihoods_prod, test_likelihoods_logprod, test_wious = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    channel_relevances, global_coeffs, global_weights = [], [], []
    for i, seed in enumerate(args.seeds.split("/")):
        seed = int(seed)
        print(f"\n\n#D#Running with seed = {seed}")
        
        args.random_seed = seed
        args.exp_round = seed
        
        config = config_summoner(args)
        config["mitigation_backbone"] = args.mitigation_backbone
        print(config.random_seed, config.exp_round)
        print(args)
        if i == 0:
            load_logger(config)
        
        model, loader = initialize_model_dataset(config)
        ood_algorithm = load_ood_alg(config.ood.ood_alg, config)

        pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)

        if config.task == 'train':
            startTrainTime = datetime.now()
            if config.wandb:
                run = wandb.init(
                    project="explanation-shortcuts",
                    config=config,
                    entity="mcstewe",
                    name=f'{config.dataset.dataset_name}_{config.dataset.domain}{config.ood_dirname}_{config.util_model_dirname}_{config.random_seed}'
                )
                wandb.watch(pipeline.model, log="all", log_freq=10)

            # Train model
            pipeline.load_task()            
            print(f'\nTraining end ({datetime.now() - startTrainTime}).\n')

            # Eval model
            pipeline.task = 'test'
            test_score, test_loss = pipeline.load_task(load_param=True, load_split="id")
            test_scores["saved_score"].append(test_score)
            for s in ["id_val", "id_test", "val", "test"]:
                sa = pipeline.evaluate(s, compute_suff=False)
                test_scores[s].append(sa['score'])
            
            if config.global_side_channel and "simple_concept" in config.global_side_channel:
                channel_relevances.append(model.combinator.classifier[0].alpha_norm.cpu().numpy())
                print("\nConcept relevance scores for this run:\n", channel_relevances[-1], "\n")
        elif config.task == 'test':
            test_score, test_loss = pipeline.load_task(load_param=True, load_split="id")

            for s in ["train", "id_val", "id_test", "val", "test"]:
                sa = pipeline.evaluate(
                    s,
                    compute_suff=False, 
                    compute_wiou=(config.dataset.dataset_name == "TopoFeature" or config.dataset.dataset_name == "SimpleMotif" or config.dataset.dataset_name == "GOODMotif")
                                    and 
                                 config.model.model_name != "GIN"
                )
                test_scores[s].append(sa['score'])
                test_losses[s].append(sa['loss'].item())
                test_wious[s].append(sa['wiou'])
                test_likelihoods_avg[s].append(sa['likelihood_avg'].item())
                test_likelihoods_prod[s].append(sa['likelihood_prod'].item())
                test_likelihoods_logprod[s].append(sa['likelihood_logprod'].item())
            
            if "GiSST" in config.model.model_name and config.dataset.dataset_name in ("BAColor", "TopoFeature", "AIDS", "AIDSC1"):
                print("\nFeature explanation coeff. for this run:\n", model.prob_mask())
            
                
    
    if config.save_metrics:
        with open(f"storage/metric_results/acc_plaus.json", "r") as jsonFile:
            results_aggregated = json.load(jsonFile)
    
    print("\n\nFinal accuracies: ")
    for s in test_scores.keys():
        print(f"{s.upper():<10} = {np.mean(test_scores[s]):.3f} +- {np.std(test_scores[s]):.3f}")

    if config.global_side_channel and config.model.model_name != "GIN":
        # threshold = 0.9
        threshold = 0.0
        id_val_accs = np.array(test_scores["id_val"])

        print(f"\n\nFinal accuracies (model with id_val acc above {threshold}% - {sum(id_val_accs >= threshold)} runs): ")
        for s in test_scores.keys():
            tmp = np.array(test_scores[s])[id_val_accs >= threshold]
            print(f"{s.upper():<10} = {np.mean(tmp):.3f} +- {np.std(tmp):.3f}")

        if "simple_concept" in config.global_side_channel or config.global_side_channel == "simple_linear":
            channel_relevances = np.concatenate(channel_relevances, axis=0)
            print(f"\n\nAveraged channel relevance scores (model with id_val acc above {threshold}% - {sum(id_val_accs >= threshold)} runs): ")
            print(f"{channel_relevances[id_val_accs >= threshold].mean(0)} +- {channel_relevances[id_val_accs >= threshold].std(0)}")

        print(f"\n\nFinal Test WIoUs (model with id_val acc above {threshold}% - {sum(id_val_accs >= threshold)} runs):")
        for s in test_wious.keys():
            tmp = np.array(test_wious[s])[id_val_accs >= threshold]
            print(f"{s.upper():<10} = {np.mean(tmp):.3f} +- {np.std(tmp):.3f}")

        if config.dataset.dataset_name in ("BAColor", "TopoFeature", "AIDS", "AIDSC1"):
            print(f"\n\nGlobal side channel coefficient wrt x1 (model with id_val acc above {threshold}% - {sum(id_val_accs >= threshold)} runs):")
            tmp = np.array(global_coeffs)[id_val_accs >= threshold]
            print(f"{tmp.mean(0)} +- {tmp.std(0)}")

            print(f"\n\nAverage global channel weights (model with id_val acc above {threshold}% - {sum(id_val_accs >= threshold)} runs):")
            tmp = np.array(global_weights)[id_val_accs >= threshold]
            print(f"{tmp.mean(0)} +- {tmp.std(0)}")

        # print(f"\n\nCorrelation local channel importance-OOD Test Acc")
        # print(pearsonr(test_scores["test"], channel_relevances[:, 0]))   



    print("\n\nFinal losses: ")
    for s in test_losses.keys():
        print(f"{s.upper():<10} = {np.mean(test_losses[s]):.4f} +- {np.std(test_losses[s]):.4f}")
            
    for s in [""]: #"ood_"
        print(f"Diff id_val-test {s} = {abs(np.mean(test_losses[s + 'id_val']) - np.mean(test_losses[s + 'test'])):.4f} ")

    if config.save_metrics:
        print("Saving metrics to json...")
        for s in test_losses.keys():
            for name, d in zip(
                ["loss_entiresplit", "likelihood_avg_entiresplit", "likelihood_prod_entiresplit", "likelihood_logprod_entiresplit"], 
                [test_losses, test_likelihoods_avg, test_likelihoods_prod, test_likelihoods_logprod]
            ):
                key = [config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, s, name]
                if s in results_aggregated[key[0]][key[1]].keys():            
                    assign_dict(
                        results_aggregated,
                        key,
                        np.mean(d[s])
                    )
                    key[-1] += "_std"
                    assign_dict(
                        results_aggregated,
                        key,
                        np.std(d[s])
                    )
        with open(f"storage/metric_results/acc_plaus.json", "w") as f:
            json.dump(results_aggregated, f)  

    if config.wandb and run:
        run.finish()

def goodtg():
    try:
        main()
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f'#E#{e}')
            exit(OOM_CODE)
        else:
            raise e


if __name__ == '__main__':
    main()
