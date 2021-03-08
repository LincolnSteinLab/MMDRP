from ray.tune import Analysis, ExperimentAnalysis

# PATH = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/sync_to_mac/"
PATH = "/scratch/l/lstein/ftaj/"

drug_analysis = Analysis(experiment_dir=PATH + "HyperOpt_Test_Morgan",
                        default_metric="sum_valid_loss", default_mode="min")
drug_analysis.get_best_config()
# drug_analysis = Analysis(experiment_dir=PATH + "Test_Morgan",
#                         default_metric="sum_valid_loss", default_mode="min")
# drug_analysis.get_best_config()

mut_analysis = Analysis(experiment_dir=PATH + "HyperOpt_Test_mut",
                        default_metric="sum_valid_loss", default_mode="min")
mut_analysis.get_best_config()
mut_analysis = ExperimentAnalysis(experiment_checkpoint_path=PATH+"HyperOpt_Test_mut/experiment_state-2021-01-21_20-39-50.json",)

# experiment_state-2021-01-21_20-39-50.json

cnv_analysis = Analysis(experiment_dir=PATH + "HyperOpt_Test_cnv",
                        default_metric="sum_valid_loss", default_mode="min")
cnv_analysis.get_best_config()

exp_analysis = Analysis(experiment_dir=PATH + "HyperOpt_Test_exp",
                        default_metric="sum_valid_loss", default_mode="min")
exp_analysis.get_best_config()

prot_analysis = Analysis(experiment_dir=PATH + "HyperOpt_Test_prot",
                        default_metric="sum_valid_loss", default_mode="min")
prot_analysis.get_best_config()
