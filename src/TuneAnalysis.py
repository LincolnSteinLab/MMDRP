from ray.tune import Analysis, ExperimentAnalysis

# PATH = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/sync_to_mac/"
PATH = "/scratch/l/lstein/ftaj/"

# drug_analysis = Analysis(experiment_dir=PATH + "HyperOpt_DRP_FullModel_drug_prot_CTRP_Full/",
#                             default_metric="avg_cv_valid_loss", default_mode="min")
drug_analysis = Analysis(experiment_dir=PATH + "HyperOpt_DRP_ResponseOnly_drug_prot_CTRP_EncoderTrain/",
                            default_metric="avg_cv_valid_loss", default_mode="min")
drug_analysis.get_best_config()
drug_analysis.get_best_checkpoint()
analysis = Analysis(experiment_dir=PATH + "HyperOpt_CV_pretrain_cnv/", default_metric="avg_cv_valid_loss", default_mode="min")
analysis = Analysis(experiment_dir=PATH + "HyperOpt_CV_exp/")
best_checkpoint = analysis.get_best_checkpoint(analysis.get_best_logdir(),)

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
