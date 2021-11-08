#!/usr/bin/python3
## file: submit_jobs.py
import argparse
import os
import csv, subprocess

def create_dirs(n_dir):
    if not os.path.isdir(n_dir + "job_logs"):
        os.mkdir(n_dir + "job_logs/")
    if not os.path.isdir(n_dir + "results"):
        os.mkdir(n_dir + "results/")
            

def main():
    parser = argparse.ArgumentParser(
        description="Automatically submit jobs using a csv file")

    parser.add_argument('jobscript',help="job script to use")
    parser.add_argument('parameters',help="csv parameter file to use")
    #parser.add_argument('account',help="project account to charge jobs to")
    parser.add_argument('-t','--test',action='store_false',help="test script without submitting jobs")
    args = parser.parse_args()

    with open(args.parameters,mode='r',newline='',encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile)
        created_dirs = []
        for job in reader:
            # "TRAIN_FILE" "GPU_PER_TRIAL" "NUM_SAMPLES"   "N_FOLDS""DATA_TYPES""NAME_TAG""SUBSET_TYPE""STRATIFY""BOTTLENECK""FULL""ENCODER_TRAIN"
            # n_dir = os.getenv("HOME") + "/{9}/".format(*job)
            # if dir not in created_dirs:
            #     create_dirs(n_dir)
            #     created_dirs.append(n_dir)
            
            submit_command = (
                "sbatch " +
                "-o /scratch/l/lstein/ftaj/grid_job_logs/{5}.job_log ".format(*job) +
                "--job-name={5} "
                "--export=TRAIN_FILE={0},GPU_PER_TRIAL={1},NUM_SAMPLES={2},N_FOLDS={3},DATA_TYPES='{4}',NAME_TAG={5},SUBSET_TYPE={6},STRATIFY={7},BOTTLENECK={8},FULL={9},PRETRAIN={10},MERGE_METHOD={11},LOSS_TYPE={12}, ".format(*job) + args.jobscript)
                

            if not args.test:
                print(submit_command)
                template_command = "python3 -u DRP/src/drp_full_model.py --machine mist --train_file {0} --optimize 1 --max_num_epochs 100 --init_cpus 128 --init_gpus 4 --gpus_per_trial {1} --num_samples {2} --n_folds {3} --data_types {4} --name_tag {5} --cv_subset_type {6} --stratify {7} --bottleneck {8} --full {9} --pretrain {10} --merge_method {11} --loss_type {12}  --omic_standardize".format(*job)
                print(template_command)

            else:
                exit_status = subprocess.call(submit_command,shell=True)
                # Check to make sure the job submitted
                if exit_status is 1:
                    print("Job {5} failed to submit".format(*job))

    print("Done submitting jobs")

if __name__ == "__main__":
    main()
