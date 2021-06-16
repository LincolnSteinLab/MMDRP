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
                "--export=TRAIN_FILE={0},GPU_PER_TRIAL={1},NUM_SAMPLES={2},N_FOLDS={3},DATA_TYPES='{4}',NAME_TAG={5},SUBSET_TYPE={6},STRATIFY={7},BOTTLENECK={8},FULL={9},ENCODER_TRAIN={10},PRETRAIN={11} ".format(*job) + args.jobscript)
                

            if not args.test:
                print(submit_command)
            else:
                exit_status = subprocess.call(submit_command,shell=True)
                # Check to make sure the job submitted
                if exit_status is 1:
                    print("Job {5} failed to submit".format(submit_command))

    print("Done submitting jobs")

if __name__ == "__main__":
    main()
