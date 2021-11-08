#!/usr/bin/python3
import argparse
import os
import csv, subprocess


def main():
    parser = argparse.ArgumentParser(
        description="Automatically submit jobs using a csv file")

    # parser.add_argument('jobscript',help="job script to use")
    parser.add_argument('parameters',help="csv parameter file to use")
    #parser.add_argument('account',help="project account to charge jobs to")
    parser.add_argument('-t','--test',action='store_false',help="test script without submitting jobs")
    args = parser.parse_args()

    with open(args.parameters,mode='r',newline='',encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile)
        created_dirs = []
        for job in reader:
            final_command = "python3 -u DRP/src/drp_full_model.py --machine mist --train_file {0} --optimize 1 --max_num_epochs 30 --init_cpus 32 --init_gpus 1 --gpus_per_trial {1} --num_samples {2} --n_folds {3}  --data_types {4} --name_tag {5} --cv_subset_type {6} --stratify {7} --bottleneck {8} --full {9} --encoder_train {10} --pretrain {11} --merge_method {12} --global_code_size 512 --loss_type {13} --one_hot_drugs {14} --resume 1".format(*job)
            print(final_command)

    print("Done submitting jobs")


if __name__ == "__main__":
    main()
