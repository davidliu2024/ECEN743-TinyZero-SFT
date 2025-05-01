#!/bin/bash
# set -e

problem_list=(100 250 500 750 1000)
# problem_list=(50)

python TinyZeroTry2.py -m tinyzero-1.5 -d gsm8k -p 50 -o gsm_trained_tinyzero-1.5_50_problems
python TinyZeroTry2.py -m gsm_trained_tinyzero-1.5_50_problems -d prm800k -p 50 -o curriculum_trained_tinyzero-1.5_50_problems

for i in ${problem_list[@]}; do
    echo Executing at $i problems:
    # python TinyZeroTry2.py -m tinyzero-1.5 -d gsm8k -p $i -o gsm_trained_tinyzero-1.5
    python TinyZeroTry2.py -m tinyzero-1.5 -d prm800k -p $i -o prm_trained_tinyzero-1.5_${i}_problems
    python TinyZeroTry2.py -m gsm_trained_tinyzero-1.5_${i}_problems -d prm800k -p $i -o curriculum_trained_tinyzero-1.5_${i}_problems

    # mv gsm_trained_tinyzero-1.5 gsm_trained_tinyzero-1.5_${i}_problems
    # mv prm_trained_tinyzero-1.5 prm_trained_tinyzero-1.5_${i}_problems
    # mv curriculum_trained_tinyzero-1.5 curriculum_trained_tinyzero-1.5_${i}_problems
done


# for i in ${problem_list[@]}; do
#     python TinyZeroTry2.py -m tinyzero -d gsm8k -p $i -o gsm_trained_tinyzero
#     python TinyZeroTry2.py -m tinyzero -d prm800k -p $i -o prm_trained_tinyzero
#     python TinyZeroTry2.py -m gsm_trained_tinyzero -d prm800k -p $i -o curriculum_trained_tinyzero

#     mv gsm_trained_tinyzero gsm_trained_tinyzero_${i}_problems
#     mv prm_trained_tinyzero prm_trained_tinyzero_${i}_problems
#     mv curriculum_trained_tinyzero curriculum_trained_tinyzero_${i}_problems
# done