# Applying Supervised Fine-Tuning on TinyZero for Reasoning Tasks Using Curriculum Learning Methods

## Abstract

Current LLMs are curated on various types of datasets for specific problems. For example, TinyZero is trained on various types of multiplication, division, and countdown problems to improve reasoning performance, and Verigen is trained on Verilog HDL to generate Verilog code better. However, in psychology, children are found to learn through various levels of the curriculum, where the child learns the basic tasks first before moving onto more difficult tasks (i.e., learning algebra before calculus). In this project, we aim to use a curriculum learning approach, and apply supervised fine-tuning with increasingly challenging problem datasets to TinyZero, a reproduction of Deepseek Zero that uses reinforcement learning for self-verification and more accurate searching abilities and evaluate its performance first with curriculum learning, and then without.

## About

### Authors
 - David Liu (david_liu@tamu.edu)
 - Amarachukwu Nzedibe (amaranzedibe1@tamu.edu)
 - Nicole LoGiudice (nicolelogiudice30@tamu.edu)

### Organization
 - Texas A&M University - College Station

### Purpose

## Project Structure

Main script:
```
TinyZeroTry2.py
```
Executes training and collects loss.

Results are under:
```
./results
```
Includes testing loss datapoints and plots.

## Requirements
### OS and Software requirements
 - Ubuntu > 20.0 (Or any Debian environment)
 - Python > 3.9
 - MiniConda > 25.0
### Hardware requirements
 - CPU memory > 8GB
 - GPU memory > 32GB

## Initialization and Set Up

To set up the MiniConda Environment:
```
conda create -n tinyzero-env python=3.9
conda activate tinyzero-env
```

To install all packages:
```
pip install -r freeze.txt
```

## Execution

To run a single round of training:
```
python TinyZeroTry2.py -m <model> -d <dataset> -p <problems> -o <output directory>
```
Avaliable models include:
 - `tinyzero`
 - `tinyzero-1.5`
 - Any local model saved to the machine
Available datasets include:
 - `gsm8k`
 - `prm800k`


To run and collect results at 50,100,250,500,750,1000 problem sets.
```
chmod +x ./curriculum_learning.sh
./curriculum_learning.sh
```


