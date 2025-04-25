#!/bin/bash

python TinyZeroTry2.py -m tinyzero -d gsm8k -p 1000 -o grm_trained_tinyzero
python TinyZeroTry2.py -m tinyzero -d prm800k -p 1000 -o prm_trained_tinyzero
python TinyZeroTry2.py -m grm_trained_tinyzero -d prm800k -p 1000 -o curriculum_trained_tinyzero
