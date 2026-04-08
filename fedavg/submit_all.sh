#!/bin/bash
# submit_all.sh
# 用法：bash submit_all.sh

# 实验组 A：IID vs Non-IID
sbatch --export=ALL,PARTITION=iid,ALPHA=0.5,RUN_TAG=50-iid          experiment.sh
sbatch --export=ALL,PARTITION=noniid,ALPHA=0.9,RUN_TAG=50-noniid-0.9 experiment.sh
sbatch --export=ALL,PARTITION=noniid,ALPHA=0.7,RUN_TAG=50-noniid-0.7 experiment.sh
sbatch --export=ALL,PARTITION=noniid,ALPHA=0.5,RUN_TAG=50-noniid-0.5 experiment.sh
sbatch --export=ALL,PARTITION=noniid,ALPHA=0.3,RUN_TAG=50-noniid-0.3 experiment.sh
sbatch --export=ALL,PARTITION=noniid,ALPHA=0.1,RUN_TAG=50-noniid-0.1 experiment.sh

# 实验组 B：local_epochs
sbatch --export=ALL,EPOCHS=1,N_ROUNDS=250,RUN_TAG=50-epochs-1   experiment.sh
sbatch --export=ALL,EPOCHS=5,N_ROUNDS=50,RUN_TAG=50-epochs-5   experiment.sh
sbatch --export=ALL,EPOCHS=10,N_ROUNDS=25,RUN_TAG=50-epochs-10 experiment.sh
sbatch --export=ALL,EPOCHS=25,N_ROUNDS=10,RUN_TAG=50-epochs-25 experiment.sh

# 实验组 C：client_fraction
sbatch --export=ALL,FRACTION=0.1,RUN_TAG=50-frac-0.1 experiment.sh
sbatch --export=ALL,FRACTION=0.3,RUN_TAG=50-frac-0.3 experiment.sh
sbatch --export=ALL,FRACTION=0.5,RUN_TAG=50-frac-0.5 experiment.sh
sbatch --export=ALL,FRACTION=0.7,RUN_TAG=50-frac-0.7 experiment.sh
sbatch --export=ALL,FRACTION=1.0,RUN_TAG=50-frac-1.0 experiment.sh
