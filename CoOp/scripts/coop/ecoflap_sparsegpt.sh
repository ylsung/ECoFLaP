#!/bin/bash

# custom config
DATA=/nas/longleaf/home/ylsung/work/clip_data
TRAINER=ZeroshotCLIP
CFG=vit_b16 # rn50, rn101, vit_b32 or vit_b16


seed=1
remaining_sparsity=0.6
max_sparsity_per_layer=0.5

method="sparsegpt"

for DATASET in caltech101 dtd eurosat fgvc_aircraft food101 imagenet oxford_flowers oxford_pets stanford_cars sun397 ucf101
do
    output_dir=output/${TRAINER}/${method}${remaining_sparsity}_${CFG}_s${seed}/${DATASET}
    rm -r ${output_dir}
    python train.py \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/CoOp/${CFG}.yaml \
    --output-dir ${output_dir} \
    --eval-only \
    --pruning_method ${method} \
    --visual_prune_spec 1-${remaining_sparsity}-1-1 \
    --language_prune_spec 1-${remaining_sparsity}-1-1 \
    --seed ${seed}
done

for DATASET in caltech101 dtd eurosat fgvc_aircraft food101 imagenet oxford_flowers oxford_pets stanford_cars sun397 ucf101
do
    output_dir=output/${TRAINER}/${method}_mezo${remaining_sparsity}_${CFG}_s${seed}/${DATASET}
    rm -r ${output_dir}
    python train.py \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/CoOp/${CFG}.yaml \
    --output-dir ${output_dir} \
    --eval-only \
    --pruning_method ${method} \
    --visual_prune_spec 1-${remaining_sparsity}-1-1 \
    --language_prune_spec 1-${remaining_sparsity}-1-1 \
    --sparsity_ratio_granularity block \
    --max_sparsity_per_layer ${max_sparsity_per_layer} \
    --score_method MEZO-GradOnly_sum \
    --seed ${seed}
done
