sparsity_ratio=0.6
max_sparsity_per_layer=0.7

device=$1

CUDA_VISIBLE_DEVICES=${device} python main.py \
    --model huggyllama/llama-7b \
    --prune_method wanda \
    --sparsity_ratio ${sparsity_ratio} \
    --sparsity_type unstructured \
    --save out/llama_7b/unstructured/ecoflap_zero${sparsity_ratio}/ \
    --approach_for_sparsity block \
    --aggregate_method sum \
    --score_method GradOnly \
    --max_sparsity_per_layer ${max_sparsity_per_layer} \
    --use_mezo \
    --num_samples_for_first_stage 32 