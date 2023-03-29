# This scripts trains Adapters method.
# For smaller datasets of GLUE (mrpc, cola, and stsb), we set the `num_train_epochs` to 20,
# for other larger datasets in GLUE we used `num_train_epochs` of 3. For all datasets we tried
# with the adapter's bottleneck size of `task_reduction_factor`=[32, 16, 8], and report the 
# results on the test set for the model performing the best on the validation set.

folder_name=all_output_logs/
if [ ! -d ${folder_name} ] ; then
    mkdir -p ${folder_name}
fi


source scripts/env.sh

file_name=weight_init

lr=3e-4
side_pretrained_weight=6-768
model_name_or_path=t5-base
distillation_init="learnable"
distilled_block_ids="[0,1,2,3,4,[5,6,7,8,9,10,11]]"
learnable_weight_type="softscalar-unshared"

log_file_name=${model_name_or_path}_full_finetuning_${side_pretrained_weight}_${file_name}_lr${lr}_${distillation_init}_${learnable_weight_type}_joint_merge_last
output_dir=${home_path}/data/outputs/${file_name}_6

for seed in 0
do

for task in "cola" "mrpc" "rte" "stsb" "qnli" "sst2" "mnli" "qqp"

do
    model_name_or_path=outputs/full_finetuning_${task}@0
    rm -rf ${output_dir}
    
    CUDA_VISIBLE_DEVICES=$1 python run_seq2seq.py \
        --training_config_file configs/${file_name}.json \
        --task_name $task --eval_dataset_name $task --test_dataset_name $task \
        --seed ${seed}  \
        --num_train_epochs ${num_epochs[$task]} \
        --learning_rate ${lr} \
        --output_dir ${output_dir} \
        --side_pretrained_weight ${side_pretrained_weight} \
        --model_name_or_path ${model_name_or_path} \
        --tokenizer_name ${model_name_or_path} \
        --distilled_block_ids ${distilled_block_ids} \
        --distillation_init ${distillation_init} \
        --learnable_weight_type ${learnable_weight_type} \
        --trainable_param_names ".*" \
        --parameters_with_larger_lr ".*merge_weights.*" \
        --larger_learning_rate_mult 10

    cp ${output_dir}/all_results.json  all_output_logs/${log_file_name}_$task@${seed}.json

done
done
