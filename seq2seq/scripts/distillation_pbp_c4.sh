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

file_name=distillation

pbp_reduction_factor=8

enc_num_tokens=10
dec_num_tokens=10
prompts_expand_after=True
lr=3e-4
side_pretrained_weight=t5-small
sample_type=interleaving
larger_learning_rate_mult=10


log_file_name=${file_name}_t5-base_t5-small
output_dir=${home_path}/data/outputs/${file_name}_2

for seed in 0
do

for task in "c4"

do

    rm -rf ${output_dir}
    
    CUDA_VISIBLE_DEVICES=$1 python run_distillation.py \
        --training_config_file configs/${file_name}.json \
        --task_name $task --eval_dataset_name $task --test_dataset_name $task \
        --seed ${seed}  \
        --num_train_epochs 3 \
        --preprocessing_num_workers 16 \
        --model_name_or_path t5-base \
        --tokenizer_name t5-base \
        --learning_rate ${lr} \
        --per_device_train_batch_size 1 \
        --output_dir ${output_dir} \
        --petl_method "pbp" \
        --pbp_reduction_factor ${pbp_reduction_factor} \
        --enc_num_tokens ${enc_num_tokens} \
        --dec_num_tokens ${dec_num_tokens} \
        --prompts_expand_after ${prompts_expand_after} \
        --learning_rate ${lr} \
        --output_dir ${output_dir} \
        --side_pretrained_weight $side_pretrained_weight \
        --sample_type $sample_type \
        --parameters_with_larger_lr ".*prompts.*|.*embed_to_kv.*|.*upsample.*|.*downsample.*" \
        --trainable_param_names ".*prompts.*|.*embed_to_kv.*|.*upsample.*|.*side.*|.*downsample.*" \
        --larger_learning_rate_mult $larger_learning_rate_mult

    cp ${output_dir}/all_results.json  all_output_logs/${log_file_name}_$task@${seed}.json

done
done
