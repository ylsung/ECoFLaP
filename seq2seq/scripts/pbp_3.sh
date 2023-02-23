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

file_name=pbp

pbp_reduction_factor=1

enc_num_tokens=100
dec_num_tokens=0
prompts_expand_after=True

log_file_name=${file_name}_r${pbp_reduction_factor}_e${enc_num_tokens}_d${dec_num_tokens}
output_dir=${home_path}/data/outputs/${file_name}_3

for seed in 0
do

for task in "cola" "mrpc" "qnli" "sst2" "rte" "mnli" "qqp" "stsb"

do

    rm -rf ${output_dir}
    
    CUDA_VISIBLE_DEVICES=$1 python run_seq2seq.py \
        --training_config_file configs/${file_name}.json \
        --task_name $task --eval_dataset_name $task --test_dataset_name $task \
        --seed ${seed}  \
        --num_train_epochs ${num_epochs[$task]} \
        --pbp_reduction_factor ${pbp_reduction_factor} \
        --enc_num_tokens ${enc_num_tokens} \
        --dec_num_tokens ${dec_num_tokens} \
        --prompts_expand_after ${prompts_expand_after} \
        --output_dir ${output_dir}

    cp ${output_dir}/all_results.json  all_output_logs/${log_file_name}_$task@${seed}.json

done
done
