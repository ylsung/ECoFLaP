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

file_name=prefix_tuning

enc_num_tokens=10
dec_num_tokens=10
prompts_expand_after=True
lr=1e-2
prefix_dim=64

log_file_name=${file_name}_e${enc_num_tokens}_d${dec_num_tokens}_dim${prefix_dim}_lr${lr}
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
        --enc_num_tokens ${enc_num_tokens} \
        --dec_num_tokens ${dec_num_tokens} \
        --prompts_expand_after ${prompts_expand_after} \
        --learning_rate ${lr} \
        --prefix_dim ${prefix_dim} \
        --output_dir ${output_dir}

    cp ${output_dir}/all_results.json  all_output_logs/${log_file_name}_$task@${seed}.json

done
done
