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
side_pretrained_weight=10-768
model_name_or_path=t5-base
distillation_init="sum"
distilled_block_ids="[0,1,2,3,4,5,6,7,8,[9,10,11]]"
distilled_block_weights_template="1,1,1,1,1,1,1,1,1"

output_dir=${home_path}/data/outputs/${file_name}_5

for seed in 0
do

for weights in "[0.3,0.1,0.6]" "[0.6,0.1,0.3]"
do
    distilled_block_weights=[${distilled_block_weights_template},${weights}]
    log_file_name=${model_name_or_path}_${side_pretrained_weight}_${file_name}_lr${lr}_${distillation_init}_merge${weights}_last

for task in "cola" "mrpc" "qnli" "sst2" "rte" "mnli" "qqp" "stsb"

do

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
        --distilled_block_weights ${distilled_block_weights}

    cp ${output_dir}/all_results.json  all_output_logs/${log_file_name}_$task@${seed}.json

done
done
done
