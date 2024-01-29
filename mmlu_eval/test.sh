
pruned_checkpoint=your_checkpoint
folder=results/${pruned_checkpoint}
output_file=outputs/${pruned_checkpoint}.txt

python evaluate_flan_new.py -k 5 -g 1 -d mmlu -s ${folder} -m google/flan-t5-xl \
    --pruned_checkpoint ../LAVIS/pruned_checkpoint/${pruned_checkpoint} \
    -o ${output_file}


