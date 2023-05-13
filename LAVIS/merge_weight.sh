path="lavis/output/BLIP2/Pretrain_stage2/pretrain2_only-vit+t5-obs_prune_1.0-1.0-0.5"
python merge_weight.py \
    ${path}/checkpoint_0.pth \
    /home/yilin/.cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth \
    ${path}/checkpoint_merge.pth


path="lavis/output/BLIP2/Pretrain_stage2/pretrain2_only-vit+t5-unstrct_mag_prune_1.0-1.0-0.5"
python merge_weight.py \
    ${path}/checkpoint_0.pth \
    /home/yilin/.cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth \
    ${path}/checkpoint_merge.pth

path="lavis/output/BLIP2/Pretrain_stage2/pretrain2_only-t5-obs_prune_1.0-1.0-0.5"
python merge_weight.py \
    ${path}/checkpoint_0.pth \
    /home/yilin/.cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth \
    ${path}/checkpoint_merge.pth

path="lavis/output/BLIP2/Pretrain_stage2/pretrain2_only-t5-unstrct_mag_prune_1.0-1.0-0.5"
python merge_weight.py \
    ${path}/checkpoint_0.pth \
    /home/yilin/.cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth \
    ${path}/checkpoint_merge.pth