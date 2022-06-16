CUDA_VISIBLE_DEVICES=0 python main_pretrain.py \
    --batch_size 32 \
    --accum_iter 1 \
    --model mae_vit_large_patch16 \
    --mask_ratio 0.75 \
    --epochs 1600 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --num_workers 12 \
    --resume mae_visualize_vit_large.pth
    #  --norm_pix_loss \