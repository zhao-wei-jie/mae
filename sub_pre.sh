python submitit_pretrain.py \
    --nodes 1 \
    --ngpus 1\
    --batch_size 16 \
    --model mae_vit_large_patch16 \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --seg_num 1 \
    --resume output_dir/20220418_1019/model-799.pth \
    --comment TTPLA自监督预训练+224+加权loss
    # --norm_pix_loss \
    # --data_path ${IMAGENET_DIR}