####### baseline #########
python train.py --seed 4023 --fold 0 --epoch 300 --batch_size 64 --gpus 4 --lr 1e-3 --logdir biased_conv_transformer_192_12_32_7_RMSNorm_ALiBi_swiGLU_ep300_lr1e_3_bs256 \
   --pseudo_label_df ../../logs/exp064_finetune/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256/fold0/test_pl_filterling_0.75_half.csv \
   --num_workers 4 --disable_compile
python train_finetune.py --seed 4023 --fold 0 --epoch 15 --batch_size 64 --gpus 4 --lr 5e-4 \
   --logdir biased_conv_transformer_192_12_32_7_RMSNorm_ALiBi_swiGLU_ep300_lr1e_3_bs256 --resumedir exp070/biased_conv_transformer_192_12_32_7_RMSNorm_ALiBi_swiGLU_ep300_lr1e_3_bs256 \
   --num_workers 4 \
   --pseudo_label_df ../../logs/exp064_finetune/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256/fold0/test_pl_filterling_0.75_half.csv 
python eval.py --fold 0 --batch_size 512 --logdir exp070_finetune/biased_conv_transformer_192_12_32_7_RMSNorm_ALiBi_swiGLU_ep300_lr1e_3_bs256 

python train.py --seed 4024 --fold 1 --epoch 300 --batch_size 64 --gpus 4 --lr 1e-3 --logdir biased_conv_transformer_192_12_32_7_RMSNorm_ALiBi_swiGLU_ep300_lr1e_3_bs256 \
   --pseudo_label_df ../../logs/exp064_finetune/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256/fold1/test_pl_filterling_0.75_half.csv \
   --num_workers 4 --disable_compile
python train_finetune.py --seed 4024 --fold 1 --epoch 15 --batch_size 64 --gpus 4 --lr 5e-4 \
   --logdir biased_conv_transformer_192_12_32_7_RMSNorm_ALiBi_swiGLU_ep300_lr1e_3_bs256 --resumedir exp070/biased_conv_transformer_192_12_32_7_RMSNorm_ALiBi_swiGLU_ep300_lr1e_3_bs256 \
   --num_workers 4 \
   --pseudo_label_df ../../logs/exp064_finetune/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256/fold1/test_pl_filterling_0.75_half.csv 
python eval.py --fold 1 --batch_size 512 --logdir exp070_finetune/biased_conv_transformer_192_12_32_7_RMSNorm_ALiBi_swiGLU_ep300_lr1e_3_bs256 

python train.py --seed 4025 --fold 2 --epoch 300 --batch_size 64 --gpus 4 --lr 1e-3 --logdir biased_conv_transformer_192_12_32_7_RMSNorm_ALiBi_swiGLU_ep300_lr1e_3_bs256 \
   --pseudo_label_df ../../logs/exp064_finetune/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256/fold2/test_pl_filterling_0.75_half.csv \
   --num_workers 4 --disable_compile
python train_finetune.py --seed 4025 --fold 2 --epoch 15 --batch_size 64 --gpus 4 --lr 5e-4 \
   --logdir biased_conv_transformer_192_12_32_7_RMSNorm_ALiBi_swiGLU_ep300_lr1e_3_bs256 --resumedir exp070/biased_conv_transformer_192_12_32_7_RMSNorm_ALiBi_swiGLU_ep300_lr1e_3_bs256 \
   --num_workers 4 \
   --pseudo_label_df ../../logs/exp064_finetune/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256/fold2/test_pl_filterling_0.75_half.csv 
python eval.py --fold 2 --batch_size 512 --logdir exp070_finetune/biased_conv_transformer_192_12_32_7_RMSNorm_ALiBi_swiGLU_ep300_lr1e_3_bs256 

python train.py --seed 4026 --fold 3 --epoch 300 --batch_size 64 --gpus 4 --lr 1e-3 --logdir biased_conv_transformer_192_12_32_7_RMSNorm_ALiBi_swiGLU_ep300_lr1e_3_bs256 \
   --pseudo_label_df ../../logs/exp064_finetune/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256/fold3/test_pl_filterling_0.75_half.csv \
   --num_workers 4 --disable_compile
python train_finetune.py --seed 4026 --fold 3 --epoch 15 --batch_size 64 --gpus 4 --lr 5e-4 \
   --logdir biased_conv_transformer_192_12_32_7_RMSNorm_ALiBi_swiGLU_ep300_lr1e_3_bs256 --resumedir exp070/biased_conv_transformer_192_12_32_7_RMSNorm_ALiBi_swiGLU_ep300_lr1e_3_bs256 \
   --num_workers 4 \
   --pseudo_label_df ../../logs/exp064_finetune/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256/fold3/test_pl_filterling_0.75_half.csv 
python eval.py --fold 3 --batch_size 512 --logdir exp070_finetune/biased_conv_transformer_192_12_32_7_RMSNorm_ALiBi_swiGLU_ep300_lr1e_3_bs256 

python train.py --seed 4027 --fold 4 --epoch 300 --batch_size 64 --gpus 4 --lr 1e-3 --logdir biased_conv_transformer_192_12_32_7_RMSNorm_ALiBi_swiGLU_ep300_lr1e_3_bs256 \
   --pseudo_label_df ../../logs/exp064_finetune/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256/fold4/test_pl_filterling_0.75_half.csv \
   --num_workers 4 --disable_compile
python train_finetune.py --seed 4027 --fold 4 --epoch 15 --batch_size 64 --gpus 4 --lr 5e-4 \
   --logdir biased_conv_transformer_192_12_32_7_RMSNorm_ALiBi_swiGLU_ep300_lr1e_3_bs256 --resumedir exp070/biased_conv_transformer_192_12_32_7_RMSNorm_ALiBi_swiGLU_ep300_lr1e_3_bs256 \
   --num_workers 4 \
   --pseudo_label_df ../../logs/exp064_finetune/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256/fold4/test_pl_filterling_0.75_half.csv 
python eval.py --fold 4 --batch_size 512 --logdir exp070_finetune/biased_conv_transformer_192_12_32_7_RMSNorm_ALiBi_swiGLU_ep300_lr1e_3_bs256 

python make_oof.py --batch_size 512 --logdir exp070_finetune/biased_conv_transformer_192_12_32_7_RMSNorm_ALiBi_swiGLU_ep300_lr1e_3_bs256 
##########################


######### tiny model #########
python train.py --seed 5023 --fold 0 --epoch 300 --batch_size 256 --gpus 1 --lr 2e-3 --dim 128 --head_size 16 --depth 8 \
   --logdir biased_conv_transformer_128_8_16_7_RMSNorm_ALiBi_swiGLU_ep300_lr2e_3_bs256 \
   --pseudo_label_df ../../logs/exp064_finetune/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256/fold4/test_pl_filterling_0.75_half.csv \
   --num_workers 24 --disable_compile
python train_finetune.py --seed 5023 --fold 0 --epoch 15 --batch_size 64 --gpus 4 --lr 1e-3 \
   --logdir biased_conv_transformer_128_8_16_7_RMSNorm_ALiBi_swiGLU_ep300_lr2e_3_bs256 --resumedir exp070/biased_conv_transformer_128_8_16_7_RMSNorm_ALiBi_swiGLU_ep300_lr2e_3_bs256 --num_workers 4
python eval.py --fold 0 --batch_size 512 --logdir exp070_finetune/biased_conv_transformer_128_8_16_7_RMSNorm_ALiBi_swiGLU_ep300_lr2e_3_bs256

python train.py --seed 5028 --fold 1 --epoch 300 --batch_size 256 --gpus 1 --lr 2e-3 --dim 128 --head_size 16 --depth 8 \
   --logdir biased_conv_transformer_128_8_16_7_RMSNorm_ALiBi_swiGLU_ep300_lr2e_3_bs256 \
   --pseudo_label_df ../../logs/exp064_finetune/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256/fold4/test_pl_filterling_0.75_half.csv \
   --num_workers 24 --disable_compile
python train_finetune.py --seed 5028 --fold 1 --epoch 15 --batch_size 64 --gpus 4 --lr 1e-3 \
   --logdir biased_conv_transformer_128_8_16_7_RMSNorm_ALiBi_swiGLU_ep300_lr2e_3_bs256 --resumedir exp070/biased_conv_transformer_128_8_16_7_RMSNorm_ALiBi_swiGLU_ep300_lr2e_3_bs256 --num_workers 4
python eval.py --fold 0 --batch_size 512 --logdir exp070_finetune/biased_conv_transformer_128_8_16_7_RMSNorm_ALiBi_swiGLU_ep300_lr2e_3_bs256

python train.py --seed 5025 --fold 2 --epoch 300 --batch_size 256 --gpus 1 --lr 2e-3 --dim 128 --head_size 16 --depth 8 \
   --logdir biased_conv_transformer_128_8_16_7_RMSNorm_ALiBi_swiGLU_ep300_lr2e_3_bs256 \
   --pseudo_label_df ../../logs/exp064_finetune/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256/fold4/test_pl_filterling_0.75_half.csv \
   --num_workers 24 --disable_compile
python train_finetune.py --seed 5025 --fold 2 --epoch 15 --batch_size 64 --gpus 4 --lr 1e-3 \
   --logdir biased_conv_transformer_128_8_16_7_RMSNorm_ALiBi_swiGLU_ep300_lr2e_3_bs256 --resumedir exp070/biased_conv_transformer_128_8_16_7_RMSNorm_ALiBi_swiGLU_ep300_lr2e_3_bs256 --num_workers 4
python eval.py --fold 0 --batch_size 512 --logdir exp070_finetune/biased_conv_transformer_128_8_16_7_RMSNorm_ALiBi_swiGLU_ep300_lr2e_3_bs256

python train.py --seed 5026 --fold 3 --epoch 300 --batch_size 256 --gpus 1 --lr 2e-3 --dim 128 --head_size 16 --depth 8 \
   --logdir biased_conv_transformer_128_8_16_7_RMSNorm_ALiBi_swiGLU_ep300_lr2e_3_bs256 \
   --pseudo_label_df ../../logs/exp064_finetune/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256/fold4/test_pl_filterling_0.75_half.csv \
   --num_workers 24 --disable_compile
python train_finetune.py --seed 5026--fold 3 --epoch 15 --batch_size 64 --gpus 4 --lr 1e-3 \
   --logdir biased_conv_transformer_128_8_16_7_RMSNorm_ALiBi_swiGLU_ep300_lr2e_3_bs256 --resumedir exp070/biased_conv_transformer_128_8_16_7_RMSNorm_ALiBi_swiGLU_ep300_lr2e_3_bs256 --num_workers 4
python eval.py --fold 0 --batch_size 512 --logdir exp070_finetune/biased_conv_transformer_128_8_16_7_RMSNorm_ALiBi_swiGLU_ep300_lr2e_3_bs256

python train.py --seed 5029 --fold 4 --epoch 300 --batch_size 256 --gpus 1 --lr 2e-3 --dim 128 --head_size 16 --depth 8 \
   --logdir biased_conv_transformer_128_8_16_7_RMSNorm_ALiBi_swiGLU_ep300_lr2e_3_bs256 \
   --pseudo_label_df ../../logs/exp064_finetune/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256/fold4/test_pl_filterling_0.75_half.csv \
   --num_workers 24 --disable_compile
python train_finetune.py --seed 5029 --fold 4 --epoch 15 --batch_size 64 --gpus 4 --lr 1e-3 \
   --logdir biased_conv_transformer_128_8_16_7_RMSNorm_ALiBi_swiGLU_ep300_lr2e_3_bs256 --resumedir exp070/biased_conv_transformer_128_8_16_7_RMSNorm_ALiBi_swiGLU_ep300_lr2e_3_bs256 --num_workers 4
python eval.py --fold 0 --batch_size 512 --logdir exp070_finetune/biased_conv_transformer_128_8_16_7_RMSNorm_ALiBi_swiGLU_ep300_lr2e_3_bs256

python make_oof.py --batch_size 512 --logdir exp070_finetune/biased_conv_transformer_128_8_16_7_RMSNorm_ALiBi_swiGLU_ep300_lr2e_3_bs256
############################

