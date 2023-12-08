######## baseline #########
python train.py --seed 2023 --fold 0 --epoch 300 --batch_size 64 --gpus 4 --lr 1e-3 --logdir biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256 --num_workers 8 --disable_compile
python train_finetune.py --seed 2023 --fold 0 --epoch 15 --batch_size 64 --gpus 4 --lr 5e-4 \
   --logdir biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256 --resumedir exp064/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256 --num_workers 8
python eval.py --fold 0 --batch_size 512 --logdir exp064_finetune/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256 
python eval_with_err.py --fold 0 --batch_size 512 --logdir exp064_finetune/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256 
python apply_filter_pl.py --fold 0 --logdir exp064_finetune/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256 --filter 0.75

python train.py --seed 2024 --fold 1 --epoch 300 --batch_size 64 --gpus 4 --lr 1e-3 --logdir biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256 --num_workers 8 --disable_compile
python train_finetune.py --seed 2024 --fold 1 --epoch 15 --batch_size 64 --gpus 4 --lr 5e-4 \
   --logdir biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256 --resumedir exp064/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256 --num_workers 8
python eval.py --fold 1 --batch_size 512 --logdir exp064_finetune/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256 
python eval_with_err.py --fold 1 --batch_size 512 --logdir exp064_finetune/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256 
python apply_filter_pl.py --fold 1 --logdir exp064_finetune/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256 --filter 0.75

python train.py --seed 2025 --fold 2 --epoch 300 --batch_size 64 --gpus 4 --lr 1e-3 --logdir biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256 --num_workers 8 --disable_compile
python train_finetune.py --seed 2025 --fold 2 --epoch 15 --batch_size 64 --gpus 4 --lr 5e-4 \
   --logdir biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256 --resumedir exp064/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256 --num_workers 8
python eval.py --fold 2 --batch_size 512 --logdir exp064_finetune/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256 
python eval_with_err.py --fold 2 --batch_size 512 --logdir exp064_finetune/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256 
python apply_filter_pl.py --fold 2 --logdir exp064_finetune/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256 --filter 0.75

python train.py --seed 2026 --fold 3 --epoch 300 --batch_size 64 --gpus 4 --lr 1e-3 --logdir biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256 --num_workers 8 --disable_compile
python train_finetune.py --seed 2026 --fold 3 --epoch 15 --batch_size 64 --gpus 4 --lr 5e-4 \
   --logdir biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256 --resumedir exp064/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256 --num_workers 8
python eval.py --fold 3 --batch_size 512 --logdir exp064_finetune/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256 
python eval_with_err.py --fold 3 --batch_size 512 --logdir exp064_finetune/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256 
python apply_filter_pl.py --fold 3 --logdir exp064_finetune/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256 --filter 0.75

python train.py --seed 2027 --fold 4 --epoch 300 --batch_size 64 --gpus 4 --lr 1e-3 --logdir biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256 --num_workers 8 --disable_compile
python train_finetune.py --seed 2027 --fold 4 --epoch 15 --batch_size 64 --gpus 4 --lr 5e-4 \
   --logdir biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256 --resumedir exp064/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256 --num_workers 8
python eval.py --fold 4 --batch_size 512 --logdir exp064_finetune/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256 
python eval_with_err.py --fold 4 --batch_size 512 --logdir exp064_finetune/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256 
python apply_filter_pl.py --fold 4 --logdir exp064_finetune/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256 --filter 0.75

python make_oof.py --batch_size 512 --logdir exp064_finetune/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256 
###########################
