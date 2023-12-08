####### baseline #########
python train.py --seed 6023 --fold 0 --epoch 300 --head_size 24 --batch_size 64 --gpus 4 --lr 1e-3 --logdir biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 --num_workers 8 --disable_compile
python train_finetune.py --seed 6023 --fold 0 --epoch 15 --batch_size 64 --gpus 4 --lr 5e-4 \
   --logdir biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 --resumedir exp071/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 --num_workers 8
python eval.py --fold 0 --batch_size 512 --logdir exp071_finetune/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 
python eval_with_err.py --fold 0 --batch_size 512 --logdir exp071_finetune/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 
python apply_filter_pl.py --fold 0 --logdir exp071_finetune/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 --filter 0.75

python train.py --seed 6024 --fold 1 --epoch 300 --head_size 24 --batch_size 64 --gpus 4 --lr 1e-3 --logdir biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 --num_workers 8 --disable_compile
python train_finetune.py --seed 6024 --fold 1 --epoch 15 --batch_size 64 --gpus 4 --lr 5e-4 \
   --logdir biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 --resumedir exp071/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 --num_workers 8
python eval.py --fold 1 --batch_size 512 --logdir exp071_finetune/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 
python eval_with_err.py --fold 1 --batch_size 512 --logdir exp071_finetune/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 
python apply_filter_pl.py --fold 1 --logdir exp071_finetune/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 --filter 0.75

python train.py --seed 6025 --fold 2 --epoch 300 --head_size 24 --batch_size 64 --gpus 4 --lr 1e-3 --logdir biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 --num_workers 8 --disable_compile
python train_finetune.py --seed 6025 --fold 2 --epoch 15 --batch_size 64 --gpus 4 --lr 5e-4 \
   --logdir biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 --resumedir exp071/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 --num_workers 8
python eval.py --fold 2 --batch_size 512 --logdir exp071_finetune/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 
python eval_with_err.py --fold 2 --batch_size 512 --logdir exp071_finetune/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 
python apply_filter_pl.py --fold 2 --logdir exp071_finetune/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 --filter 0.75

python train.py --seed 6026 --fold 3 --epoch 300 --head_size 24 --batch_size 64 --gpus 4 --lr 1e-3 --logdir biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 --num_workers 8 --disable_compile
python train_finetune.py --seed 6026 --fold 3 --epoch 15 --batch_size 64 --gpus 4 --lr 5e-4 \
   --logdir biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 --resumedir exp071/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 --num_workers 8
python eval.py --fold 3 --batch_size 512 --logdir exp071_finetune/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 
python eval_with_err.py --fold 3 --batch_size 512 --logdir exp071_finetune/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 
python apply_filter_pl.py --fold 3 --logdir exp071_finetune/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 --filter 0.75

python train.py --seed 6027 --fold 4 --epoch 300 --head_size 24 --batch_size 64 --gpus 4 --lr 1e-3 --logdir biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 --num_workers 8 --disable_compile
python train_finetune.py --seed 6027 --fold 4 --epoch 15 --batch_size 64 --gpus 4 --lr 5e-4 \
   --logdir biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 --resumedir exp071/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 --num_workers 8
python eval.py --fold 4 --batch_size 512 --logdir exp071_finetune/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 
python eval_with_err.py --fold 4 --batch_size 512 --logdir exp071_finetune/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 
python apply_filter_pl.py --fold 4 --logdir exp071_finetune/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 --filter 0.75

python make_oof.py --batch_size 512 --logdir exp071_finetune/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 
##########################
