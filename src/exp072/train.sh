####### baseline #########
python train.py --seed 7023 --fold 0 --epoch 300 --head_size 24 --batch_size 32 --gpus 4 --lr 1e-3 --logdir biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 \
   --pseudo_label_df ../../logs/exp071_finetune/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256/fold0/test_pl_filterling_0.75_half.csv \
   --num_workers 4 --disable_compile
python train_finetune.py --seed 7023 --fold 0 --epoch 15 --batch_size 32 --gpus 4 --lr 5e-4 \
   --logdir biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 --resumedir exp072/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 \
   --num_workers 4 \
   --pseudo_label_df ../../logs/exp071_finetune/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256/fold0/test_pl_filterling_0.75_half.csv 
python eval.py --fold 0 --batch_size 512 --logdir exp072_finetune/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 

python train.py --seed 7024 --fold 1 --epoch 300 --head_size 24 --batch_size 32 --gpus 4 --lr 1e-3 --logdir biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 \
   --pseudo_label_df ../../logs/exp071_finetune/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256/fold1/test_pl_filterling_0.75_half.csv \
   --num_workers 4 --disable_compile
python train_finetune.py --seed 7024 --fold 1 --epoch 15 --batch_size 32 --gpus 4 --lr 5e-4 \
   --logdir biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 --resumedir exp072/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 \
   --num_workers 4 \
   --pseudo_label_df ../../logs/exp071_finetune/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256/fold1/test_pl_filterling_0.75_half.csv 
python eval.py --fold 1 --batch_size 512 --logdir exp072_finetune/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 

python train.py --seed 7025 --fold 2 --epoch 300 --head_size 24 --batch_size 32 --gpus 4 --lr 1e-3 --logdir biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 \
   --pseudo_label_df ../../logs/exp071_finetune/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256/fold2/test_pl_filterling_0.75_half.csv \
   --num_workers 4 --disable_compile
python train_finetune.py --seed 7025 --fold 2 --epoch 15 --batch_size 32 --gpus 4 --lr 5e-4 \
   --logdir biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 --resumedir exp072/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 \
   --num_workers 4 \
   --pseudo_label_df ../../logs/exp071_finetune/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256/fold2/test_pl_filterling_0.75_half.csv 
python eval.py --fold 2 --batch_size 512 --logdir exp072_finetune/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 

python train.py --seed 7026 --fold 3 --epoch 300 --head_size 24 --batch_size 32 --gpus 4 --lr 1e-3 --logdir biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 \
   --pseudo_label_df ../../logs/exp071_finetune/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256/fold3/test_pl_filterling_0.75_half.csv \
   --num_workers 4 --disable_compile
python train_finetune.py --seed 7026 --fold 3 --epoch 15 --batch_size 32 --gpus 4 --lr 5e-4 \
   --logdir biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 --resumedir exp072/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 \
   --num_workers 4 \
   --pseudo_label_df ../../logs/exp071_finetune/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256/fold3/test_pl_filterling_0.75_half.csv 
python eval.py --fold 3 --batch_size 512 --logdir exp072_finetune/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 

python train.py --seed 7027 --fold 4 --epoch 300 --head_size 24 --batch_size 32 --gpus 4 --lr 1e-3 --logdir biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 \
   --pseudo_label_df ../../logs/exp071_finetune/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256/fold4/test_pl_filterling_0.75_half.csv \
   --num_workers 4 --disable_compile
python train_finetune.py --seed 7027 --fold 4 --epoch 15 --batch_size 32 --gpus 4 --lr 5e-4 \
   --logdir biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 --resumedir exp072/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 \
   --num_workers 4 \
   --pseudo_label_df ../../logs/exp071_finetune/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256/fold4/test_pl_filterling_0.75_half.csv 
python eval.py --fold 4 --batch_size 512 --logdir exp072_finetune/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 

python make_oof.py --batch_size 512 --logdir exp072_finetune/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256 
##########################