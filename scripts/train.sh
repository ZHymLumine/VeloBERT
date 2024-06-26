CUDA_VISIBLE_DEVICES=0 python ../velobert/train.py \
    --do_train \
    --do_eval \
    --aa \
    --ss \
    --batch_size 8 \
    --max_seq_length 512 \
    --learning_rate 1e-4 \
    --data_dir /home/lr/zym/research/VeloBERT/dataset \
    --model_name_or_path /home/lr/zym/research/VeloBERT/3-new-12w-0 \
    --output_dir /raid_elmo/home/lr/zym/velobert_calm_test/outputs \
    --num_train_epoch 200 \
    --calm \

