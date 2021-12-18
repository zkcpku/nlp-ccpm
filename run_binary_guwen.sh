# CUDA_VISIBLE_DEVICES=0,1,2,3 python binary_main.py \
#           --model_name_or_path  ethanyt/guwenbert-base \
#           --do_train --train_file data/train.jsonl \
#           --do_eval  --validation_file data/valid.jsonl \
#           --learning_rate 5e-5  \
#           --save_steps 3000 \
#           --num_train_epochs 10 \
#           --output_dir save/binary/guwen \
#           --per_gpu_eval_batch_size=50 \
#           --per_device_train_batch_size=59 \
#           # --overwrite_output


# CUDA_VISIBLE_DEVICES=6 python binary_main.py \
#           --model_name_or_path  ethanyt/guwenbert-base \
#           --load_model_path save/binary/guwen/checkpoint-3000/pytorch_model.bin \
#           --do_predict --test_file data/valid.jsonl \
#           --learning_rate 5e-5  \
#           --num_train_epochs 5 \
#           --output_dir predict/binary/ \
#           --per_gpu_eval_batch_size=50 \
#           --per_device_train_batch_size=59


CUDA_VISIBLE_DEVICES=6 python binary_main.py \
          --model_name_or_path  ethanyt/guwenbert-base \
          --load_model_path save/binary/guwen/checkpoint-3000/pytorch_model.bin \
          --do_predict --test_file data/test_placeholder.jsonl \
          --learning_rate 5e-5  \
          --num_train_epochs 5 \
          --output_dir predict/binary/ \
          --per_gpu_eval_batch_size=50 \
          --per_device_train_batch_size=59