# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
#           --model_name_or_path  ethanyt/guwenbert-base \
#           --do_train --train_file data/train.jsonl \
#           --do_eval  --validation_file data/valid.jsonl \
#           --learning_rate 5e-4  --fp16 \
#           --num_train_epochs 20 \
#           --save_steps 3000 \
#           --output_dir results_guwen \
#           --per_gpu_eval_batch_size=20 \
#           --per_device_train_batch_size=20 \
        #   --overwrite_output



# CUDA_VISIBLE_DEVICES=0 python main.py \
#           --model_name_or_path  ethanyt/guwenbert-base \
#           --load_model_path results_guwen/pytorch_model.bin \
#           --do_predict --validation_file data/valid.jsonl \
#           --learning_rate 5e-5  \
#           --num_train_epochs 5 \
#           --output_dir predict/results_guwen/ \
#           --per_gpu_eval_batch_size=50 \
#           --per_device_train_batch_size=59


CUDA_VISIBLE_DEVICES=6 python main.py \
          --model_name_or_path  ethanyt/guwenbert-base \
          --load_model_path results_guwen/pytorch_model.bin \
          --do_predict --validation_file /home/zhangkechi/workspace/nlp_course/data/test_placeholder.jsonl \
          --learning_rate 5e-5  \
          --num_train_epochs 5 \
          --output_dir predict/results_guwen/ \
          --per_gpu_eval_batch_size=50 \
          --per_device_train_batch_size=59