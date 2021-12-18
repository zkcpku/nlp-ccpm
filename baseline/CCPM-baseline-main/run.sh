# CUDA_VISIBLE_DEVICES=6 python main.py run.json
        #   --model_name_or_path  hfl/chinese-roberta-wwm-ext \
        #   --do_train --train_file data/train.jsonl \
        #   --do_eval  --validation_file data/valid.jsonl \
        #   --learning_rate 5e-5  --fp16 \
        #   --num_train_epochs 3 \
        #   --output_dir results_o \
        #   --per_gpu_eval_batch_size=16 \
        #   --per_device_train_batch_size=16 \
        #   --overwrite_output


CUDA_VISIBLE_DEVICES=6 python main.py \
          --model_name_or_path  hfl/chinese-roberta-wwm-ext \
          --load_model_path results_o/pytorch_model.bin \
          --do_predict --validation_file /home/zhangkechi/workspace/nlp_course/data/test_placeholder.jsonl \
          --learning_rate 5e-5  \
          --num_train_epochs 5 \
          --output_dir predict/results_guwen/ \
          --per_gpu_eval_batch_size=50 \
          --per_device_train_batch_size=59