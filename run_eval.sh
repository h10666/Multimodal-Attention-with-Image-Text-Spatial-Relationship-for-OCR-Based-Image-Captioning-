CUDA_VISIBLE_DEVICES=0,1 python eval.py \
  --input_json /mnt/ssd/wangjing/textcaps/data/step5_6_6_test/TC_test.json \
  --split test \
  --dump_images 0 \
  --dump_json 0 \
  --num_images -1 \
  --model results/test/log_topdown/model-39516.pth \
  --infos_path results/test/log_topdown/infos_topdown-39516.pkl \
  --language_eval 1 \
  --beam_size 2 \
  --verbose_beam 0