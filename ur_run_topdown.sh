result_dir="results/4"
if [ ! -d $result_dir ]; then
  mkdir -p $result_dir
fi
id="topdown"
ckpt_path=$result_dir"/log_"$id
if [ ! -d $ckpt_path ]; then
  mkdir $ckpt_path
fi
if [ ! -f $ckpt_path"/infos_"$id"-best.pkl" ]; then
start_from=""
else
start_from="--start_from "$ckpt_path
fi

CUDA_VISIBLE_DEVICES=1 python train.py --id $id \
  --caption_model topdown \
  --input_json ../data/step5_6_6/TC.json \
  --input_label_h5 ../data/step5_6_6/TC_label.h5 \
  --input_fc_dir ../data/open_images/resnet152 \
  --input_att_dir ../data/open_images/detectron_fix_100/fc6 \
  --input_box_dir ../data/open_images/detectron_fix_100/fc6 \
  --input_ocr_dir ../data/m4c_textvqa_ocr_en_frcn_features \
  --batch_size 10 \
  --learning_rate_decay_every 3 \
  --learning_rate 2e-4 \
  --learning_rate_decay_start 0 \
  --scheduled_sampling_start 0 \
  --checkpoint_path $ckpt_path \
  $start_from \
  --language_eval 1 \
  --val_images_use -1 \
  --max_epochs 100 \
  --result_dir $result_dir \
  --beam_size 2 \
  --rnn_size 1000 \
  --input_encoding_size 1000 \
  --ocr_size 50 \
  --iteration_eval_every 100000000000000000 \
  --epoch_eval_every 1 \
  --epoch_test_more 100