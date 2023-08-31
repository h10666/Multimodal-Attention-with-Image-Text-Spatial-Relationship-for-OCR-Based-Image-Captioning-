result_dir="results/0928-topdown-1-rl"
if [ ! -d $result_dir ]; then
  mkdir -p $result_dir
fi
id="topdown"
ckpt_path=$result_dir"/log_"$id
if [ ! -d $ckpt_path ]; then
  mkdir $ckpt_path
fi
if [ ! -f $ckpt_path"/infos_"$id".pkl" ]; then
start_from=""
else
start_from="--start_from "$ckpt_path
fi

CUDA_VISIBLE_DEVICES=2 python train.py \
  --id $id \
  --caption_model topdown \
  --input_json ../../Transformer_Captioning/data/cocotalk.json \
  --input_label_h5 ../../Transformer_Captioning/data/cocotalk_label.h5 \
  --input_fc_dir ../../Transformer_Captioning/data/cocobu_fc \
  --input_att_dir ../../Transformer_Captioning/data/cocobu_att \
  --input_box_dir ../../Transformer_Captioning/data/cocobu_box \
  --seq_per_img 5 \
  --batch_size 10 \
  --beam_size 2 \
  --learning_rate 2e-5 \
  --input_encoding_size 1000 \
  --rnn_size 1000 \
  --checkpoint_path $ckpt_path \
  $start_from \
  --save_checkpoint_every 6000 \
  --language_eval 1 \
  --val_images_use 5000 \
  --self_critical_after 25 \
  --result_dir $result_dir \
  --language_eval 1 \
