result_dir="results/0310-4"
if [ ! -d $result_dir ]; then
  mkdir -p $result_dir
fi
id="transformer"
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
  --caption_model transformer \
  --reduce_on_plateau \
  --input_json /mnt/ssd/wangjing/Transformer_Captioning/data/cocotalk.json \
  --input_label_h5 /mnt/ssd/wangjing/Transformer_Captioning/data/cocotalk_label.h5 \
  --input_fc_dir /mnt/ssd/wangjing/Transformer_Captioning/data/cocobu_fc_100 \
  --input_att_dir /mnt/ssd/wangjing/Transformer_Captioning/data/cocobu_att_100 \
  --seq_per_img 5 \
  --batch_size 10 \
  --beam_size 2 \
  --learning_rate 2e-5 \
  --num_layers 2 \
  --input_encoding_size 512 \
  --rnn_size 2048 \
  --checkpoint_path $ckpt_path \
  $start_from \
  --save_checkpoint_every 3000 \
  --language_eval 1 \
  --val_images_use 5000 \
  --self_critical_after 10 \
  --result_dir $result_dir
