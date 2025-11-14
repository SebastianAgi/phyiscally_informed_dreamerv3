#!/bin/bash

docker run -it --rm --gpus "device=1" \
  -v /home/Sebastian/repos/dreamerv3:/app \
  -w /app \
  -v ~/logdir/docker:/logdir \
  -v /home/Sebastian/dataset/isaacsim/object_drop/:/object_drop \
  -e WANDB_API_KEY=a40f3bcefefa7686b1a0e9262153601b5e3694b2 \
  img \
  python dreamerv3/main.py \
  --configs defaults,image_wm \
  --logdir /logdir/dreamerv3-objectdrop-image-only-{timestamp} \
  --task video_dataset \
  --env.video.path /object_drop/ \
  --env.video.pattern "*.png" \
  --env.video.subdir_prefix "_run631" \
  --env.video.max_subdirs 1 \
  --env.video.file_prefix "rgb" \
  --env.video.exts .png \
  --batch_size 16 \
  --batch_length 64 \
  --run.envs 1 \
  --run.steps 300000 \
  --run.train_ratio 256 \
  --run.report_every 50 \
  --run.log_every 50 \
  --run.save_every 500 \
  --logger.outputs jsonl,scope,wandb
  
  # -e CUDA_VISIBLE_DEVICES=1 \
  
  # python dreamerv3/main.py \
  #   --logdir /logdir/{timestamp} \
  #   --configs crafter \
  #   --run.train_ratio 32 \
  #   --batch_size 1 \
  #   --jax.compute_dtype float32