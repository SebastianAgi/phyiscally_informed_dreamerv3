#!/bin/bash

docker run -it --rm --gpus 2 \
  -v /home/Sebastian/repos/dreamerv3:/app \
  -w /app \
  -v ~/logdir:/logdir \
  -v /home/Sebastian/dataset/isaacsim/object_drop/:/object_drop \
  -e WANDB_API_KEY=a40f3bcefefa7686b1a0e9262153601b5e3694b2 \
  img \
  python dreamerv3/main.py \
  --configs defaults,image_wm \
  --logdir ~/logdir/dreamerv3-eval-objectdrop-{timestamp} \
  --task video_dataset \
  --script recon_eval \
  --env.video.path /object_drop/_run1 \
  --env.video.pattern "*.png" \
  --env.video.file_prefix "rgb" \
  --env.video.exts .png \
  --batch_size 1 \
  --batch_length 64 \
  --run.envs 1 \
  --run.steps 320 \
  --run.log_every 50 \
  --run.report_every 50 \
  --run.from_checkpoint /logdir/docker/dreamerv3-objectdrop-image-only-20251103T012549/ckpt/20251103T030702F757016 \
  --logger.outputs jsonl,scope,wandb