#!/usr/bin/env bash
# Evaluate SHERF on a *RenderPeople-style* custom dataset (e.g. THuman2.0 prepped as below).
#
# Layout (per subject directory; names are relative lines in human_list.txt):
#   cameras.json          # keys camera0000, camera0001, ... (K, R, T per view)
#   img/camera0000/0000.jpg, img/camera0001/0000.jpg, ...
#   mask/camera0000/0000.png, ...
#   outputs_re_fitting/refit_smpl_2nd.npz   # SMPL refit in the same schema as RenderPeople
#
# human_list.txt lives in the *parent* of the subject folder passed as --data.
# Example (your tree):
#   <root>/RenderPeople_recon/20230228/human_list.txt   # one basename per line
#   <root>/RenderPeople_recon/20230228/seq_000001-thuman_0001/   # img/, mask/, cameras.json, ...
# Pass --data as the full path to that subject folder. Lines in human_list.txt should be
# folder names only, e.g. seq_000001-thuman_0001 (relative to the 20230228 directory).
#
# Adjust env vars, then run from repo root:  bash sherf/eval_custom.sh
# or:  cd sherf && bash eval_custom.sh

set -euo pipefail
cd "$(dirname "$0")"

# One subject path (must match a line in human_list.txt). Parent must contain human_list.txt.
# Default assumes data lives next to the repo: ../RenderPeople_recon/20230228/<subject>
export DATA_ROOT="${DATA_ROOT:-../RenderPeople_recon/20230228/seq_000001-thuman_0001}"
export RESUME_PKL="${RESUME_PKL:-logs/training-RenderPeople-runs-450-subject-rs-512-1d-2d-3d-feature-NeRF-decoder-use-trans-sample-obs-view-gpu-4/SHERF_RenderPeople.pkl}"
export OUTDIR="${OUTDIR:-logs/custom-eval-runs}"

# How many subject lines in human_list.txt to use for this eval (from index START below).
export RP_NUM_INSTANCE="${RP_NUM_INSTANCE:-4}"
# Which lines in human_list.txt: [start, end). Use RP_EVAL_HUMAN_END=-1 for "until EOF".
export RP_EVAL_HUMAN_START="${RP_EVAL_HUMAN_START:-0}"
export RP_EVAL_HUMAN_END="${RP_EVAL_HUMAN_END:--1}"

# Four fixed views (camera0000..camera0003). Match order to cameras.json.
export RP_CAMERA_VIEWS="${RP_CAMERA_VIEWS:-4}"
# Single canonical pose index -> only img/.../0000.jpg needed per camera.
export RP_POSES_NUM="${RP_POSES_NUM:-1}"
export RP_POSES_INTERVAL="${RP_POSES_INTERVAL:-1}"

# Observation views to cycle (each produces a subfolder under novel_view/). Use all four:
export RP_EVAL_OBS_VIEWS="${RP_EVAL_OBS_VIEWS:-0,1,2,3}"
# Subsample factor for novel-view targets (1 = keep every eligible camera index).
export RP_EVAL_NV_DATA_INTERVAL="${RP_EVAL_NV_DATA_INTERVAL:-1}"
export RP_EVAL_POSE_NUM="${RP_EVAL_POSE_NUM:-1}"
export RP_EVAL_POSE_INTERVAL="${RP_EVAL_POSE_INTERVAL:-1}"
export RP_EVAL_NP_POSE_START="${RP_EVAL_NP_POSE_START:-0}"
export RP_EVAL_NV_POSE_START="${RP_EVAL_NV_POSE_START:-0}"

python -u train.py \
    --outdir="${OUTDIR}" \
    --cfg=RenderPeople \
    --data="${DATA_ROOT}" \
    --gpus=1 \
    --batch=4 \
    --gamma=5 \
    --aug=noaug \
    --neural_rendering_resolution_initial=512 \
    --gen_pose_cond=True \
    --gpc_reg_prob=0.8 \
    --kimg=1 \
    --workers=2 \
    --use_1d_feature=True \
    --use_2d_feature=True \
    --use_3d_feature=True \
    --use_sr_module=False \
    --sample_obs_view=False \
    --fix_obs_view=True \
    --use_nerf_decoder=True \
    --use_trans=True \
    --test_flag=True \
    --resume="${RESUME_PKL}" \
    --rp-camera-views="${RP_CAMERA_VIEWS}" \
    --rp-num-instance="${RP_NUM_INSTANCE}" \
    --rp-poses-num="${RP_POSES_NUM}" \
    --rp-poses-interval="${RP_POSES_INTERVAL}" \
    --rp-eval-human-start="${RP_EVAL_HUMAN_START}" \
    --rp-eval-human-end="${RP_EVAL_HUMAN_END}" \
    --rp-eval-obs-views="${RP_EVAL_OBS_VIEWS}" \
    --rp-eval-nv-pose-start="${RP_EVAL_NV_POSE_START}" \
    --rp-eval-np-pose-start="${RP_EVAL_NP_POSE_START}" \
    --rp-eval-pose-interval="${RP_EVAL_POSE_INTERVAL}" \
    --rp-eval-pose-num="${RP_EVAL_POSE_NUM}" \
    --rp-eval-nv-data-interval="${RP_EVAL_NV_DATA_INTERVAL}"
