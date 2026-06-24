<div align="center">

<h1 style="font-family: Georgia; font-weight: 600; letter-spacing: 0.5px;">
&#x2728; On the Position Bias of On-Policy Distillation &#x2728;
</h1>
<h3 align="left" style="max-width: 850px; margin: 0 auto; font-weight: 500; line-height: 1.4; color: #444;">
Analyzing position bias in On-Policy Distillation through the lens of constrained-optimization and reweighting OPD via prefix-importance.
</h3>

<br>

<p style="font-family: Charter, serif; font-size: 15px; line-height: 1.6; color: #444;">
<b>Yan Xie</b><sup>*1</sup>,
<b>Sijie Zhu</b><sup>*1</sup>,
<b>Tiansheng Wen</b><sup>2</sup>,
<b>Bo Chen</b><sup>1</sup>,
<b>Yifei Wang</b><sup>3</sup>
</p>

<p style="font-size: 14px; color: #555; margin-top: 8px;">
<sup>1</sup>Xidian University &emsp;
<sup>2</sup>Georgia Institute of Technology &emsp;
<sup>3</sup>Amazon AGI SF Lab
<br>
<sup>*</sup>Equal contribution
</p>

<p>
  <a href="https://arxiv.org/abs/2606.22600">
    <img src="https://img.shields.io/badge/ArXiv-2606.22600-B31B1B?style=flat-square&logo=arxiv" alt="Paper">
  </a>
  <a href="https://yannx1e.github.io/IW-OPD/">
    <img src="https://img.shields.io/badge/Blog-Website-blue?style=flat-square&logo=googlechrome" alt="Blog">
  </a>
  <a href="https://huggingface.co/IW-OPD">
    <img src="https://img.shields.io/badge/Hugging%20Face-IW--OPD-yellow?style=flat-square&logo=huggingface" alt="Hugging Face">
  </a>
  <a href="https://github.com/YannX1e/Importance-Weighted-On-Policy-Distillation">
    <img src="https://img.shields.io/badge/Code-GitHub-black?style=flat-square&logo=github" alt="Code">
  </a>
</p>

<img src="./assets/on-position-bias-of-on-policy-distillation.png" width="850" style="border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.15);" alt="On the Position Bias of On-Policy Distillation">

<br>
<br>

</div>

## &#x1F680; &#x1F680; News
- 2026.06.24 We release our code.
- 2026.06.21 We release our paper on [arXiv](https://arxiv.org/abs/2606.22600).
- 2026.06.17 We release our work in [blog](https://yannx1e.github.io/IW-OPD/).

This repository contains the training and evaluation code for **IW-OPD**: an importance-weighted variant of on-policy distillation that reweights token-level OPD advantages by cumulative teacher--student disagreement along the sampled response. The implementation is based on `verl` PPO training with on-policy student rollouts, teacher log-probability evaluation on the same trajectories, and a stop-gradient IW-OPD weight applied directly to the PPO advantage.

## Installation

The training code is built on `verl` and uses vLLM for rollout generation.

```bash
conda create -n iw-opd python=3.10
conda activate iw-opd

cd IW-OPD/verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install math-verify
```

## Models and Data

The default training script expects the following local layout:

```text
IW-OPD/
|-- data/
|   |-- DeepMath-103K/train_filtered_level6.parquet
|   |-- AIME2024/test.parquet
|   `-- AIME2025/test.parquet
|-- models/
|   |-- Qwen3-4B/
|   `-- Qwen3-30B-A3B-Instruct-2507/
`-- verl/
```

Download the student and teacher models from Hugging Face:

- Student: [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
- Teacher: [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)

For example:

```bash
huggingface-cli download Qwen/Qwen3-4B \
  --local-dir models/Qwen3-4B

huggingface-cli download Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --local-dir models/Qwen3-30B-A3B-Instruct-2507
```

The math training and validation parquet files can be downloaded from the released OPD training data:

- Data: [Keven16/G-OPD-Training-Data](https://huggingface.co/datasets/Keven16/G-OPD-Training-Data)

Place the required files under `data/` as shown above. The main script uses `DeepMath-103K/train_filtered_level6.parquet` for training and evaluates on `AIME2024/test.parquet` and `AIME2025/test.parquet` during training.

## Training

The main IW-OPD example distills a Qwen3-4B student from a Qwen3-30B-A3B-Instruct-2507 teacher on 4 nodes with 8 GPUs per node.

```bash
cd IW-OPD/verl
bash examples/opd/run_qwen3-30b-a3b-instruct-opd_4b_iw_opd.sh
```

The script sets:

- `trainer.nnodes=4`
- `trainer.n_gpus_per_node=8`
- `actor_rollout_ref.model.path=../models/Qwen3-4B`
- `actor_rollout_ref.ref.model.path=../models/Qwen3-30B-A3B-Instruct-2507`
- `actor_rollout_ref.actor.policy_loss.iw_opd_weight_enable=true`
- `actor_rollout_ref.actor.policy_loss.iw_opd_weight_max=1.5`

You can override paths and the IW-OPD weight without editing the script:

```bash
DATA_ROOT=/path/to/data \
STUDENT_MODEL_PATH=/path/to/Qwen3-4B \
TEACHER_MODEL_PATH=/path/to/Qwen3-30B-A3B-Instruct-2507 \
OUTPUT_DIR=/path/to/checkpoints \
bash examples/opd/run_qwen3-30b-a3b-instruct-opd_4b_iw_opd.sh
```

To run the vanilla OPD baseline with the same student, teacher, data, and distributed settings, disable the IW-OPD weight:

```bash
IW_OPD_WEIGHT_ENABLE=false \
OUTPUT_DIR=/path/to/opd-checkpoints \
bash examples/opd/run_qwen3-30b-a3b-instruct-opd_4b_iw_opd.sh
```

For a multi-node run, start or submit the job from the Ray head node after all worker nodes are visible to Ray. The script itself only specifies the logical resource layout; cluster launch details depend on the scheduler used by your environment.

## Evaluation

Math evaluation scripts are in `math_eval/`.

```bash
cd IW-OPD/math_eval
bash run_eval_math.sh
```

Code evaluation uses EvalPlus-compatible scripts in `code_eval/scripts/`.

```bash
cd IW-OPD
CUDA_VISIBLE_DEVICES=0 bash code_eval/scripts/run_evalplus.sh humaneval <MODEL_PATH> 0 1.0 1.0 4
```

## Acknowledgments

This codebase is built on top of [G-OPD](https://github.com/RUCBM/G-OPD) and `verl`. We thank the G-OPD authors for releasing their implementation and training data.

## Citation

If you find this work useful, please cite:

```bibtex
@misc{xie2026positionbiasopd,
  title={On the Position Bias of On-Policy Distillation},
  author={Xie, Yan and Zhu, Sijie and Wen, Tiansheng and Chen, Bo and Wang, Yifei},
  year={2026},
  eprint={2606.22600},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2606.22600}
}
```
