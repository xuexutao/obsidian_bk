---
pretty_name: Hy-Nomas-Bench
license: apache-2.0
language:
  - en
  - zh
size_categories:
  - n<1K
task_categories:
  - text-to-video
tags:
  - hy-nomas
  - world-model
  - video-generation
  - interactive-video-generation
  - multi-turn
  - camera-control
  - spatial-memory
  - evaluation
configs:
  - config_name: bench1_multiturn_text_control
    data_files:
      - split: test
        path: bench1_multiturn_text_control/bench1_multiturn_text_control.json
  - config_name: bench2_multiturn_camera_control
    data_files:
      - split: test
        path: bench2_multiturn_camera_control/bench2_multiturn_camera_control.json
  - config_name: bench3_multiturn_interleaved_control
    data_files:
      - split: test
        path: bench3_multiturn_interleaved_control/bench3_multiturn_interleaved_control.json
  - config_name: bench4_spatial_memory
    data_files:
      - split: test
        path: bench4_spatial_memory/bench4_spatial_memory.json
---

<p align="left">
  <a href="https://huggingface.co/datasets/tencent/Hy-Nomas-Bench/blob/main/README_CN.md">中文</a>&nbsp;｜&nbsp;English
</p>

# Hy-Nomas-Bench

**280 test cases for interaction controllability in video world models.**

<p align="center">
  <a href="https://github.com/Tencent-Hunyuan/Hy-Nomas"><img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&amp;logo=github&amp;logoColor=white" alt="GitHub"/></a>
  <a href="https://github.com/Tencent-Hunyuan/Hy-Nomas/blob/main/Hy_Nomas_Tech_Report.pdf"><img src="https://img.shields.io/badge/Technical%20Report%20%28Coming%20Soon%29-B31B1B?style=flat-square&amp;logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cGF0aCBkPSJNMTQgMkg1djIwaDE0Vjd6TTE0IDJ2NWg1TTggMTJoOE04IDE2aDgiLz48L3N2Zz4%3D" alt="Technical Report (Coming Soon)"/></a>
  <a href="https://huggingface.co/tencent/Hy-Nomas"><img src="https://img.shields.io/badge/Model%20Weights-FFD21E?style=flat-square&amp;logo=huggingface&amp;logoColor=black" alt="Model Weights"/></a>
  <a href="https://aistudio.tencent.com/"><img src="https://img.shields.io/badge/Official%20Website-666666?style=flat-square&amp;logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIj48Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIvPjxlbGxpcHNlIGN4PSIxMiIgY3k9IjEyIiByeD0iNCIgcnk9IjEwIi8%2BPHBhdGggZD0iTTIgMTJoMjAiLz48L3N2Zz4%3D" alt="Official Website"/></a>
</p>

<h3 align="center">
  🚀 <a href="https://hy-nomas.hy.tencent.com/">Try the Demo →</a>
</h3>

Hy-Nomas-Bench evaluates continuous video generation under text instructions, keyboard camera controls, and combinations of both. The release contains **structured prompts, interaction sequences, and 220 reference camera trajectories**, organized into four evaluation-only `test` splits. Videos, extracted reference/revisit frames, and evaluation scores are not included.

Scenarios cover real-world, game, CG, and animated environments. Prompts are mainly English, with a few Chinese or mixed-language user prompts in Bench-1.

## Benchmarks

| Benchmark | Cases | Primary evaluation | Turns / segments | Mean duration |
|:---|---:|:---|:---:|---:|
| [Bench-1: Multi-turn text](bench1_multiturn_text_control/bench1_multiturn_text_control.json) | 60 | Prompt following, physical plausibility | 2–6 | 37.88 s |
| [Bench-2: Camera navigation](bench2_multiturn_camera_control/bench2_multiturn_camera_control.json) | 100 | Camera control | 4–20 | 27.05 s |
| [Bench-3: Interleaved text and camera](bench3_multiturn_interleaved_control/bench3_multiturn_interleaved_control.json) | 60 | Prompt following, physical plausibility, camera control | 2–7 | 30.59 s |
| [Bench-4: Spatial memory](bench4_spatial_memory/bench4_spatial_memory.json) | 60 | Memory consistency on revisits | 8 | 18.00 s |

- **Multi-turn text:** Successive text instructions test action execution and subject consistency across continuous generation.
- **Camera navigation:** Covers 49 scenes, eight basic camera actions, and 24 two-key combinations. Each fixed action lasts 2–4 seconds and is paired with an equal-duration inverse return.
- **Interleaved control:** The first segment is TEXT; `segment_control="prompt"` disables camera conditioning, while `"camera"` supplies environment text and camera controls. The release contains 125 TEXT and 68 CAMERA segments.
- **Spatial memory:** Each first-person scenario follows four outbound/return pairs around a distinctive static target. Real/game-style indoor/outdoor settings are balanced. Compare generated frames at matching effective camera poses to assess object and scene consistency.

Statistics describe the released JSON files; turn counts include initialization. Requested durations range from 8 to 86 seconds.

## Files and Core Fields

The four directory names are also the dataset configuration names:

```text
bench1_multiturn_text_control/
bench2_multiturn_camera_control/
bench3_multiturn_interleaved_control/
bench4_spatial_memory/
```

Each directory contains `<directory_name>.json` (a list of cases) and `benchmark_config.json` (reference rules). Bench-2/3/4 also contain `trajectories/` with 100/60/60 NPZ files. Each dataset row represents one case; loading the JSON does not load the NPZ arrays.

| Field | Meaning |
|:---|:---|
| `test_id`, `total_duration` | Unique public case ID and requested duration (`MM:SS.mmm`). |
| `perspective_view`, `Style` | Viewpoint and visual style; the uppercase `S` is part of the field name. |
| `subject_list` | Subject IDs, display names, and descriptions. |
| `segments` | Ordered timestamps, visual descriptions, changes, and subject actions; used for offline inference. |
| `user_prompts` | Bench-1/3/4 user interactions, aligned with `segments`; `turn_number` starts at 1 and includes initialization. Bench-4 includes the original scene text and per-turn `camera_control` requests. |
| `scene_id`, `scene_name`, `scene_category`, `scene_environment` | Bench-2/4 scene grouping, English names, and categories. |
| `camera_trajectory_spec` | Bench-2/3/4 trajectory reference and camera-control information. |

**Resolve `camera_trajectory_spec.ground_truth_npz` relative to its JSON file.** Keep each JSON beside its trajectory directory for inference. Frame ranges are start-inclusive and end-exclusive. Bench-2/4 use `w/a/s/d` and `Arrow*` key codes; Bench-3 uses `KeyW/KeyA/KeyS/KeyD` and `Arrow*` in its timed key intervals.

See [Data Format](docs/data_format.md) for detailed field definitions, JSON loading, and NPZ array conventions.

## Inference Guide

Run all four benchmarks with the Hy-Nomas codebase using **`scripts/inference/inference_from_config.sh`** and **`configs/experiment/inference.yaml`**. Pass each benchmark JSON directly through `inference.prompt`. All inference settings use the model preset defaults; only the model, input, and output paths need to be set. No dataset-specific reader or configuration-loading step is required. This runs offline inference from the structured `segments` descriptions and does not require the online prompt-enhancement service.

### 1. Clone the Codebase and Set Up the Environment

Download this dataset repository, including the `trajectories/` directories, and set `HY_NOMAS_BENCH_PATH` below to its absolute local path. Prepare a Linux host with a compatible NVIDIA driver, Docker, and NVIDIA Container Toolkit.

```bash
export HY_NOMAS_BENCH_PATH=/absolute/path/to/Hy-Nomas-Bench

git clone https://github.com/Tencent-Hunyuan/Hy-Nomas.git
cd Hy-Nomas

export HY_NOMAS_IMAGE=hynomasoss/hynomasoss:cu130
docker pull "$HY_NOMAS_IMAGE"
docker run --rm -it --gpus all --ipc=host --network=host \
  -v "$PWD:/workspace/Hy-Nomas" \
  -v "$HY_NOMAS_BENCH_PATH:/workspace/Hy-Nomas-Bench:ro" \
  -w /workspace/Hy-Nomas \
  "$HY_NOMAS_IMAGE" bash
```

Run the remaining commands inside this container, from `/workspace/Hy-Nomas`. The dataset is mounted at `/workspace/Hy-Nomas-Bench`; downloaded weights and generated outputs are saved in the mounted code directory. For manual installation or single-GPU inference, see the codebase's usage guide in `README.md` and its `configs/experiment/single_gpu_low_mem_inference.yaml` preset.

### 2. Download the Model Weights

```bash
export Hy_Nomas_Model_Path=/workspace/Hy-Nomas/weights
export HY_NOMAS_BENCH_PATH=/workspace/Hy-Nomas-Bench

hf download tencent/Hy-Nomas --local-dir "$Hy_Nomas_Model_Path"
```

The commands below use the distilled DiT and fine-tuned PixelUMM decoder. If the weights are already downloaded, mount that directory into the container and set `Hy_Nomas_Model_Path` to its container-visible path. The preset also loads `Qwen/Qwen2.5-VL-7B-Instruct` as its text encoder, so it must be accessible through Hugging Face or the local cache.

### 3. Run the Four Benchmarks

Define the helper and run the four benchmarks in sequence.

```bash
run_benchmark() {
  local benchmark="$1"
  bash scripts/inference/inference_from_config.sh configs/experiment/inference.yaml \
    dit_weight="${Hy_Nomas_Model_Path}/DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors" \
    model.vae.checkpoint="${Hy_Nomas_Model_Path}/VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors" \
    inference.prompt="${HY_NOMAS_BENCH_PATH}/${benchmark}/${benchmark}.json" \
    inference.save_path="./outputs/Hy-Nomas-Bench/${benchmark}"
}

run_benchmark bench1_multiturn_text_control && \
run_benchmark bench2_multiturn_camera_control && \
run_benchmark bench3_multiturn_interleaved_control && \
run_benchmark bench4_spatial_memory
```

To run only one benchmark, call `run_benchmark` with its directory name after defining the function. Videos and accompanying inference files are saved under `outputs/Hy-Nomas-Bench/<benchmark>/`. The preset uses 24 FPS, 8 denoising steps, seed 42, and the case timestamps to determine segment durations; it processes all cases and skips existing outputs by default.

Bench-1 uses text only. For Bench-2/3/4, the loader resolves `camera_trajectory_spec.ground_truth_npz` relative to the benchmark JSON; keep each JSON beside its `trajectories/` directory. Bench-3 automatically switches camera conditioning according to `segments[].segment_control`. The default normalization and trajectory gains satisfy Bench-4's rotation-compensation requirements. The default `inference.overlay_keys=true` adds keyboard overlays to camera-controlled videos. These commands generate videos; scoring is a separate step.

## Evaluation and License

Replay cases in order and report case IDs, prompt representation, generation settings, and any exclusions. These benchmarks provide evaluation inputs and geometry; scoring is a separate step, and the scenarios do not cover every type of world-model behavior.

Released under [Apache License 2.0](LICENSE). For questions, open an [issue](https://github.com/Tencent-Hunyuan/Hy-Nomas/issues) or contact [hy_nomas@tencent.com](mailto:hy_nomas@tencent.com).
