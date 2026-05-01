# Self-Hosted NVIDIA GPU Runner

Use a self-hosted runner for full benchmark execution on a local NVIDIA machine. The GitHub Pages workflow should remain lightweight and should only publish already-generated static data. Method pull requests should be validated separately with CPU-only smoke checks; full GPU benchmark results should be produced in a separate maintainer commit.

## Recommended Security Model

- Do not run untrusted pull-request code on your local GPU runner.
- Use `workflow_dispatch` or trusted-branch triggers only.
- Run the runner as a dedicated low-privilege OS user.
- Give the runner a specific label such as `gpu`; do not use it for generic CI jobs.
- Review method-contribution PRs with CPU smoke tests first, then run full GPU benchmarks after approval or after merge.
- Commit regenerated benchmark artifacts separately from the method-code PR.

## One-Time Machine Setup

Install NVIDIA drivers and verify CUDA visibility:

```bash
nvidia-smi
```

Install system packages:

```bash
sudo apt-get update
sudo apt-get install -y git python3 python3-venv python3-pip
```

In GitHub, go to:

```text
Repository -> Settings -> Actions -> Runners -> New self-hosted runner
```

Follow GitHub's commands for Linux x64. When configuring labels, include:

```text
self-hosted, linux, x64, gpu
```

For a persistent service, run GitHub's generated service commands, usually:

```bash
sudo ./svc.sh install
sudo ./svc.sh start
```

## Benchmark Workflow

The `.github/workflows/benchmark-self-hosted.yml` workflow runs only when manually dispatched. It:

- checks out the repository,
- verifies `nvidia-smi`,
- creates or refreshes `methods/.venv`,
- installs `methods/requirements.txt` with CUDA PyTorch wheels,
- optionally regenerates datasets,
- runs the benchmark suite with `--device cuda`,
- regenerates LaTeX and website data assets,
- uploads benchmark artifacts.

The default workflow settings use:

```text
jobs=30
threads_per_worker=1
max_gpu_workers=2
input_channel=u_cmd
```

Those match the current local workstation assumptions: high CPU parallelism, one BLAS thread per worker, and at most two concurrent GPU-training methods.

## Local Dry Run

Before using GitHub Actions, run this directly:

```bash
python3 -m venv methods/.venv
methods/.venv/bin/python -m pip install --upgrade pip
methods/.venv/bin/python -m pip install -r methods/requirements.txt
./results.py suite --device cuda --jobs 30 --threads-per-worker 1 --max-gpu-workers 2 --input-channel u_cmd
./results.py latex-assets
./results.py web-data
```

If GPU use is not visible in `nvidia-smi`, check:

```bash
methods/.venv/bin/python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

## Two-Phase Results Update

1. Merge or check out the trusted method-code change.
2. Run the full benchmark locally or through the manual self-hosted workflow.
3. Review regenerated CSV files, figures, LaTeX assets, website JSON, and paper output.
4. Commit only the trusted generated results in a separate commit.

## Updating the Public Site

After a trusted self-hosted benchmark run updates `methods/results` and `site/public/data`, commit those changes and push to `main`. The normal Pages workflow then publishes the static site.
