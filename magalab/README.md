# Maga Lab scripts (not for upstream)

This directory exists only on the `magalab` branch of the muratmaga/ANTsTorch fork. The library
changes these scripts rely on are on `mem-opt`, the branch to open pull requests from against
ANTsX/ANTsTorch. `magalab` is `mem-opt` plus this directory, so nothing here goes into a PR made
from `mem-opt`.

| Directory | What it is |
|---|---|
| `e15_benchmarks/` | ANTsTorch GPU registration and memory tests on the E15.5 template (18 µm), run on the RTX PRO 6000 Blackwell |
| `e18_gpu_template/` | E18 template rounds with GPU SyN on the Sasquatch A100s: per-subject registration, Slurm job scripts, convergence and multi-GPU tests. `README.md` is the lab notes on running a GPU round |
| `lamnr_e15/` | LAMNr normative-flow model of wild-type E15.5 embryos, scored against exencephaly (M3T) and Gli2 genotypes |

Paths in the scripts point to lab storage (`/home2/...`, `/data/hps/...`) and have to be edited
to run elsewhere. Images, masks, crops, checkpoints and score tables are not in the repository.
