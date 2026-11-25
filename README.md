# DigitDefender — CSC 8370 Data Security
Monica Sai Chalasani and Harika Chinnam | Fall 2026

## Project status and tested environment
All five MNIST experiments were completed on an Apple Silicon MacBook Air using CPU on November 24, 2026. The smoke test passed. The saved checkpoints, CSV/JSON results and seven plots are included. Tested Python series: 3.9 (exact patch version not recorded).

| Package | Tested version |
| --- | --- |
| PyTorch | 2.8.0 |
| torchvision | 0.23.0 |
| NumPy | 2.0.2 |
| Matplotlib | 3.9.4 |
| ReportLab | 4.5.1 |
| Pillow | 11.3.0 |
| pip | 26.0.1 |

## Setup (macOS/Linux)
From the extracted project directory:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python test_smoke.py
```
Use Python 3.9 for the recorded environment. On Windows activate with `.venv\Scripts\activate`. Internet access is required for the first MNIST download and package installation. The program selects CUDA if available, otherwise CPU.

## Dataset and instructor template alignment
Course references (provided in the official project handout):
- Level 1: https://github.com/DongYang26/CSC8370_26FullDemo/blob/main/Templates/dataloader4level1.py
- Levels 2/3: https://github.com/DongYang26/CSC8370_26FullDemo/blob/main/Templates/Level_2_solution_1.py
- Attack interface: https://github.com/DongYang26/CSC8370_26FullDemo/blob/main/Templates/malicious_client.py

`data_loader.py` adapts these dataset-loading conventions: torchvision MNIST, root `./data/mnist`, tensor conversion and normalization with mean 0.5 and standard deviation 0.5. Like the federated template, it partitions data into ten equal random subsets. Our implementation reserves 5,000 of the 60,000 training images for validation, uses 55,000 for training (5,500 per client), and retains the 10,000-image standard test split. Fixed seeds 8370 and 6370 control the splits. Batch sizes are 128 for training and 512 for evaluation. These are intentional experiment settings, not identical copies of the template defaults. The local loader returns datasets rather than the Level 1 template's pair of DataLoaders. The instructor's separate demo test dataset has not been supplied or tested; adapt the test dataset in `load_data()` to its actual format using the same normalization. No claim of hidden-dataset compatibility is made.

## Run and evaluate
To reproduce all experiments (overwrites existing results and checkpoints; first make a backup):
```bash
python main.py --mode all
python visualize.py
```
Individual modes:
```bash
python main.py --mode centralized
python main.py --mode clean
python main.py --mode attacked
python main.py --mode defended
python main.py --mode stress
```
Test existing checkpoints without retraining:
```bash
python main.py --mode test --checkpoint checkpoints/centralized_best.pth
python main.py --mode test --checkpoint checkpoints/clean_best.pth
python main.py --mode test --checkpoint checkpoints/defended_final.pth
```
Test mode uses standard MNIST by default. The first two experiments report best-validation checkpoint accuracy; attacked, defended and stress report final checkpoint accuracy. Resume an interrupted individual mode with `--resume checkpoints/<mode>_final.pth` and preserve the corresponding CSV and best checkpoint; optimizer and RNG state are not fully restored.

## Completed results
| Experiment | Test accuracy |
| --- | ---: |
| Centralized | 98.87% |
| Clean FedAvg | 98.35% |
| Attacked FedAvg | 98.19% |
| Defended | 98.30% |
| Stress | 98.30% |

In both defended and stress simulations, client 10 was flagged in rounds 2–10 with zero false positives. These results are for the completed simulations, not the instructor's unseen attack.

## Demo procedure
Use `python main.py --mode defended` for ten clients, ten global rounds, one local epoch per round. Client 10 is benign in round 1; the simulated attack starts in round 2 by default (`--attack-start` accepts 2–10). The TA replacement entry point in `malicious_client.py` is `generate_malicious_update(client_model_state_dict, local_dataset, device)`, matching the published template signature. It returns modified model parameters, not deltas. The call site is marked in `main.py`. Stress mode is a separate sign-reversal simulation. Preserve original checkpoints and results before live training.

## Submission contents
- Python implementation, smoke test and visualization scripts
- `requirements.txt`: pinned direct dependencies from the recorded run
- `checkpoints/`: five best and five final checkpoints
- `results/`: five CSV logs and five JSON test summaries
- `graphs/`: seven generated figures
- `docs/DigitDefender_Graduate_Final_Report.pdf`: enhanced eight-page graduate report including all seven figures (submit separately as well)
- `team_notes.json`: team contributions and limitations; confirm accuracy before submission

The report generators are retained for reproducibility of earlier report versions; do not run them to replace the supplied graduate report. Raw MNIST data and virtual environments are excluded. Review and understand the implementation and comply with the course originality policy.

## Package verification
Run `python verify_submission.py` to check checkpoint loading, result consistency and required evidence without retraining. This is not a replacement for the instructor’s test.
