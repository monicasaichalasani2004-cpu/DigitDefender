# DigitDefender — CSC 8370 Data Security
Monica Sai Chalasani and Harika Chinnam | Fall 2026

## Setup
Python 3.10–3.13; install compatible PyTorch/torchvision versions using requirements.txt. CUDA GPU recommended. Tested smoke checks with the container CPU PyTorch installation; full MNIST experiments not executed because the download was blocked by DNS. Do not submit this starter archive as completed training evidence.

```bash
python -m pip install -r requirements.txt
python test_smoke.py
python main.py --mode all
python visualize.py
```

The download uses torchvision's public MNIST mirror. Course template adaptations: normalized MNIST, random equal client partitioning, separate validation subset, and exact attack interface. Instructor source: https://github.com/DongYang26/CSC8370_26FullDemo/tree/main/Templates . The attack interface is generate_malicious_update(client_model_state_dict, local_dataset, device). The returned values are model parameters, not deltas. The TA can replace malicious_client.py for defended mode. All clients behave normally in round one; attack begins round two by default.

## Individual runs and test-only evaluation
```bash
python main.py --mode centralized
python main.py --mode clean
python main.py --mode attacked
python main.py --mode defended
python main.py --mode stress
python main.py --mode test --checkpoint checkpoints/centralized_best.pth
python main.py --mode test --checkpoint checkpoints/clean_best.pth
python main.py --mode test --checkpoint checkpoints/defended_final.pth
```
Resume an interrupted mode with `python main.py --mode defended --resume checkpoints/defended_final.pth` (preserve results CSV and best checkpoint). Optimizers reset per epoch; this is not exact RNG-state resumption. For the TA's hidden data, replace the test dataset in data_loader.py with the provided dataset using the same normalization; format not yet supplied. Demo: `python main.py --mode defended` (10 rounds, one epoch, flags use client IDs 1–10). The stress mode is an independent sign-reversal simulation and not the TA replacement entry point.

## Produce report and package
After all five real runs finish, run `python visualize.py`. Edit team_notes.json with truthful individual contributions, learning and limitations, then run `python generate_report.py`. Inspect the six-page PDF and confirm all five experiments finished. Archive source, README, requirements, actual results, graphs, checkpoints and notes; exclude raw dataset and __pycache__. Submit source ZIP and PDF separately. Verify the hidden dataset interface with the instructor before demo. Review and understand generated code and check course policy on AI assistance; do not claim work you did not perform.
