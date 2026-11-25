"""Verify stored evidence without downloading MNIST or retraining."""
import csv, json, math
from pathlib import Path
import torch
from model import ConvNet
root=Path(__file__).resolve().parent
for mode in ("centralized", "clean", "attacked", "defended", "stress"):
    steps=5 if mode=="centralized" else 10
    for kind in ("best", "final"):
        state=torch.load(root/"checkpoints"/f"{mode}_{kind}.pth",map_location="cpu",weights_only=True)
        model=ConvNet(); model.load_state_dict(state["model"]); model.eval()
        with torch.no_grad(): assert torch.isfinite(model(torch.zeros(1,1,28,28))).all()
        assert 1 <= state["step"] <= steps
        if kind=="final": assert state["step"]==steps
    with (root/"results"/f"{mode}.csv").open() as f: rows=list(csv.DictReader(f))
    assert [int(r["round"]) for r in rows]==list(range(1,steps+1))
    d=json.loads((root/"results"/f"{mode}_test.json").read_text()); cm=d["confusion"]
    assert len(cm)==10 and all(len(r)==10 for r in cm)
    assert sum(map(sum,cm))==10000 and math.isclose(sum(cm[i][i] for i in range(10))/10000,d["accuracy"])
    if mode in ("defended","stress"):
        assert [int(r["detected"]) for r in rows]==[0]+[1]*9
        assert all(int(r["false_positives"])==0 for r in rows)
    assert (root/"graphs"/f"{mode}_test_confusion.png").stat().st_size>0
    print(f"PASS {mode}: stored accuracy {d['accuracy']:.2%}, checkpoints and logs")
for name in ("val_accuracy.png","val_loss.png"):
    assert (root/"graphs"/name).stat().st_size>0
assert (root/"docs"/"DigitDefender_Graduate_Final_Report.pdf").stat().st_size>0
print("PASS: stored evidence checks; hidden test and full training not performed")
