import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('graphs'); p.mkdir(exist_ok=True)
for metric in ['val_accuracy','val_loss']:
    for f in sorted(Path('results').glob('*.csv')):
        with f.open() as h: rows=list(csv.DictReader(h))
        plt.plot([int(r['round']) for r in rows],[float(r[metric]) for r in rows],marker='o',label=f.stem)
    plt.xlabel('Epoch / global round'); plt.ylabel(metric.replace('_',' ')); plt.legend(); plt.tight_layout(); plt.savefig(p/f'{metric}.png',dpi=160); plt.close()
for f in Path('results').glob('*_test.json'):
    cm=json.loads(f.read_text())['confusion']; plt.imshow(cm,cmap='Blues'); plt.colorbar(); plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title(f.stem); plt.xticks(range(10)); plt.yticks(range(10)); plt.tight_layout(); plt.savefig(p/f'{f.stem}_confusion.png',dpi=160); plt.close()
