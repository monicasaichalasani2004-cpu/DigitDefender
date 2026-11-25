"""Generate a six-page report from completed runs and verified team notes."""
import csv,json
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import simpleSplit
R=Path('results'); modes=['centralized','clean','attacked','defended','stress']
results={m:json.loads((R/f'{m}_test.json').read_text()) for m in modes}
rows={}
for m in modes:
    with (R/f'{m}.csv').open() as f: rows[m]=list(csv.DictReader(f))
    if len(rows[m])!=(5 if m=='centralized' else 10): raise ValueError('Incomplete training: '+m)
notes=json.loads(Path('team_notes.json').read_text())
if any(not notes.get(k,'').strip() for k in ['monica','harika','lessons','limitations']): raise ValueError('Enter factual team notes first.')
c=canvas.Canvas('DigitDefender_Final_Project_Report.pdf',pagesize=letter); c.setTitle('DigitDefender | Final project report'); W,H=letter
page=0; y=0

def start(title):
    global page,y
    if page: c.showPage()
    page+=1; c.setFillColor(HexColor('#18334A')); c.setFont('Helvetica-Bold',10); c.drawString(48,750,'GEORGIA STATE UNIVERSITY  /  CSC 8370 DATA SECURITY'); c.setStrokeColor(HexColor('#9DB0BF')); c.line(48,738,564,738); c.setFont('Helvetica-Bold',21); c.drawString(48,704,title); c.setFont('Helvetica',9); c.drawString(48,32,'DigitDefender  |  Fall 2026'); c.drawRightString(564,32,str(page)); y=674

def para(t):
    global y
    c.setFillColor(HexColor('#263747')); c.setFont('Helvetica',11)
    for line in simpleSplit(t,'Helvetica',11,516):
        if y<55: raise ValueError('Report page overflow; shorten team notes.')
        c.drawString(48,y,line); y-=16
    y-=12

def fig(path,h):
    global y
    if Path(path).exists(): c.drawImage(path,48,y-h,width=516,height=h,preserveAspectRatio=True,anchor='c'); y-=h+14
start('1  Introduction')
para('DigitDefender: Detecting Data Poisoning Attacks in Federated Learning for MNIST Classification')
para('Monica Sai Chalasani and Harika Chinnam | Final course project | November 2026')
para('Federated learning allows a server to train a shared classifier using updates from multiple local datasets. Keeping raw images local does not guarantee integrity: a participant may submit model parameters influenced by incorrect labels or manipulated updates. This project investigates that integrity problem in a controlled MNIST simulation.')
para('The research question is whether simple server-side anomaly screening can identify one malicious client and preserve digit classification accuracy over ten communication rounds. The system compares centralized training, ordinary federated averaging, poisoned averaging and a defended variant. A separate sign-reversal stress test examines a different update pattern.')
para('The threat model contains one attacker among ten clients, initially benign. The defense does not receive the attacker identity. The server observes individual parameters; this is not a differential privacy or secure aggregation protocol. The experiment is a single-machine simulation of distributed training, rather than a deployment across independent devices.')
start('2  Centralized methodology')
para('The MNIST dataset contains 60,000 training examples and 10,000 test examples. The training set is randomly divided into 55,000 training and 5,000 validation examples using a fixed seed. Images are normalized to approximately [-1, 1]. The official course loading pattern is adapted to separate validation from final testing and limit evaluation batch memory.')
para('The convolutional network maps a 1 x 28 x 28 image through two padded 3 x 3 convolution layers with 20 and 40 channels, ReLU activations and 2 x 2 max pooling. The resulting 40 x 7 x 7 feature tensor is flattened into an 80-unit hidden layer and a ten-logit output layer.')
para('Cross entropy is the negative log probability assigned to the correct class. Adam optimization uses a learning rate of 0.001 and a training batch size of 128. Centralized training runs for five epochs; the checkpoint with highest validation accuracy is retained for test-only inference.')
para('The test split is not used to select checkpoints or detector thresholds. Model checkpoints include the parameter dictionary, completed step and best validation accuracy. The code supports loading a saved checkpoint and continuing at the next epoch or round; the optimizer is reset each epoch by design.')
start('3  Federated methodology')
para('The training split is randomly partitioned into ten equal subsets of 5,500 samples. Each round begins with the same global parameters at each client. Each client performs one local epoch using Adam; optimizer state is local to that epoch. The server collects model parameters, not raw training examples.')
para('For client i, let w denote current global parameters and w_i the locally trained parameters. The update is delta_i = w_i - w. With n_i local examples, weighted FedAvg computes w_next = w + sum(n_i delta_i)/sum(n_i). Equal partition sizes make the weights equal in this experiment.')
para('The clean, attacked and defended runs use the same network, initial seed, client partitions, learning rate, batch size and ten-round budget. The controlled attacker acts normally in round one, then trains on its local data with class 3 relabeled as class 8. The function in malicious_client.py matches the instructor interface and returns model parameters rather than a raw gradient.')
para('The stress run uses a sign-reversed, scaled local update instead of label flipping. It is a separate local experiment; the defended mode is the one intended for the instructor to replace the malicious-client function during the live demo.')
start('4  Robust aggregation')
para('For each round, the server flattens parameter differences into vectors, checks whether values are finite and computes a coordinate-wise median reference. Update magnitude is measured using the Euclidean norm. Directional agreement is measured by cosine similarity to the reference vector.')
para('The median absolute deviation (MAD) is the median of absolute differences from the scalar median. The detector flags unusually large norms or unusually low cosine similarity using a threshold of 3.5 times 1.4826 MAD, with minimum margins of 35% of the median norm and 0.12 cosine units, respectively. These are engineering defaults rather than validated optimal thresholds.')
para('Flagged updates are excluded. Accepted vectors are clipped to 1.5 times the median norm before weighted averaging. When fewer than six finite, unflagged updates remain, the coordinate-wise median is used as a fallback. The detector is not guaranteed to identify subtle attacks; benign data heterogeneity may also generate outliers.')
para('Each round records validation accuracy, loss, flagged client identifiers, whether the active attacker was detected, and benign false-positive counts. The final round checkpoint is used to evaluate defended performance; best-validation checkpoints are retained for clean and centralized inference.')
start('5  Experimental evaluation')
for m in modes:
    detected=sum(int(r['detected']) for r in rows[m]); fp=sum(int(r['false_positives']) for r in rows[m]); para(f'{m.title()}: test accuracy {results[m]["accuracy"]:.2%}; test loss {results[m]["loss"]:.4f}; detected attacker rounds {detected}; benign flag events {fp}.')
para(f'Defended minus attacked test accuracy: {(results["defended"]["accuracy"]-results["attacked"]["accuracy"])*100:+.2f} percentage points. A positive difference supports mitigation for this trial; a negative difference indicates that the defense did not improve this comparison. These measurements are from the public MNIST test set, not the instructor hidden test set.')
fig('graphs/val_accuracy.png',205)
start('6  Learning outcomes and limitations')
para('Monica Sai Chalasani: '+notes['monica']); para('Harika Chinnam: '+notes['harika']); para('Learning outcomes: '+notes['lessons']); para('Limitations and uncompleted tasks: '+notes['limitations'])
para('The live demonstration should run the defended mode from scratch for ten rounds, display client flags and load the final checkpoint for evaluation. Hidden attack generalization is not established by the public experiments.')
para('References: McMahan et al., Communication-Efficient Learning of Deep Networks from Decentralized Data (AISTATS, 2017). LeCun et al., Gradient-Based Learning Applied to Document Recognition (Proceedings of the IEEE, 1998). CSC 8370 Course Project 2026 guidelines and instructor templates, DongYang26/CSC8370_26FullDemo, Templates directory.')
c.save(); print('Created six-page project report.')
