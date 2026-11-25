"""Generate a eight-page report from completed runs and verified team notes."""
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
def sub(t):
    global y
    c.setFillColor(HexColor('#18334A')); c.setFont('Helvetica-Bold',12)
    c.drawString(48,y,t); y-=22

def equation(t):
    global y
    c.setFillColor(HexColor('#EEF3F7')); c.roundRect(48,y-22,516,35,5,fill=1,stroke=0)
    c.setFillColor(HexColor('#18334A')); c.setFont('Helvetica-Bold',10)
    c.drawString(60,y-8,t); y-=47

start('1  Introduction')
para('DigitDefender: Detecting Data Poisoning Attacks in Federated Learning for MNIST Classification')
para('Monica Sai Chalasani and Harika Chinnam | Final course project | November 2026')
sub('Abstract and research motivation')
para('Federated learning allows a server to train a shared classifier using updates from multiple local datasets. Keeping raw images local does not guarantee integrity: a participant may submit model parameters influenced by incorrect labels or manipulated updates. This project investigates that integrity problem in a controlled MNIST simulation.')
para('The research question is whether simple server-side anomaly screening can identify one malicious client and preserve digit classification accuracy over ten communication rounds. The system compares centralized training, ordinary federated averaging, poisoned averaging and a defended variant. A separate sign-reversal stress test examines a different update pattern.')
sub('Scope and research question')
para('The threat model contains one attacker among ten clients, initially benign. The defense does not receive the attacker identity. The server observes individual parameters; this is not a differential privacy or secure aggregation protocol. The experiment is a single-machine simulation of distributed training, rather than a deployment across independent devices.')
sub('Research objectives')
para('Establish a reproducible centralized baseline; compare clean and poisoned FedAvg; detect abnormal client updates without assuming their identity; and quantify accuracy, detection and false-positive behavior under a fixed training budget.')
start('2  System and security model')
sub('Architecture and data flow')
para('A simulated central server broadcasts the current CNN to ten logical clients. Each client retains its own partition and returns trained model parameters. The server converts parameters into update vectors, optionally screens them, aggregates accepted updates and broadcasts the resulting global model. Validation data remains outside local training.')
equation('Global CNN  ->  10 local clients  ->  update screening  ->  aggregation')
sub('Threat assumptions and security objectives')
para('The attacker controls one client and can manipulate its local labels or submitted parameters. It behaves honestly in round one. The server and remaining clients are assumed honest; the server can inspect individual updates. The objective is model integrity and availability of useful classification, not confidentiality of updates.')
para('Label flipping changes the learning signal for one digit, while sign reversal changes the direction and magnitude of the submitted update. An ordinary weighted mean can be influenced by either. Detection relies on the majority of client updates remaining representative of legitimate training.')
sub('Evaluation boundaries')
para('This experiment uses IID-style random partitions on one computer. It does not model network delays, encrypted transport, differential privacy, secure aggregation, colluding attackers or heterogeneous devices. No claim is made that keeping images local alone prevents inference attacks.')
start('3  Centralized methodology')
para('The MNIST dataset contains 60,000 training examples and 10,000 test examples. The training set is randomly divided into 55,000 training and 5,000 validation examples using a fixed seed. Images are normalized to approximately [-1, 1]. The official course loading pattern is adapted to separate validation from final testing and limit evaluation batch memory.')
para('The convolutional network maps a 1 x 28 x 28 image through two padded 3 x 3 convolution layers with 20 and 40 channels, ReLU activations and 2 x 2 max pooling. The resulting 40 x 7 x 7 feature tensor is flattened into an 80-unit hidden layer and a ten-logit output layer.')
sub('Optimization and objective')
equation('L = -(1/B) sum_j log softmax(z_j)[y_j]')
para('Cross entropy is the negative log probability assigned to the correct class. Adam optimization uses a learning rate of 0.001 and a training batch size of 128. Centralized training runs for five epochs; the checkpoint with highest validation accuracy is retained for test-only inference.')
para('The test split is not used to select checkpoints or detector thresholds. Model checkpoints include the parameter dictionary, completed step and best validation accuracy. The code supports loading a saved checkpoint and continuing at the next epoch or round; the optimizer is reset each epoch by design.')
start('4  Federated methodology')
para('The training split is randomly partitioned into ten equal subsets of 5,500 samples. Each round begins with the same global parameters at each client. Each client performs one local epoch using Adam; optimizer state is local to that epoch. The server collects model parameters, not raw training examples.')
para('For client i, let w denote current global parameters and w_i the locally trained parameters. The update is delta_i = w_i - w. With n_i local examples, weighted FedAvg computes w_next = w + sum(n_i delta_i)/sum(n_i). Equal partition sizes make the weights equal in this experiment.')
equation('w(t+1) = w(t) + sum_i n_i delta_i / sum_i n_i')
sub('Controlled attack scenarios')
para('The clean, attacked and defended runs use the same network, initial seed, client partitions, learning rate, batch size and ten-round budget. The controlled attacker acts normally in round one, then trains on its local data with class 3 relabeled as class 8. The function in malicious_client.py matches the instructor interface and returns model parameters rather than a raw gradient.')
para('The stress run uses a sign-reversed, scaled local update instead of label flipping. It is a separate local experiment; the defended mode is the one intended for the instructor to replace the malicious-client function during the live demo.')
start('5  Robust aggregation')
para('For each round, the server flattens parameter differences into vectors, checks whether values are finite and computes a coordinate-wise median reference. Update magnitude is measured using the Euclidean norm. Directional agreement is measured by cosine similarity to the reference vector.')
para('The median absolute deviation (MAD) is the median of absolute differences from the scalar median. The detector flags unusually large norms or unusually low cosine similarity using a threshold of 3.5 times 1.4826 MAD, with minimum margins of 35% of the median norm and 0.12 cosine units, respectively. These are engineering defaults rather than validated optimal thresholds.')
equation('MAD(x) = median_i |x_i - median(x)|')
para('Flagged updates are excluded. Accepted vectors are clipped to 1.5 times the median norm before weighted averaging. When fewer than six finite, unflagged updates remain, the coordinate-wise median is used as a fallback. The detector is not guaranteed to identify subtle attacks; benign data heterogeneity may also generate outliers.')
para('Each round records validation accuracy, loss, flagged client identifiers, whether the active attacker was detected, and benign false-positive counts. The final round checkpoint is used to evaluate defended performance; best-validation checkpoints are retained for clean and centralized inference.')
start('6  Experimental evaluation')
sub('Environment and experimental protocol')
para('The recorded run used an Apple Silicon MacBook, CPU execution, Python 3.9, PyTorch 2.8.0 and torchvision 0.23.0. Each federated experiment used ten rounds, one local epoch, 5,500 examples per client and fixed seeds. The independent public MNIST test set contains 10,000 images.')
sub('Measured test performance')
# Compact table, using measured values from saved files.
c.setFillColor(HexColor('#18334A')); c.rect(48,y-12,516,23,fill=1,stroke=0)
c.setFillColor(HexColor('#FFFFFF')); c.setFont('Helvetica-Bold',10)
for x,t in zip([58,226,322,403,484],['Experiment','Accuracy','Loss','Detections','False flags']): c.drawString(x,y-4,t)
y-=28
for i,m in enumerate(modes):
    if i%2==0:
        c.setFillColor(HexColor('#EEF3F7')); c.rect(48,y-9,516,23,fill=1,stroke=0)
    c.setFillColor(HexColor('#263747')); c.setFont('Helvetica',10)
    vals=[m.title(),f'{results[m]["accuracy"]:.2%}',f'{results[m]["loss"]:.4f}',str(sum(int(r['detected']) for r in rows[m])),str(sum(int(r['false_positives']) for r in rows[m]))]
    for x,t in zip([58,226,322,403,484],vals): c.drawString(x,y,t)
    y-=24
y-=12
para('Detection counts are attack-round flags, not unique attackers. The attacked baseline has no enabled detector; zero flags there do not establish safety.')
for m in ['defended','stress']:
    hits=sum(int(r['detected']) for r in rows[m][1:]); fp=sum(int(r['false_positives']) for r in rows[m]); para(f'{m.title()}: {hits}/9 active attack rounds detected; {fp} benign flag events. The denominator assumes the recorded default attack start at round 2.')
clean=results['clean']['accuracy']; attacked=results['attacked']['accuracy']; defended=results['defended']['accuracy']
para(f'The defended model changes test accuracy by {(defended-attacked)*100:+.2f} percentage points relative to poisoned FedAvg and {(defended-clean)*100:+.2f} points relative to clean FedAvg. These are single-run descriptive comparisons, not statistical significance claims.')
start('7  Learning curves and error analysis')
fig('graphs/val_accuracy.png',190)
para('Figure 1. Validation accuracy across training. Centralized epochs and federated rounds are distinct units; compare curves with this distinction in mind.')
fig('graphs/val_loss.png',190)
para('Figure 2. Validation cross-entropy. Lower loss generally reflects stronger confidence in correct labels, but does not by itself establish attack resistance.')
para('The per-model confusion matrices in graphs/ provide class-level evidence. A useful follow-up is the 3-to-8 error count, rather than relying solely on overall accuracy.')
start('8  Outcomes, limitations and references')
para('Monica Sai Chalasani: '+notes['monica']); para('Harika Chinnam: '+notes['harika']); para('Learning outcomes: '+notes['lessons']); para('Limitations and uncompleted tasks: '+notes['limitations'])
sub('Interpretation and future work')
para('The experiments support the feasibility of update screening under the tested single-attacker conditions. Repeated seeds, class-specific error analysis, non-IID client partitions and unseen attacks would better establish generalization. The two defended runs may converge to identical parameters because the same malicious client is excluded in both.')
para('The live demonstration should run the defended mode from scratch for ten rounds, display client flags and load the final checkpoint for evaluation. Hidden attack generalization is not established by the public experiments.')
para('References: McMahan et al., Communication-Efficient Learning of Deep Networks from Decentralized Data (AISTATS, 2017). LeCun et al., Gradient-Based Learning Applied to Document Recognition (Proceedings of the IEEE, 1998). CSC 8370 Course Project 2026 guidelines and instructor templates, DongYang26/CSC8370_26FullDemo, Templates directory.')
c.save(); print('Created eight-page project report.')
