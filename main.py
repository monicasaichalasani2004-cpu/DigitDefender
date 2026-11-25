import argparse, copy, json, csv, random
from pathlib import Path
import torch
from model import ConvNet
from data_loader import load_data,partition_dataset,loader
from defense import aggregate
from malicious_client import generate_malicious_update

OUT=Path('results'); CK=Path('checkpoints')
def seed(n): random.seed(n); torch.manual_seed(n); torch.cuda.manual_seed_all(n)
def train(model,data,device,seed_value):
    model.train(); opt=torch.optim.Adam(model.parameters(),lr=.001); total=0; count=0
    for x,y in loader(data,True,seed_value):
        x,y=x.to(device),y.to(device); opt.zero_grad(); loss=torch.nn.functional.cross_entropy(model(x),y); loss.backward(); opt.step(); total+=loss.item()*len(y); count+=len(y)
    return total/count
@torch.no_grad()
def evaluate(model,data,device):
    model.eval(); correct=0; total=0; loss=0; cm=torch.zeros(10,10,dtype=torch.long)
    for x,y in loader(data):
        x,y=x.to(device),y.to(device); z=model(x); p=z.argmax(1); correct+=(p==y).sum().item(); total+=len(y); loss+=torch.nn.functional.cross_entropy(z,y,reduction='sum').item(); cm+=torch.bincount((10*y+p).cpu(),minlength=100).reshape(10,10)
    return {'accuracy':correct/total,'loss':loss/total,'confusion':cm.tolist()}
def save(model,path,step,best): torch.save({'model':model.state_dict(),'step':step,'best':best},path)
def run(mode,train_data,val,test,device,resume=None,attack_start=2):
    seed(8370); model=ConvNet().to(device); best=-1.; start=1; rows=[]
    history=OUT/f'{mode}.csv'
    if resume:
        state=torch.load(resume,map_location=device,weights_only=True); model.load_state_dict(state['model']); start=state['step']+1; best=state['best']
        if history.exists():
            with history.open() as f: rows=[r for r in csv.DictReader(f) if int(r['round'])<start]
    clients=partition_dataset(train_data); steps=5 if mode=='centralized' else 10
    for r in range(start,steps+1):
        flags=[]; losses=[]
        if mode=='centralized': losses.append(train(model,train_data,device,100*r))
        else:
            base=copy.deepcopy(model.state_dict()); states=[]
            for i,ds in enumerate(clients):
                seed(10000*r+i)
                if mode in ('attacked','defended','stress') and i==9 and r>=attack_start:
                    # === MALICIOUS CLIENT LOGIC HERE ===
                    if mode=='stress':
                        local=copy.deepcopy(model); losses.append(train(local,ds,device,r*100+i)); states.append({k:base[k]-3*(v-base[k]) for k,v in local.state_dict().items()})
                    else: states.append(generate_malicious_update(copy.deepcopy(base),ds,device))
                else:
                    local=copy.deepcopy(model); losses.append(train(local,ds,device,r*100+i)); states.append(copy.deepcopy(local.state_dict()))
            state,flags=aggregate(base,states,[len(d) for d in clients],mode in ('defended','stress')); model.load_state_dict(state)
        metrics=evaluate(model,val,device); acc=metrics['accuracy']; active=mode in ('attacked','defended','stress') and r>=attack_start
        row={'round':r,'val_accuracy':acc,'val_loss':metrics['loss'],'mean_local_loss':sum(losses)/len(losses),'flags':';'.join(str(i+1) for i in flags),'detected':int(active and 9 in flags),'false_positives':sum(not(active and i==9) for i in flags)}; rows.append(row); print(mode,row,flush=True)
        if acc>best: best=acc; save(model,CK/f'{mode}_best.pth',r,best)
        save(model,CK/f'{mode}_final.pth',r,best)
        with history.open('w') as f: w=csv.DictWriter(f,fieldnames=list(row)); w.writeheader(); w.writerows(rows)
    if mode in ('centralized','clean'): model.load_state_dict(torch.load(CK/f'{mode}_best.pth',map_location=device,weights_only=True)['model'])
    result=evaluate(model,test,device); (OUT/f'{mode}_test.json').write_text(json.dumps(result,indent=2)); print(mode,'test accuracy',result['accuracy'],flush=True)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['all','centralized','clean','attacked','defended','stress','test'],default='all'); p.add_argument('--checkpoint'); p.add_argument('--resume'); p.add_argument('--attack-start',type=int,default=2,choices=range(2,11)); args=p.parse_args()
    OUT.mkdir(exist_ok=True); CK.mkdir(exist_ok=True); torch.set_num_threads(min(4,torch.get_num_threads())); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('Device:',device)
    tr,val,te=load_data()
    if args.mode=='test':
        if not args.checkpoint: p.error('--checkpoint required')
        model=ConvNet().to(device); model.load_state_dict(torch.load(args.checkpoint,map_location=device,weights_only=True)['model']); print(evaluate(model,te,device)); return
    if args.resume and args.mode=='all': p.error('Resume one mode at a time')
    for mode in (['centralized','clean','attacked','defended','stress'] if args.mode=='all' else [args.mode]): run(mode,tr,val,te,device,args.resume,args.attack_start)
if __name__=='__main__': main()
