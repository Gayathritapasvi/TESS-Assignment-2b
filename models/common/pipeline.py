"""Shared utilities for reproducible TESS speech, text and fusion experiments."""
from pathlib import Path
from functools import lru_cache
import argparse, json, random, re
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import librosa

LABELS = ['angry','disgust','fear','happy','neutral','ps','sad']
LABEL_NAMES = {'angry':'angry','disgust':'disgust','fear':'fear','happy':'happy','neutral':'neutral','ps':'pleasant_surprise','sad':'sad'}


def canonical_label(name):
    n=name.lower()
    for key in ('pleasant_surprised','pleasant_surprise','angry','disgust','fear','happy','neutral','sad','ps'):
        if n.endswith('_'+key): return 'ps' if key.startswith('pleasant') else key
    return None

def scan(root):
    root=Path(root)
    wavs=list(root.rglob('*.wav'))
    if not wavs: raise FileNotFoundError(f'No WAV files found under {root}. Point --data-dir at the extracted TESS folder.')
    # Some Kaggle ZIP copies contain the same corpus twice (one nested). Select
    # one complete corpus directory only, preventing duplicated observations.
    dirs={p.parent for p in wavs}
    candidates=[]
    for d in dirs:
        immediate=[x for x in d.iterdir() if x.is_dir() and canonical_label(x.name)]
        if len(immediate)>=7: candidates.append(d)
    corpus=min(candidates,key=lambda x:len(x.parts)) if candidates else root
    files=list(corpus.rglob('*.wav'))
    rows=[]; seen=set()
    for p in files:
        label=canonical_label(p.parent.name)
        if label is None: continue
        # Standard filenames end in the emotion label, e.g. OAF_youth_angry.
        parts=p.stem.split('_')
        target_parts=parts[1:]
        if target_parts and target_parts[-1].lower() in {'angry','disgust','fear','happy','neutral','sad','ps'}:
            target_parts=target_parts[:-1]
        target=' '.join(target_parts).lower()
        if not target: continue
        # Keep a single physical copy if the archive itself has nested duplicates.
        key=(parts[0].lower(),target,label)
        if key in seen: continue
        seen.add(key)
        rows.append({'path':str(p),'label':label,'text':f'say the word {target}', 'speaker_prefix':parts[0], 'utterance':target})
    df=pd.DataFrame(rows)
    if df.empty: raise ValueError('Could not parse emotion labels. Expected standard TESS emotion-named folders.')
    return df

def split_data(df, seed=42):
    # Fixed, stratified split at file level; document repeated-content limitation in README/report.
    tr, temp=train_test_split(np.arange(len(df)),test_size=.30,random_state=seed,stratify=df.label)
    va, te=train_test_split(temp,test_size=.50,random_state=seed,stratify=df.label.iloc[temp])
    return tr,va,te

@lru_cache(maxsize=4096)
def audio_feature(path):
    y,sr=librosa.load(path,sr=16000,mono=True)
    if len(y)==0: y=np.zeros(16000,dtype=np.float32)
    y,_=librosa.effects.trim(y,top_db=30)
    if len(y)<1600: y=np.pad(y,(0,1600-len(y)))
    # Fixed duration keeps batching simple; TESS utterances are short.
    y=y[:int(4*sr)]
    mel=librosa.feature.melspectrogram(y=y,sr=sr,n_fft=400,hop_length=160,n_mels=64)
    feat=librosa.power_to_db(mel,ref=np.max)
    feat=((feat+80)/80).astype(np.float32).T
    # Fixed 4-second limit, 10 ms hop: pad/crop to 400 time frames.
    feat=feat[:400]
    if feat.shape[0]<400: feat=np.pad(feat,((0,400-feat.shape[0]),(0,0)))
    return feat

def vocab_for(df):
    words=sorted({w for t in df.text for w in t.split()})
    return {'<pad>':0,'<unk>':1,**{w:i+2 for i,w in enumerate(words)}}

def encode_text(text,vocab,max_len=12):
    a=[vocab.get(w,1) for w in text.split()[:max_len]]
    return a+[0]*(max_len-len(a))

class TESSData(Dataset):
    def __init__(self,df,vocab,mode): self.df=df.reset_index(drop=True); self.vocab=vocab; self.mode=mode
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        row=self.df.iloc[i]; y=LABELS.index(row.label)
        if self.mode in ('speech','fusion'): x=audio_feature(row.path)
        if self.mode in ('text','fusion'): tok=np.array(encode_text(row.text,self.vocab),dtype=np.int64)
        if self.mode=='speech': return torch.tensor(x),torch.tensor(y)
        if self.mode=='text': return torch.tensor(tok),torch.tensor(y)
        return torch.tensor(x),torch.tensor(tok),torch.tensor(y)

class Net(nn.Module):
    def __init__(self,mode,n_vocab):
        super().__init__(); self.mode=mode
        self.a=nn.Sequential(nn.Conv1d(64,96,5,padding=2),nn.BatchNorm1d(96),nn.GELU(),nn.MaxPool1d(2),nn.Conv1d(96,128,5,padding=2),nn.GELU(),nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(128,128),nn.GELU())
        self.emb=nn.Embedding(n_vocab,64,padding_idx=0); self.gru=nn.GRU(64,96,batch_first=True,bidirectional=True)
        if mode=='speech': self.head=nn.Linear(128,7)
        elif mode=='text': self.head=nn.Linear(192,7)
        else: self.head=nn.Sequential(nn.Linear(320,128),nn.GELU(),nn.Dropout(.25),nn.Linear(128,7))
    def representations(self,*xs):
        if self.mode=='speech': return self.a(xs[0].transpose(1,2))
        t=self.gru(self.emb(xs[-1]))[0].mean(1)
        if self.mode=='text': return t
        a=self.a(xs[0].transpose(1,2)); return torch.cat((a,t),dim=1)
    def forward(self,*xs):
        return self.head(self.representations(*xs))

def batch_logits(model,batch,mode,device):
    batch=[x.to(device) for x in batch]
    if mode=='speech': return model(batch[0]),batch[-1]
    if mode=='text': return model(batch[0]),batch[-1]
    return model(batch[0],batch[1]),batch[-1]

def run_train(mode,args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    df=scan(args.data_dir); tr,va,te=split_data(df,args.seed)
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    vocab=vocab_for(df); (out/'vocab.json').write_text(json.dumps(vocab,indent=2))
    for name,idx in [('train',tr),('validation',va),('test',te)]: df.iloc[idx].to_csv(out/f'{name}_split.csv',index=False)
    device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model=Net(mode,len(vocab)).to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4); lossfn=nn.CrossEntropyLoss()
    train=DataLoader(TESSData(df.iloc[tr],vocab,mode),batch_size=args.batch_size,shuffle=True,num_workers=0)
    valid=DataLoader(TESSData(df.iloc[va],vocab,mode),batch_size=args.batch_size)
    best=1e9; patience=0
    for epoch in range(args.epochs):
        model.train(); losses=[]
        for b in train:
            opt.zero_grad(); logits,y=batch_logits(model,b,mode,device); loss=lossfn(logits,y); loss.backward(); opt.step(); losses.append(loss.item())
        model.eval(); vl=[]
        with torch.no_grad():
            for b in valid:
                z,y=batch_logits(model,b,mode,device); vl.append(lossfn(z,y).item())
        val=float(np.mean(vl)); print(f'epoch {epoch+1:02d}: train_loss={np.mean(losses):.4f} val_loss={val:.4f}')
        if val<best: best=val; patience=0; torch.save({'state_dict':model.state_dict(),'mode':mode,'n_vocab':len(vocab),'labels':LABELS},out/'model.pt')
        else:
            patience+=1
            if patience>=args.patience: break
    print(f'Saved best checkpoint: {out}/model.pt')

def run_test(mode,args):
    out=Path(args.out_dir); df=pd.read_csv(out/'test_split.csv'); vocab=json.loads((out/'vocab.json').read_text())
    # Split manifests store portable paths rooted at data/; remap them to the
    # caller's dataset directory so test.py works after moving the project.
    data_root=Path(args.data_dir)
    def resolve(p):
        p=Path(p)
        if p.is_absolute(): return str(p)
        parts=p.parts[1:] if p.parts and p.parts[0]=='data' else p.parts
        return str(data_root.joinpath(*parts))
    df['path']=df['path'].map(resolve)
    ck=torch.load(out/'model.pt',map_location='cpu',weights_only=False); device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model=Net(mode,ck['n_vocab']).to(device); model.load_state_dict(ck['state_dict']); model.eval()
    loader=DataLoader(TESSData(df,vocab,mode),batch_size=args.batch_size); ys=[]; preds=[]
    with torch.no_grad():
        for b in loader:
            z,y=batch_logits(model,b,mode,device); preds.extend(z.argmax(1).cpu().tolist()); ys.extend(y.cpu().tolist())
    report=classification_report(ys,preds,labels=list(range(7)),target_names=[LABEL_NAMES[x] for x in LABELS],output_dict=True,zero_division=0)
    print(f"{mode} test accuracy: {accuracy_score(ys,preds):.4f}")
    print(classification_report(ys,preds,labels=list(range(7)),target_names=[LABEL_NAMES[x] for x in LABELS],zero_division=0))
    (out/'test_metrics.json').write_text(json.dumps({'accuracy':accuracy_score(ys,preds),'classification_report':report},indent=2))
    pd.DataFrame(confusion_matrix(ys,preds,labels=list(range(7))),index=[LABEL_NAMES[x] for x in LABELS],columns=[LABEL_NAMES[x] for x in LABELS]).to_csv(out/'confusion_matrix.csv')
    errors=df.copy(); errors['true_label']=[LABELS[i] for i in ys]; errors['predicted_label']=[LABELS[i] for i in preds]; errors=errors[errors.true_label!=errors.predicted_label]
    errors['path']=errors['path'].map(lambda p: str(Path('data')/Path(p).relative_to(data_root)) if Path(p).is_absolute() and Path(p).is_relative_to(data_root) else str(p))
    errors.to_csv(out/'errors.csv',index=False)

def parser(mode,action):
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',default='data'); p.add_argument('--out-dir',default=f'Results/{mode}'); p.add_argument('--seed',type=int,default=42); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--epochs',type=int,default=40); p.add_argument('--patience',type=int,default=7); p.add_argument('--lr',type=float,default=1e-3); p.add_argument('--cpu',action='store_true'); return p
