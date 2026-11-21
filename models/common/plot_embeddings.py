"""Create requested 2-D t-SNE plots from held-out learned representations."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from pipeline import LABELS, LABEL_NAMES, TESSData, Net

ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(); parser.add_argument('--data-dir',default='data'); DATA=Path(parser.parse_args().data_dir)
COLORS=['#d95f02','#1b9e77','#7570b3','#e7298a','#66a61e','#e6ab02','#1f78b4']
for mode in ('speech','text','fusion'):
    out=ROOT/'Results'/mode
    if not (out/'model.pt').exists():
        print(f'Skipping {mode}: train the model first'); continue
    df=pd.read_csv(out/'test_split.csv'); vocab=json.loads((out/'vocab.json').read_text())
    df['path']=df['path'].map(lambda p: str(DATA.joinpath(*Path(p).parts[1:])) if not Path(p).is_absolute() and Path(p).parts[0]=='data' else p)
    ck=torch.load(out/'model.pt',map_location='cpu',weights_only=False)
    model=Net(mode,ck['n_vocab']); model.load_state_dict(ck['state_dict']); model.eval()
    loader=DataLoader(TESSData(df,vocab,mode),batch_size=64)
    reps=[]; labels=[]
    with torch.no_grad():
        for batch in loader:
            if mode=='speech': rep=model.representations(batch[0])
            elif mode=='text': rep=model.representations(batch[0])
            else: rep=model.representations(batch[0],batch[1])
            reps.append(rep.numpy()); labels.extend(batch[-1].numpy().tolist())
    arr=np.concatenate(reps)
    points=TSNE(n_components=2,random_state=42,perplexity=30,init='pca',learning_rate='auto').fit_transform(arr)
    fig,ax=plt.subplots(figsize=(8,6))
    for i,label in enumerate(LABELS):
        mask=np.array(labels)==i
        ax.scatter(points[mask,0],points[mask,1],s=24,alpha=.75,label=LABEL_NAMES[label],color=COLORS[i])
    ax.set_title(f'{mode.title()} representation — held-out test set (t-SNE)')
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2'); ax.legend(fontsize=8,ncol=2,frameon=False)
    fig.tight_layout(); dest=ROOT/'Results'/'plots'/f'{mode}_embedding_tsne.png'; dest.parent.mkdir(parents=True,exist_ok=True); fig.savefig(dest,dpi=180); plt.close(fig)
    print(dest)
