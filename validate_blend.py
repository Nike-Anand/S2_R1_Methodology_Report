import pandas as pd, numpy as np, pickle
from rate import *
rowsd={'bat':pd.read_pickle('bat_rows.pkl'),'bowl':pd.read_pickle('bowl_rows.pkl')}
base=pickle.load(open('base_results.pkl','rb'))
def wcorr(x,y,w):
    mx,my=np.average(x,weights=w),np.average(y,weights=w)
    return np.sum(w*(x-mx)*(y-my))/np.sqrt(np.sum(w*(x-mx)**2)*np.sum(w*(y-my)**2))
rng=np.random.default_rng(5); out={}
for side,pi in [('bat','PI_bat'),('bowl','PI_bowl')]:
    T,var=make_table(rowsd[side],side,pi); elig=base[side]['elig']
    mids=np.unique(T.mid.values); M=len(mids); rec=[]
    for s in range(100):
        wA=np.zeros(M); wA[rng.permutation(M)[:M//2]]=1; wB=1-wA
        A0=score(T,var,elig,w_match=wA); A1=score(T,var,elig,w_match=wA,hierarchical=True); B0=score(T,var,elig,w_match=wB)
        ok=(A0.nP>=40)&(B0.nP>=40); w=(1/(1/A0.nP+1/B0.nP))[ok].values
        zA=(A0.score-A0.score.mean())/A0.score.std(); zB=(A1.score-A1.score.mean())/A1.score.std()
        vA=(A0.Value_shr-A0.Value_shr.mean())/A0.Value_shr.std(); vB=(A1.Value_shr-A1.Value_shr.mean())/A1.Value_shr.std()
        row={}
        for lam in [0,.25,.5,.75,1.0]:
            row[f'comp_{lam}']=wcorr(((1-lam)*zA+lam*zB)[ok].values,B0.score_unshrunk[ok].values,w)
            row[f'val_{lam}']=wcorr(((1-lam)*vA+lam*vB)[ok].values,B0.Value_raw[ok].values,w)
        rec.append(row)
    r=pd.DataFrame(rec); print('=====',side); print(r.mean().round(3).to_string()); out[side]=r
pickle.dump(out,open('blend_validation.pkl','wb'))
