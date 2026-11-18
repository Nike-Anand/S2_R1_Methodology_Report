import pandas as pd, numpy as np, pickle, json
from rate import *
rowsd={'bat':pd.read_pickle('bat_rows.pkl'),'bowl':pd.read_pickle('bowl_rows.pkl')}
base=pickle.load(open('base_results.pkl','rb'))
def wcorr(x,y,w):
    mx,my=np.average(x,weights=w),np.average(y,weights=w)
    return np.sum(w*(x-mx)*(y-my))/np.sqrt(np.sum(w*(x-mx)**2)*np.sum(w*(y-my)**2))
out={}
rng=np.random.default_rng(5)
for side,pi in [('bat','PI_bat'),('bowl','PI_bowl')]:
    T,var=make_table(rowsd[side],side,pi); elig=base[side]['elig']
    mids=np.unique(T.mid.values); M=len(mids)
    rec=[]
    for s in range(100):
        wA=np.zeros(M); wA[rng.permutation(M)[:M//2]]=1; wB=1-wA
        A0=score(T,var,elig,w_match=wA); A1=score(T,var,elig,w_match=wA,hierarchical=True); B0=score(T,var,elig,w_match=wB)
        ok=(A0.nP>=40)&(B0.nP>=40)
        w=(1/(1/A0.nP+1/B0.nP))[ok].values
        # normal-only baseline from half A: raw V in normal situations
        yB=B0.Value_raw[ok].values; cB=B0.score_unshrunk[ok].values
        row=dict(split=s,n=int(ok.sum()))
        for nm,pa in [('raw_pressure',A0.Value_raw),('A_shrunk_pressure_only',A0.Value_shr),('B_hierarchical',A1.Value_shr)]:
            row[f'val_{nm}']=wcorr(pa[ok].values,yB,w)
        # normal-situation value alone (what a 'just a good player' model predicts)
        yN_A=pd.Series(np.nan,index=elig)
        Tm=T.copy()
        row['val_normal_only']=np.nan
        for nm,pa in [('raw_pressure',A0.score_unshrunk),('A_shrunk_pressure_only',A0.score),('B_hierarchical',A1.score)]:
            row[f'comp_{nm}']=wcorr(pa[ok].values,cB,w)
        rec.append(row)
    r=pd.DataFrame(rec); out[side]=r
    print('=====',side,'mean over',len(r),'random match-splits; players per split',r.n.mean().round(0))
    print(r.drop(columns=['split','n','val_normal_only']).mean().round(3).to_string())
    print('B beats A in',(r.val_B_hierarchical>r.val_A_shrunk_pressure_only).mean(),'of splits (value); comp:',(r.comp_B_hierarchical>r.comp_A_shrunk_pressure_only).mean())
pickle.dump(out,open('pred_validation.pkl','wb'))
