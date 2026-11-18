import pandas as pd, numpy as np, pickle, sys
from scipy.stats import kendalltau, spearmanr
from rate import *
rowsd={'bat':pd.read_pickle('bat_rows.pkl'),'bowl':pd.read_pickle('bowl_rows.pkl')}
final=pickle.load(open('final_results.pkl','rb')); names=pd.read_pickle('names.pkl').set_index('pid').label
z=lambda x:(x-x.mean())/x.std()
def run(side,pi,hi=HI,lo=LO,gate=GATE,wts=WTS,mask=None,est='blend'):
    rows=rowsd[side]; T,var=make_table(rows,side,pi,hi=hi,lo=lo,mask=mask); elig=eligible(T,gate)
    if len(elig)<20: return None
    if est=='A': return score(T,var,elig,wts=wts).score
    if est=='B': return score(T,var,elig,wts=wts,hierarchical=True).score
    A=score(T,var,elig,wts=wts); B=score(T,var,elig,wts=wts,hierarchical=True)
    return 0.5*z(A.score)+0.5*z(B.score)
eq={k:1/6 for k in WTS}
variants={
 'Base (blend, hi=.75)':{},
 'High-pressure cut 0.70':dict(hi=.70),'High-pressure cut 0.80':dict(hi=.80),
 'Stakes multiplier 0':dict(pi_s='lam0'),'Stakes multiplier 0.1':dict(pi_s='lam1'),'Stakes multiplier 0.4':dict(pi_s='lam4'),
 'Leverage weight 0.4':dict(pi_s='al4'),'Leverage weight 0.8':dict(pi_s='al8'),
 'PI pooled across innings':dict(pi_s='pool'),'PI percentile within phase':dict(pi_s='ph'),
 'Equal component weights':dict(wts=eq),
 'Value-only (R+Wsave)':dict(wts={'R':.6,'Wsave':.4,'Delta':0,'DOT':0,'BD':0,'CONS':0}),
 'No clutch differential':dict(wts={'R':.3125,'Wsave':.1875,'Delta':0,'DOT':.1875,'BD':.125,'CONS':.1875}),
 'Differential-heavy (40%)':dict(wts={'R':.1875,'Wsave':.1125,'Delta':.40,'DOT':.1125,'BD':.075,'CONS':.1125}),
 'Gate 80 balls / 12 matches':dict(gate=dict(nP=80,mP=12,nN=100)),'Gate 150 balls / 20 matches':dict(gate=dict(nP=150,mP=20,nN=100)),
 'Estimator A only (pressure-only EB)':dict(est='A'),'Estimator B only (hierarchical)':dict(est='B'),
 'IPL rows only':dict(mask='ipl',gate=dict(nP=60,mP=8,nN=60)),'T20I rows only':dict(mask='t20i',gate=dict(nP=60,mP=8,nN=60)),
}
out={}
for side,pi in [('bat','PI_bat'),('bowl','PI_bowl')]:
    rows=rowsd[side]; base=final[side]['res']; b10=base.index[:10]
    tab={}; summ=[]; allr={}
    for nm,kw in variants.items():
        kw=dict(kw); ps=kw.pop('pi_s',None); mk=kw.pop('mask',None)
        col=pi if ps is None else f'{pi}_{ps}'
        mask=None if mk is None else ((rows.ctype=='IPL') if mk=='ipl' else (rows.ctype=='T20I'))
        if nm.startswith('Base'): s=base.score
        else: s=run(side,col,mask=mask,**kw)
        if s is None: continue
        r=s.rank(ascending=False)
        common=s.index.intersection(base.index)
        tau=kendalltau(s[common],base.score[common])[0]
        t10=set(s.nlargest(10).index)
        summ.append(dict(variant=nm,n_elig=len(s),kendall_tau_vs_base=tau,top10_overlap=len(t10&set(b10)),top10_overlap_within_common=len(set(s[common].nlargest(10).index)&set(b10))))
        tab[nm]=r.reindex(b10); allr[nm]=r.reindex(base.index)
        print(side,nm,len(s),round(tau,3),len(t10&set(b10)),flush=True)
    R=pd.DataFrame(tab); R.index=[names[i] for i in R.index]
    S=pd.DataFrame(summ)
    out[side]=dict(ranks=R,summary=S,allranks=pd.DataFrame(allr))
    print(S.round(3).to_string()); print(R.to_string())
# time halves (reliability): 2022-23 vs 2024-26
for side,pi in [('bat','PI_bat'),('bowl','PI_bowl')]:
    rows=rowsd[side]; g=dict(nP=50,mP=8,nN=50)
    s1=run(side,pi,mask=(rows.year<=2023),gate=g); s2=run(side,pi,mask=(rows.year>=2024),gate=g)
    c=s1.index.intersection(s2.index)
    out[side]['time_split']=dict(n=len(c),spearman=spearmanr(s1[c],s2[c])[0],kendall=kendalltau(s1[c],s2[c])[0])
    print(side,'2022-23 vs 2024-26',out[side]['time_split'])
pickle.dump(out,open('sens_results.pkl','wb'))
