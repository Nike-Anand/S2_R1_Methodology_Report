import pandas as pd, numpy as np, pickle, json
from rate import *
rowsd={'bat':pd.read_pickle('bat_rows.pkl'),'bowl':pd.read_pickle('bowl_rows.pkl')}
names=pd.read_pickle('names.pkl').set_index('pid').label
# player metadata: primary team/ctype mix
final={}
def raw_table(rows,side,pi,elig):
    pid='striker_id' if side=='bat' else 'bowler_id'
    P=rows[rows[pi]>=HI]; N=rows[rows[pi]<=LO]; out=pd.DataFrame(index=elig)
    for tag,d in [('P',P),('N',N)]:
        g=d[d[pid].isin(elig)].groupby(pid)
        if side=='bat':
            a=g.agg(n=('r_bat','size'),runs=('r_bat','sum'),outs=('out_bat','sum'),dt=('dot_bat','mean'),bdm=('bd','mean'))
            out[f'{tag}_balls']=a.n; out[f'{tag}_SR']=100*a.runs/a.n
            out[f'{tag}_avg']=a.runs/a.outs.replace(0,np.nan); out[f'{tag}_bpd']=a.n/a.outs.replace(0,np.nan)
            out[f'{tag}_dot%']=100*a['dt']; out[f'{tag}_bdry%']=100*a['bdm']; out[f'{tag}_outs']=a.outs
        else:
            dd=d[d[pid].isin(elig)].copy(); dd['lg']=dd.legal.astype(int)
            g=dd.groupby(pid)
            a=g.agg(n=('lg','sum'),dl=('runs_bowl','size'),runs=('runs_bowl','sum'),wk=('wkt_bowl','sum'),dt=('dot_bowl','sum'),bds=('bd','sum'))
            out[f'{tag}_balls']=a.n; out[f'{tag}_econ']=6*a.runs/a.n; out[f'{tag}_wkts']=a.wk
            out[f'{tag}_bpw']=a.n/a.wk.replace(0,np.nan); out[f'{tag}_dot%']=100*a['dt']/a.n; out[f'{tag}_bdry%']=100*a['bds']/a.dl
    m=P[P[pid].isin(elig)].groupby(pid).match_id.nunique(); out['P_matches']=m
    out['P_ipl_share']=P[P[pid].isin(elig)].assign(i=lambda d:(d.ctype=='IPL')).groupby(pid).i.mean()
    return out
for side,pi in [('bat','PI_bat'),('bowl','PI_bowl')]:
    rows=rowsd[side]; T,var=make_table(rows,side,pi); elig=eligible(T)
    def blend(w=None):
        A=score(T,var,elig,w_match=w); Bh=score(T,var,elig,w_match=w,hierarchical=True)
        z=lambda x:(x-x.mean())/x.std()
        sc=0.5*z(A.score)+0.5*z(Bh.score)
        return A,Bh,sc
    A,Bh,sc0=blend()
    _,partsA=score(T,var,elig,return_parts=True); _,partsB=score(T,var,elig,return_parts=True,hierarchical=True)
    res=pd.DataFrame(index=elig); res['score']=sc0; res['score_A']=A.score; res['score_B']=Bh.score
    for c in ['nP','nN','ninn']: res[c]=A[c]
    for k in COMPS:
        res[f'{k}_A']=A[f'{k}_z']; res[f'{k}_B']=Bh[f'{k}_z']; res[f'{k}_raw']=A[f'{k}_raw']; res[f'{k}_se']=A[f'{k}_se']
    res['Value_raw']=A.Value_raw; res['Value_A']=A.Value_shr; res['Value_B']=Bh.Value_shr; res['Value_shr']=0.5*A.Value_shr+0.5*Bh.Value_shr
    res['R_shr']=0.5*A.R_shr+0.5*Bh.R_shr; res['Wsave_shr']=0.5*A.Wsave_shr+0.5*Bh.Wsave_shr
    res['Delta_shr']=0.5*A.Delta_shr+0.5*Bh.Delta_shr; res['DOT_shr']=0.5*A.DOT_shr+0.5*Bh.DOT_shr; res['BD_shr']=0.5*A.BD_shr+0.5*Bh.BD_shr; res['CONS_shr']=A.CONS_shr
    parts={'A':partsA,'B':partsB}
    rng=np.random.default_rng(11); mids=np.unique(T.mid.values); M=len(mids); Bn=300
    sc=np.zeros((Bn,len(elig))); va=np.zeros_like(sc); de=np.zeros_like(sc)
    for b in range(Bn):
        w=np.bincount(rng.integers(0,M,M),minlength=M).astype(float)
        A_,B_,s_=blend(w); sc[b]=s_.values; va[b]=0.5*A_.Value_shr.values+0.5*B_.Value_shr.values; de[b]=0.5*A_.Delta_shr.values+0.5*B_.Delta_shr.values
    ranks=(-sc).argsort(1).argsort(1)+1
    res['rank']=res.score.rank(ascending=False).astype(int)
    res['boot_rank_lo']=np.percentile(ranks,5,axis=0); res['boot_rank_hi']=np.percentile(ranks,95,axis=0); res['boot_rank_med']=np.median(ranks,axis=0)
    res['p_top10']=(ranks<=10).mean(0); res['p_top20']=(ranks<=20).mean(0)
    for nm,arr in [('score',sc),('Value',va),('Delta',de)]:
        res[f'{nm}_b10']=np.percentile(arr,10,axis=0); res[f'{nm}_b90']=np.percentile(arr,90,axis=0)
    res['name']=[names[i] for i in res.index]
    rt=raw_table(rows,side,pi,elig)
    res=res.join(rt).sort_values('score',ascending=False)
    final[side]=dict(res=res,parts=parts,var=var,elig=elig,boot_scores=sc,ranks=ranks)
    print('=====',side,'eligible',len(elig))
    print({m_:{k:round(v['tau'],3) for k,v in p_.items()} for m_,p_ in parts.items()})
    cols=['name','rank','score','nP','P_matches','Value_shr','Value_raw','Delta_shr','p_top10','boot_rank_lo','boot_rank_hi']
    print(res[cols].head(14).round(2).to_string())
pickle.dump(final,open('final_results.pkl','wb'))
