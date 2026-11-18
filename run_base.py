import pandas as pd, numpy as np, json, pickle
from rate import *
bt=pd.read_pickle('bat_rows.pkl'); bw=pd.read_pickle('bowl_rows.pkl'); names=pd.read_pickle('names.pkl').set_index('pid').label
out={}
for side,rows,pi in [('bat',bt,'PI_bat'),('bowl',bw,'PI_bowl')]:
    T,var=make_table(rows,side,pi)
    elig=eligible(T)
    res,parts=score(T,var,elig,return_parts=True)
    sc=bootstrap(T,var,elig,B=300)
    ranks=(-sc).argsort(1).argsort(1)+1
    res['rank']=res.score.rank(ascending=False).astype(int)
    res['boot_rank_mean']=ranks.mean(0); res['boot_rank_lo']=np.percentile(ranks,5,axis=0); res['boot_rank_hi']=np.percentile(ranks,95,axis=0)
    res['p_top10']=(ranks<=10).mean(0); res['score_lo']=np.percentile(sc,10,axis=0); res['score_hi']=np.percentile(sc,90,axis=0)
    res['name']=[names[i] for i in res.index]
    res=res.sort_values('score',ascending=False)
    out[side]=dict(res=res,parts=parts,T=T,var=var,elig=elig,boot=sc)
    print('=====',side,'eligible',len(elig)); print(json.dumps(parts,indent=0))
    cols=['name','rank','score','nP','Value_shr','Delta_shr','DOT_shr','BD_shr','CONS_shr','p_top10','boot_rank_lo','boot_rank_hi']
    print(res[cols].head(15).round(3).to_string())
pickle.dump(out,open('base_results.pkl','wb'))
