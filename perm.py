import pandas as pd, numpy as np, pickle, json
from rate import *
rowsd={'bat':pd.read_pickle('bat_rows.pkl'),'bowl':pd.read_pickle('bowl_rows.pkl')}
final=pickle.load(open('final_results.pkl','rb'))
rng=np.random.default_rng(3); out={}
for side,pi,pid in [('bat','PI_bat','striker_id'),('bowl','PI_bowl','bowler_id')]:
    d=rowsd[side]; elig=final[side]['elig']
    R,W,V=row_components(d,side)
    cell=pd.factorize(d.match_id.astype(str)+'_'+d.innings.astype(str)+'_'+d.phase.astype(str))[0]
    PI=d[pi].values
    pm=pd.Series(np.arange(len(elig)),index=elig); pidx=d[pid].map(pm).fillna(-1).astype(int).values; ok=pidx>=0; K=len(elig)
    vP=V[PI>=HI].var(); vN=V[PI<=LO].var()
    def stat(P):
        hi=(P>=HI)&ok; lo=(P<=LO)&ok
        nP=np.bincount(pidx[hi],minlength=K); nN=np.bincount(pidx[lo],minlength=K)
        sP=np.bincount(pidx[hi],V[hi],K); sN=np.bincount(pidx[lo],V[lo],K)
        m=(nP>=30)&(nN>=30)
        dlt=(sP[m]/nP[m]-sN[m]/nN[m]); se=np.sqrt(vP/nP[m]+vN/nN[m]); zz=dlt/se
        return np.var(zz,ddof=1), np.mean(dlt), m.sum()
    obs=stat(PI)
    order0=np.argsort(cell,kind='stable'); null=[]
    for _ in range(300):
        orderp=np.lexsort((rng.random(len(cell)),cell)); Pp=np.empty_like(PI); Pp[order0]=PI[orderp]
        null.append(stat(Pp)[0])
    null=np.array(null)
    out[side]=dict(obs_var_z=float(obs[0]),mean_delta=float(obs[1]),n=int(obs[2]),null_mean=float(null.mean()),null_p95=float(np.percentile(null,95)),p_value=float((null>=obs[0]).mean()))
    print(side,out[side])
json.dump(out,open('perm_results.json','w'),indent=1)
