import pandas as pd, numpy as np, pickle
pd.set_option('display.width',260); pd.set_option('display.max_columns',60)
sens=pickle.load(open('sens_results.pkl','rb')); final=pickle.load(open('final_results.pkl','rb'))
fin={}
for side in ['bat','bowl']:
    res=final[side]['res'].copy(); R=sens[side]['allranks']
    grid=[c for c in R.columns if c not in ('IPL rows only','T20I rows only')]
    res['consensus_mean_rank']=R[grid].mean(axis=1)
    res['top10_share']=(R[grid]<=10).mean(axis=1)
    res['top5_share']=(R[grid]<=5).mean(axis=1)
    res['worst_rank']=R[grid].max(axis=1); res['best_rank']=R[grid].min(axis=1)
    res['base_rank']=res['rank']
    res=res.sort_values(['consensus_mean_rank','score'],ascending=[True,False])
    res['final_rank']=np.arange(1,len(res)+1)
    fin[side]=res
    cols=['name','final_rank','base_rank','consensus_mean_rank','top10_share','worst_rank','nP','P_matches','p_top10']
    print('=====',side); print(res[cols].head(15).round(2).to_string())
    if side=='bat': c2=['name','P_balls','P_SR','N_SR','P_avg','P_bpd','P_dot%','N_dot%','P_bdry%','N_bdry%','P_outs','P_ipl_share','Value_raw','Value_shr','Delta_shr']
    else: c2=['name','P_balls','P_econ','N_econ','P_wkts','P_bpw','P_dot%','N_dot%','P_bdry%','N_bdry%','P_ipl_share','Value_raw','Value_shr']
    print(res[c2].head(12).round(1).to_string())
pickle.dump(fin,open('fin_ranked.pkl','wb'))
