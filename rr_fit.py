import pandas as pd, numpy as np, pickle, warnings, os
DATA = os.environ.get('DATA_DIR', 'data').rstrip('/') + '/'
warnings.filterwarnings('ignore')
pd.set_option('display.width',260); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',100)
b=pd.read_pickle('balls_pi.pkl'); bt=pd.read_pickle('bat_rows.pkl'); bw=pd.read_pickle('bowl_rows.pkl'); fin=pickle.load(open('fin_ranked.pkl','rb'))
m=pd.read_pickle('matches_clean.pkl'); mp=pd.read_csv(DATA+'competition_team_player_mapping.csv'); c=pd.read_csv(DATA+'competitions.csv')
# RR 2026 results & eliminator
m26=m[(m.ctype=='IPL')&(m.year==2026)]
rrm=m26[(m26.home_team_id==3)|(m26.away_team_id==3)] if 'home_team_id' in m26 else m26
print(m26.columns.tolist()[:30])
print(rrm[['match_date','stage','home_team','away_team','result']].tail(6).to_string())
ps=b.dropna(subset=['pace_or_spin']).groupby('bowler_id').pace_or_spin.agg(lambda s:s.value_counts().index[0])
bty=b.dropna(subset=['bowling_type']).groupby('bowler_id').bowling_type.agg(lambda s:s.value_counts().index[0])
bhand=b.dropna(subset=['batting_type']).groupby('striker_id').batting_type.agg(lambda s:s.value_counts().index[0])
ipl26=m[(m.ctype=='IPL')&(m.year==2026)].comp_id.unique(); ipl_team=mp[mp.comp_id.isin(ipl26)].set_index('player_id').team_name
nat=mp[mp.comp_id.isin(c[c.match_format=='T20I'].comp_id)].groupby('player_id').team_name.agg(lambda s:s.value_counts().index[0])
prof={}
for side in ['bat','bowl']:
    rows=bt if side=='bat' else bw; pid='striker_id' if side=='bat' else 'bowler_id'; pi='PI_bat' if side=='bat' else 'PI_bowl'
    for p,x in fin[side].head(10).iterrows():
        r=rows[rows[pid]==p]; P=r[r[pi]>=.75]; ipl=r[r.ctype=='IPL']; Pi=P[P.ctype=='IPL']
        seasons=ipl.year.nunique(); mi=ipl.match_id.nunique()
        d=dict(side=side,rank=int(x.final_rank),name=x['name'],pid=p,nat=nat.get(p,'-'),ipl26=ipl_team.get(p,'none'),
               ph_pp=(P.phase=='Powerplay').mean(),ph_mid=(P.phase=='Middle').mean(),ph_dth=(P.phase=='Death').mean(),
               nP=len(P),ipl_nP=len(Pi),ipl_matches=mi,ipl_seasons=seasons,
               ipl_P_per_match=(len(Pi)/max(Pi.match_id.nunique(),1)), V=x.Value_shr)
        if side=='bat': d['med_order']=r.batting_order.median(); d['hand']=bhand.get(p,'-')
        else: d['ptype']=ps.get(p,'-'); d['btype']=bty.get(p,'-')
        prof[(side,p)]=d
PR=pd.DataFrame(prof.values()); print(PR.round(2).to_string())
PR.to_pickle('rr_profiles.pkl')
# RR 2026 roster standing
ids=mp[(mp.comp_id.isin(ipl26))&(mp.team_id==3)].set_index('player_id').player_name
rows=[]
for p,nm_ in ids.items():
    r={'pid':p,'name':nm_}
    x=bt[(bt.striker_id==p)&(bt.PI_bat>=.75)]; y=bw[(bw.bowler_id==p)&(bw.PI_bowl>=.75)]
    r['bat_nP']=len(x); r['bat_raw']=100*x.V_bat.mean() if len(x) else np.nan; r['bat_shr']=r['bat_raw']*len(x)/(len(x)+560) if len(x) else np.nan
    r['bowl_nP']=len(y); r['bowl_raw']=6*y.V_bowl.mean() if len(y) else np.nan; r['bowl_shr']=r['bowl_raw']*len(y)/(len(y)+500) if len(y) else np.nan
    r['bat_rank']=fin['bat'].final_rank.get(p,np.nan); r['bowl_rank']=fin['bowl'].final_rank.get(p,np.nan)
    rows.append(r)
RRR=pd.DataFrame(rows).sort_values('bat_nP',ascending=False); print(RRR.round(2).to_string(index=False)); RRR.to_pickle('rr_roster.pkl')
