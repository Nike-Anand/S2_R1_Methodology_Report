import pandas as pd, numpy as np, pickle, json, warnings, os
DATA = os.environ.get('DATA_DIR', 'data').rstrip('/') + '/'
warnings.filterwarnings('ignore')
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',100)
b=pd.read_pickle('balls_pi.pkl'); bt=pd.read_pickle('bat_rows.pkl'); bw=pd.read_pickle('bowl_rows.pkl')
tm=pd.read_csv(DATA+'teams.csv').drop_duplicates('team_id').set_index('team_id').team_name
inn=b.groupby(['match_id','innings']).agg(bat_team=('batting_team_id','first'),bowl_team=('bowling_team_id','first')).reset_index()
bt=bt.merge(inn,on=['match_id','innings'],how='left'); bw=bw.merge(inn,on=['match_id','innings'],how='left')
# bowler pace/spin type (modal)
ps=b.dropna(subset=['pace_or_spin']).groupby('bowler_id').pace_or_spin.agg(lambda s:s.value_counts().index[0]); print('pace_or_spin values',b.pace_or_spin.value_counts().to_dict())
bw['ptype']=bw.bowler_id.map(ps)
RR=3
def summarise(df,team_col,val,pi,years,label,per=100,scale=1.0):
    d=df[(df.ctype=='IPL')&(df.year.isin(years))&(df[pi]>=0.75)]
    rows=[]
    for ph in ['Powerplay','Middle','Death']:
        x=d[d.phase==ph]
        # team means
        g=x.groupby(team_col)[val].agg(['mean','size'])*1
        g['mean']=g['mean']*per*scale
        rr=x[x[team_col]==RR]; n=len(rr)
        # cluster se by match for RR
        s=rr.groupby('match_id')[val].agg(['sum','size']); ybar=rr[val].mean()
        se=np.sqrt(((s['sum']-ybar*s['size'])**2).sum())/n*per*scale if n>0 else np.nan
        rows.append(dict(table=label,phase=ph,RR_value=rr[val].mean()*per*scale,RR_se=se,RR_n=n,league_mean=x[val].mean()*per*scale,RR_rank=int(g['mean'].rank(ascending=False)[RR]),teams=len(g)))
    x=d; g=x.groupby(team_col)[val].mean()*per*scale; rr=x[x[team_col]==RR]
    rows.append(dict(table=label,phase='All pressure',RR_value=rr[val].mean()*per*scale,RR_se=np.nan,RR_n=len(rr),league_mean=x[val].mean()*per*scale,RR_rank=int(g.rank(ascending=False)[RR]),teams=len(g)))
    return pd.DataFrame(rows)
out=[]
for yrs,lab in [([2025,2026],'2025-26'),([2026],'2026'),([2022,2023,2024,2025,2026],'2022-26')]:
    a=summarise(bt,'bat_team','V_bat','PI_bat',yrs,f'BAT {lab}'); a2=summarise(bw,'bowl_team','V_bowl','PI_bowl',yrs,f'BOWL {lab}',scale=0.06)
    out+= [a,a2]
R=pd.concat(out); print(R.round(2).to_string(index=False))
# bowling by type
bw2=bw[(bw.ctype=='IPL')&(bw.year.isin([2025,2026]))&(bw.PI_bowl>=.75)]
rows=[]
for pt in ['Pace','Spin']:
    x=bw2[bw2.ptype==pt]; g=x.groupby('bowl_team').V_bowl.mean()*6; rr=x[x.bowl_team==RR]
    rows.append(dict(type=pt,RR_n=len(rr),RR_value=rr.V_bowl.mean()*6,league=x.V_bowl.mean()*6,rank=int(g.rank(ascending=False)[RR])))
print(pd.DataFrame(rows).round(2))
# bowling pressure workload split RR
print('RR bowling pressure deliveries 2025-26 by type/phase'); print(pd.crosstab(bw2[bw2.bowl_team==RR].ptype,bw2[bw2.bowl_team==RR].phase))
# batting by slot
bt2=bt[(bt.ctype=='IPL')&(bt.year.isin([2025,2026]))&(bt.PI_bat>=.75)].copy(); bt2['slot']=pd.cut(bt2.batting_order,[0,2,5,12],labels=['Top (1-2)','Middle (3-5)','Lower (6+)'])
rows=[]
for sl in bt2.slot.cat.categories:
    x=bt2[bt2.slot==sl]; g=x.groupby('bat_team').V_bat.mean()*100; rr=x[x.bat_team==RR]
    rows.append(dict(slot=sl,RR_n=len(rr),RR_value=rr.V_bat.mean()*100,league=x.V_bat.mean()*100,rank=int(g.rank(ascending=False)[RR])))
print(pd.DataFrame(rows).round(1))
R.to_pickle('rr_gap.pkl')
