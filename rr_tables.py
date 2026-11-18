import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
b=pd.read_pickle('balls_pi.pkl'); bt=pd.read_pickle('bat_rows.pkl'); bw=pd.read_pickle('bowl_rows.pkl')
inn=b.groupby(['match_id','innings']).agg(bat_team=('batting_team_id','first'),bowl_team=('bowling_team_id','first')).reset_index()
bt=bt.merge(inn,on=['match_id','innings']); bw=bw.merge(inn,on=['match_id','innings'])
ps=b.dropna(subset=['pace_or_spin']).groupby('bowler_id').pace_or_spin.agg(lambda s:s.value_counts().index[0]); bw['ptype']=bw.bowler_id.map(ps)
bt['slot']=pd.cut(bt.batting_order,[0,2,5,12],labels=['Top (1-2)','Middle (3-5)','Lower (6+)']).astype(str)
RR=3
def stat(df,team,val,mask,scale,years):
    d=df[(df.ctype=='IPL')&(df.year.isin(years))&mask]; g=d.groupby(team)[val].mean()*scale
    rr=d[d[team]==RR]; n=len(rr); ybar=rr[val].mean(); s=rr.groupby('match_id')[val].agg(['sum','size'])
    se=np.sqrt(((s['sum']-ybar*s['size'])**2).sum())/n*scale
    return rr[val].mean()*scale,se,d[val].mean()*scale,int(g.rank(ascending=False)[RR]),n
rows=[]
specs=[('Batting','Powerplay overs',bt,'bat_team','V_bat','PI_bat',lambda d:d.phase=='Powerplay',100),
       ('Batting','Middle overs (7-15)',bt,'bat_team','V_bat','PI_bat',lambda d:d.phase=='Middle',100),
       ('Batting','Death overs (16-20)',bt,'bat_team','V_bat','PI_bat',lambda d:d.phase=='Death',100),
       ('Batting','Slots 1-2',bt,'bat_team','V_bat','PI_bat',lambda d:d.slot=='Top (1-2)',100),
       ('Batting','Slots 3-5',bt,'bat_team','V_bat','PI_bat',lambda d:d.slot=='Middle (3-5)',100),
       ('Batting','Slots 6+',bt,'bat_team','V_bat','PI_bat',lambda d:d.slot=='Lower (6+)',100),
       ('Bowling','Powerplay overs',bw,'bowl_team','V_bowl','PI_bowl',lambda d:d.phase=='Powerplay',6),
       ('Bowling','Middle overs (7-15)',bw,'bowl_team','V_bowl','PI_bowl',lambda d:d.phase=='Middle',6),
       ('Bowling','Death overs (16-20)',bw,'bowl_team','V_bowl','PI_bowl',lambda d:d.phase=='Death',6),
       ('Bowling','Pace',bw,'bowl_team','V_bowl','PI_bowl',lambda d:d.ptype=='Pace',6),
       ('Bowling','Spin',bw,'bowl_team','V_bowl','PI_bowl',lambda d:d.ptype=='Spin',6)]
for disc,lab,df,team,val,pi,f,sc in specs:
    mask=(df[pi]>=.75)&f(df)
    v,se,lg,rk,n=stat(df,team,val,mask,sc,[2025,2026]); _,_,_,rk26,n26=stat(df,team,val,mask,sc,[2026])
    rows.append(dict(disc=disc,label=lab,rr=v,se=se,league=lg,rank=rk,rank26=rk26,n=n))
T=pd.DataFrame(rows); print(T.round(2).to_string(index=False)); T.to_pickle('rr_table.pkl')
