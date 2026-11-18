import pandas as pd, numpy as np, pickle
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
fin=pickle.load(open('fin_ranked.pkl','rb')); PR=pd.read_pickle('rr_profiles.pkl')
def val(side,name):
    r=fin[side]; x=r[r.name==name]; return float(x.Value_shr.iloc[0]) if len(x) else np.nan
# RR incumbents (2026 roster) that our model rates, by role
inc_bat={'Top (1-2)':['Yashasvi Jaiswal'],'Middle (3-5)':['Riyan Parag','Dhruv Jurel','Shimron Hetmyer'],'Lower (6+)':['Ravindra Jadeja','Shimron Hetmyer','Dasun Shanaka','Sam Curran']}
inc_bowl={'Pace':['Jofra Archer','Tushar Deshpande','Sandeep Sharma','Sam Curran'],'Spin':['Ravindra Jadeja','Ravi Bishnoi']}
# names in fin differ slightly? print what is found
for k,v in inc_bat.items(): print(k,[(n,round(val('bat',n),1)) for n in v])
for k,v in inc_bowl.items(): print(k,[(n,round(val('bowl',n)*0.06,2)) for n in v])
bench_bat={k:np.nanmean([val('bat',n) for n in v]) for k,v in inc_bat.items()}
bench_bowl={k:np.nanmean([val('bowl',n)*0.06 for n in v]) for k,v in inc_bowl.items()}
print('bench bat',{k:round(v,1) for k,v in bench_bat.items()},' bowl',{k:round(v,2) for k,v in bench_bowl.items()})
rows=[]
for _,p in PR.iterrows():
    if p.side=='bat':
        slot='Top (1-2)' if p.med_order<=2 else ('Middle (3-5)' if p.med_order<=5 else 'Lower (6+)')
        season_balls=(p.ipl_nP/max(p.ipl_matches,1))*14
        gain=(p.V-bench_bat[slot])/100*season_balls
        rows.append(dict(side='bat',rank=p['rank'],name=p['name'],slot=slot,bench=bench_bat[slot],val=p.V,season_workload=season_balls,gain=gain))
    else:
        grp=p.ptype if p.ptype in ('Pace','Spin') else 'Spin'
        season_del=(p.ipl_nP/max(p.ipl_matches,1))*14
        gain=(p.V*0.06-bench_bowl[grp])*season_del/6
        rows.append(dict(side='bowl',rank=p['rank'],name=p['name'],slot=grp,bench=bench_bowl[grp],val=p.V*0.06,season_workload=season_del/6,gain=gain))
G=pd.DataFrame(rows); print(G.round(2).to_string(index=False)); G.to_pickle('rr_gain.pkl')
