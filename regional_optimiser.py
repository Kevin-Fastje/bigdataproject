'''
Inputs
------
energy  : pandas DataFrame
          columns = MultiIndex (tech, region)
          index   = hourly timestamps
demand  : pandas DataFrame
          columns = region, index = hourly timestamps

Returns
-------
pandas DataFrame
    installed capacity [GW] per region and technology
'''

import pandas as pd
from gurobipy import Model, GRB

def optimise_region_mix(energy: pd.DataFrame,
                        demand: pd.DataFrame) -> pd.DataFrame:
    assert isinstance(energy.columns, pd.MultiIndex), \
        'energy must have a MultiIndex (tech, region)'

    techs   = ['pv', 'wind_onshore', 'wind_offshore']
    regions = demand.columns
    rows    = []

    for r in regions:
        m = Model(f'region_{r}')
        m.setParam('OutputFlag', 0)        # run silent

        cf = {t: energy[t, r] for t in techs}
        d  = demand[r]

        cap = {t: m.addVar(lb=0, name=f'{t}_cap') for t in techs}

        # demand constraint for every hour
        for ts in d.index:
            m.addConstr(sum(cap[t] * cf[t][ts] for t in techs) >= d[ts])

        # objective: minimise total capacity
        m.setObjective(sum(cap[t] for t in techs), GRB.MINIMIZE)
        m.optimize()

        row = {t: cap[t].X if m.status == GRB.OPTIMAL else float('nan')
               for t in techs}
        row['region'] = r
        rows.append(row)

    return pd.DataFrame(rows)