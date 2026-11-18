import json
import sys
from pathlib import Path  

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent_dse.store import Store
from agent_dse.dims import Space
sp = Space(json.load(open('config/space.json')))
s = Store()
p = sp.baseline()
fp = sp.fingerprint(p)
print('insert :', s.put_trial({'fingerprint': fp, 'status':'ok', 'cost':1.0,
                               'latency_s':3.3e-4, 'energy_j':0.04, **p}))
print('dup    :', s.put_trial({'fingerprint': fp, 'status':'ok', 'cost':1.0, **p}))
print('has    :', s.has_point(fp))
s.update_best(p, fp, 1.0)
print('best   :', s.best())