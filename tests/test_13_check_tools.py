import json
import sys
from pathlib import Path  

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent_dse.store import Store
from agent_dse.dims import Space
from agent_dse import tools
sp = Space(json.load(open('config/space.json')))
s = Store()
print('all_trials:', len(s.all_trials()))
t = tools.tool_trend(s, sp)
print('per_axis keys:', list(t['per_axis']))
print('global top1 :', t['global_top10'][0] if t['global_top10'] else None)
print('validate    :', tools.tool_validate(s, sp.baseline(), sp))