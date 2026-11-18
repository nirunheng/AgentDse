import json
import sys
from pathlib import Path  

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_dse.dims import Space
sp = Space(json.load(open('config/space.json')))
print('dims    :', sp.names)
print('total   :', len(sp.all_points()))
print('baseline:', sp.point_brief(sp.baseline()))
p = sp.all_points()[0]
print('fp      :', sp.fingerprint(p))
print('tag     :', sp.tag(p))