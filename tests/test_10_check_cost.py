import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from agent_dse import costfn as cf
from agent_dse.dims import Space

sp = Space(json.load(open('config/space.json')))
b = sp.baseline()
print('baseline area =', round(cf.chip_area(b, sp.core_num), 4), 'mm^2')
print('baseline mat P=', round(cf.unit_power(b,'mac_num'),4), 'W')
print('baseline vec P=', round(cf.unit_power(b,'vec_num'),4), 'W')