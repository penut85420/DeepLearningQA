import os
import datetime as dt
import subprocess as sp
from penut.utils import TimeCost

ts = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
with open(f'log_{ts}.log', 'w', encoding='UTF-8') as f:
    with TimeCost('ALBERT DRCD'):
        sp.call(['bash', 'demo_drcd.sh'], stdout=f, stderr=f)

