import subprocess as sp
from penut.utils import TimeCost

with open('a.log', 'w', encoding='UTF-8') as f:
    with TimeCost('Running SQuAD'):
        sp.call(['bash', 'run_squad.sh'], stdout=f, stderr=f)

