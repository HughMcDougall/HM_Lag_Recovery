import pandas as pd
import SIMBA
import os

table_url = "./SIMBA_jobstatus_oneline.dat"

for i in range(0,93*2):
    job_args = SIMBA.get_args(i, table_url=table_url)
    mode = job_args['mode']
    if mode == "line1":
        targ_url = job_args['out_url']+"./outchain-line1.dat"
    else:
        targ_url = job_args['out_url']+"./outchain-line2.dat"

    if not(os.path.exists(targ_url)):
        SIMBA.reset(i, table_url=table_url)

outs = []
for i in range(0,93*2):
    job_args = SIMBA.get_status(i, table_url=table_url)
    if job_args['finished']==False:
        outs.append(i)
