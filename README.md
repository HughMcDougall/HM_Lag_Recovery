# HM_Lag_Recovery
Github repo for talking to getafix and running code.

CHANGELOG
30/3 - First created repo and uploaded to getafix. `LineFit01.py` is main file, will run whatever SIMBA table you feed it at index 'i', e.g.
 ```
  python LineFit01.py -i 15 -table SIMBA_jobstatus.dat -progress_bar 0
``` 
Will run a job with whatever job parameters you have specified in job 15 of `SIMBA_jobstatus.dat`.
