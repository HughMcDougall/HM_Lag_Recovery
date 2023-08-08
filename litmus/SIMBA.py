'''
SIMBA - Slurm Itteration Manager for Batch Inputs

A handy wrapper for using pandas dataframes and .dat files to make tracking / organizing sbatch files

HM 23

Changelog:
1/4 - First release
3/4 - Changed access itteration limits to 1000 passes at 0.1 seconds

TODO:
- In make():
    . Safety check to make sure we aren't over-writing protected keys
    . Give a warning if about to over-write an existing table
    . Allow different data type inputs, default naming etc
    . Add a "comments" column
- In finish():
    .Add a warning if a job has "finished" but hasn't been started
- In reset():
    .Set to reset all jobs in a table (with a warning) if no index provided
- Add command line arguments so jobs can be reset, started and finished from the command line
'''

import pandas as pd
import time
import numpy as np
import os
from argparse import ArgumentParser
import os


#----------------------
#Default variables
_def_tab_url = "./SIMBA_jobstatus.dat"
_load_sleeptime = 1.0
_load_maxits = 50000
#----------------------
def _timef():
    '''Returns local time as a formatted string. For printing start/ end times'''
    
    t= time.localtime()
    wkdays = ["Mon","Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return "%02i:%02i %s %02i/%02i" %(t.tm_hour, t.tm_min, wkdays[t.tm_wday], t.tm_mday, t.tm_mon)
    
def _safetyload(func):
    '''
    Decorate to make sure we never open a file that's already open
    and to check it out from being edited while we're accessing it

    Decorated function must have target file url named 'table_url'
    '''

    argnames = func.__code__.co_varnames[:func.__code__.co_argcount]
    fname = func.__name__

    #--------------------
    def wrapped_func(*args, **kwargs):

        #Get input 'table_url'
        table_url=_def_tab_url
        print("SIMBA SAFETY LOAD DEBUG:\t %s" %fname)
        for argname, arg in zip(argnames, args):
            if argname=='table_url': table_url=arg
            print(arg,argname)
            break

        #Check if file exists
        if not os.path.isfile(table_url):
            if fname=="make":
                #If file doesn't exit, feel free to trigger make...
                return( func(*args, **kwargs) )
            else:
                #But don't try to edit a non existant file otherwise
                raise Exception("Attempted to edit non-existant file %s in %s" %(table_url, fname))
        else:
            #Try to rename the file to see if we have access
            its=0
            while its<_load_maxits:
                its+=1
                try:
                    os.rename(table_url,table_url+"_")
                    os.rename(table_url+"_",table_url)
                    file = open(table_url)
                    print("good login of %s in %s" %(table_url, fname))
                    break
                except:
                    print("bad login of %s attempt %i in %s" %(table_url, its, fname))
                    time.sleep(_load_sleeptime)
        
            #Execute main function
            return( func(*args, **kwargs) )

        #Close the file
        file.close()
    #--------------------
    return wrapped_func

#----------------------
@_safetyload
def make(args=None, table_url=_def_tab_url):
    '''
    Makes a job status table /w args. 
    '''
    #------------------
    #MISSINGNO - different input types

    #------------------
    #Safety Checks for over-writing existing
    for name in ["finished", "start_time", "finish_time", "comment"]:
        assert name not in args.keys(), "Cannot have arg name %s when making job table" %name
    
    print("Making table at %s" %table_url)

    #------------------
    if args!=None:
        Njobs = len(args[list(args.keys())[0]])

    #------------------
    #Make and save dataframe
    df = pd.DataFrame({"finished":[False]*Njobs,
                       "start_time":[" "*len(_timef())]*Njobs,
                       "finish_time":[" "*len(_timef())]*Njobs,
                       "comment":[" "*len(_timef())]*Njobs
                       } | args)
    df.to_csv(table_url, sep="\t")

#----------------------

@_safetyload
def start(i, table_url=_def_tab_url, comment=None):
    '''
    Marks a job as started
    '''

    #Load table
    table = pd.read_csv(table_url, sep='\t', index_col=0)

    #Reset entry
    table.loc[i,"finished"] = False
    table.loc[i,"start_time"] = " "*len(_timef())
    table.loc[i,"finish_time"] = " "*len(_timef())
    table.loc[i,"comment"] = " "
    
    #Locate index i, write to job start time
    table.loc[i,"start_time"] = _timef()
    if type(comment)!=type(None): table.loc[i,"comment"] = comment

    table.to_csv(table_url, sep="\t")

@_safetyload
def finish(i, table_url=_def_tab_url, comment=None):
    #MISSINGNO - print warning if job not started
    
    #Load table
    table = pd.read_csv(table_url, sep='\t', index_col=0)
    
    #Locate index i, write to job start time
    table.loc[i,"finished"] = True
    table.loc[i,"finish_time"] = _timef()

    if type(comment)!=type(None): table.loc[i,"comment"] = comment

    table.to_csv(table_url, sep="\t")
    

@_safetyload
def reset(i=None, table_url=_def_tab_url):
    '''
    Resets a job
    '''
    
    table = pd.read_csv(table_url, sep='\t', index_col=0)
    table.loc[i,"finished"] = False
    table.loc[i,"start_time"] = " "*len(_timef())
    table.loc[i,"finish_time"] = " "*len(_timef())
    table.loc[i,"comment"] = " "

    table.to_csv(table_url, sep="\t")

@_safetyload
def get_args(i=None, table_url=_def_tab_url):
    '''
    Returns all job parameters
    '''
    
    table = pd.read_csv(table_url, sep='\t', index_col=0)

    out = table.iloc[i].to_dict()
    out.pop("finished")
    out.pop("start_time")
    out.pop("finish_time")
    out = {"i":i} | out

    return(out)

def get_status(i=None, table_url=_def_tab_url):
    '''
    Returns status of job. Good for reset checks
    '''
    
    table = pd.read_csv(table_url, sep='\t', index_col=0)

    out = table.iloc[i].to_dict()
    
    out = {"i":i,
           "finished": out["finished"],
           "start_time": out["start_time"],
           "finish_time": out["finish_time"]
           }

    return(out)

#----------------------
def main():
    #MISSINGNO
    #Arg parsing from command line
    pass
