'''
data_utils.py

handy functions for converting inputs and outputs

'''

#===================================

import numpy as np
import pandas as pd
import jax.numpy as jnp
from copy import deepcopy as copy
import re

#===================================
def _lc_mode_and_size(data):
    if type(data)==list:
        for item in data:
            assert type(item)==dict and 'T' in item.keys() and 'Y' in item.keys(), "unbanded data in incorrect format in data_tform()"
        mode = 'unbanded'
        Nbands = len(data)
        
    elif type(data)==dict:
        assert 'T' in data.keys() and 'Y' in data.keys() and 'bands' in data.keys(), "banded data of incorrect format in data_tform()"
        mode = 'banded'
        Nbands = np.max(data['bands']) + 1
        
    else:
        
        raise TypeError("Bad input data in data_tform()")

    return(mode,Nbands)

def array_to_lc(data):
    if data.shape[1]==3:
        return {"T": data[:,0],
                "Y": data[:,1],
                "E": data[:,2]
            }
    elif data.shape[1]==2:
        return {"T": data[:,0],
                "Y": data[:,1]
            }
    else:
        raise TypeError("Band data input in array_to_lc()")
    
def lc_to_banded(lcs):
    '''
    Takes a list of dicts {T, Y, E} of lightcurve objects and returns as single banded lightcurve
    '''
    Nbands = len(lcs)

    T = jnp.concatenate([lc['T'] for lc in lcs])
    Y = jnp.concatenate([lc['Y'] for lc in lcs])
    E = jnp.concatenate([lc['E'] for lc in lcs])

    bands = jnp.concatenate([jnp.zeros(len(lc['T']), dtype='int32') + band for band,lc in zip(range(Nbands),lcs)])

    out = {
        'T': T,
        'Y': Y,
        'E': E,
        'bands': bands,
    }

    return(out)

def banded_to_lc(data):
    '''
    Takes banded data and returns as a list of dicts with keys {T, Y, E}
    '''

    #Find how many bands there are
    bands = data['bands']
    Nbands = jnp.max(bands)+1
    out=[]

    #Split into dictionaries and append
    for i in range(Nbands):
        T = data['T'][bands==i]
        Y = data['Y'][bands==i]
        E = data['E'][bands==i]

        out.append({
            'T':T,
            'Y':Y,
            'E':E,
            })

    return(out)

def data_tform(data, tform_params=None, sort = False):
    '''
    Transforms a set of data by shifting, scaling and delaying
    '''

    tformed_data = copy(data)

    #------------------------------
    
    #Determine type of input:
    mode, Nbands = _lc_mode_and_size(tformed_data)

    #------------------------------

    #Get default values
    params = {"amps":   np.ones(Nbands),
              "lags":   np.zeros(Nbands),
              "means":  np.zeros(Nbands)}
    
    if type(tform_params) != type(None):
        params = params | tform_params

    #------------------------------
    
    #Apply transformation
    if mode == 'unbanded':
        for signal, i in zip(tformed_data,range(Nbands)):
            signal["T"]-=params["lags"][i]

            signal["Y"]-=params["means"][i]
            signal["Y"]/=params["amps"][i]

            if "E" in signal.keys(): signal["E"]/=params["amps"][i]
                
    elif mode == 'banded':
        
        tformed_data["T"]-=params["lags"][tformed_data["bands"]]

        tformed_data["Y"]-=params["means"][tformed_data["bands"]]
        tformed_data["Y"]/=params["amps"][tformed_data["bands"]]

        if "E" in data.keys():
            tformed_data["E"]/=params["amps"][tformed_data["bands"]]

        if sort:
            sort_inds = jnp.argsort(tformed_data["T"])
            tformed_data["T"]=tformed_data["T"][sort_inds]
            tformed_data["Y"]=tformed_data["Y"][sort_inds]
            if "E" in data.keys(): tformed_data["E"]=tformed_data["E"][sort_inds]

    return(tformed_data)

def _banded_tform(data, tform_params):

    '''jit-friendly version of data transformation with less safety checks'''

    tformed_data = copy(data)

    tformed_data["T"] -= tform_params["lags"][data["bands"]]

    tformed_data["Y"] -= tform_params["means"][data["bands"]]
    tformed_data["Y"] /= tform_params["amps"][data["bands"]]

    tformed_data["E"] /= tform_params["amps"][data["bands"]]

    return(tformed_data)

def normalize_tform(data):
    '''
    Returns paramaters to roughly normalize a set of data
    '''
    #------------------------------
    
    #Determine type of input:
    mode, Nbands = _lc_mode_and_size(data)
    
    #------------------------------
    #Get default values
    params = {"amps":   np.ones(Nbands),
              "lags":   np.zeros(Nbands),
              "means":  np.zeros(Nbands)}

    #Put in unbanded mode, easier to work with:
    if mode =='banded':
        unbanded_data = banded_to_lc(data)
    else:
        unbanded_data = data
    tmin = np.min(unbanded_data[0]["T"])
    
    for signal, i in zip(unbanded_data, range(Nbands)):
        tmin = min(tmin,np.min(signal["T"]))
        
        if "E" in signal.keys():
            w = signal["E"]**-2
        else:
            w=np.ones(len(signal["T"]))
        wsum = np.sum(w)

        params["means"][i] = np.sum(signal["Y"]*w) / wsum
        params["amps"][i]  = np.sqrt( np.sum( (signal["Y"] - params["means"][i])**2 * w) / wsum )
    params["lags"]+=tmin
    return(params)
        
#===================================
def flatten_dict(dict):
    '''
    Unpacks all entries in a dictionary into a numpy friendly array
    An alternative to using pandas
    '''

    keys = list(dict.keys())
    sizes = [1]*len(keys)

    Ncols = 0
    Nrows = dict[keys[0]].shape[0]

    #Figure out how many cols we need
    for key,i in zip(keys,range(len(keys))):
        if len(dict[key].shape)>1: sizes[i]=dict[key].shape[1]
        assert dict[key].shape[0]==Nrows, "To flatten_dict, all entries must be same length"
        Ncols+=sizes[i]

    #Read data into numpy array
    out = np.zeros([Nrows,Ncols])
    out_keys = [''] * Ncols
    i=0

    for key,k in zip(keys, range(len(keys))) :

        #Write each entry to a column
        for j in range(sizes[k]):

            if sizes[k]>1:
                out[:, i] = dict[key][:,j]
                out_keys[i] = keys[k]+"_"+str(j)

            else:
                if len(dict[key].shape)==1:
                    out[:, i] = dict[key][:]
                else:
                    out[:, i] = dict[key][0,:]

                out_keys[i] = keys[k]
            i+=1

    return(out,out_keys)

def unflatten_dict(samples):
    # ------
    # Get names
    names = []
    keys = samples.keys()
    for key in keys:
        if bool(re.search(".*_[0-9]", key)): names.append(key[:-2])

    counts = [names.count(name) for name in np.unique(names)]
    names = np.unique(names)

    out = {key: samples[key] for key in keys if not bool(re.search(".*_[0-9]", key))}

    print(out)
    print(names, counts)

    # ------
    # Assemble
    for name, count in zip(names, counts):
        N = len(samples[name + "_0"])
        print(name, count, N)

        to_add = {name: []}
        for j in range(N):  # For each row
            to_append = [0] * count
            for i in range(count):  # Get the values from each name
                to_append[i] = samples[name + "_" + str(i)][j]
            to_append = jnp.array(to_append)
            to_add[name].append(to_append)

        out = out | to_add

        out_sorted = {}
        for key in sorted(out.keys()):
            out_sorted = out_sorted | {key: jnp.array(out[key])}
    return (out_sorted)


#===================================
def default_params(Nbands):

    out ={"log_tau" : jnp.log(400),
          "log_sigma_c": 0.0,
          "lags":       jnp.zeros(Nbands-1),
          "rel_amps":   jnp.ones(Nbands-1),
          "means":      jnp.zeros(Nbands),
    }

    return(out)

#===================================
if __name__=="__main__":
    '''
    Some simple unit tests
    '''

    #load some example data
    cont  = array_to_lc(np.loadtxt("./Data/data_fake/clearsignal/cont.dat"))
    line1 = array_to_lc(np.loadtxt("./Data/data_fake/clearsignal/line1.dat"))
    line2 = array_to_lc(np.loadtxt("./Data/data_fake/clearsignal/line2.dat"))

    #Offset times by 100
    cont["T"] +=100
    line1["T"]+=100
    line2["T"]+=100

    #Test band / unbanding
    lcs_unbanded = [cont, line1, line2]
    lcs_unbanded = [cont]
    lcs_banded = lc_to_banded(lcs_unbanded)
    lcs_unbanded = banded_to_lc(lcs_banded)

    #Test normalizing
    norm_param_unbanded = normalize_tform(lcs_unbanded)
    norm_param_banded = normalize_tform(lcs_banded)

    data_tform(lcs_unbanded)
    data_tform(lcs_banded)

    data_tform(lcs_unbanded,    norm_param_unbanded)
    data_tform(lcs_banded,      norm_param_banded)

    test = {"c_0": [0, 1, 2], "a_0": [0, 1, 2], "a_1": [0.1, 1.1, 2.1], "x": [0, 1, 2], "y": [0, 1, 2],
            "b_0": [0, 1, 2],
            "b_1": [0.1, 1.1, 2.1], "b_2": [0.2, 1.2, 2.2]}
    unflatten_dict(test)

    print("Tests done")



    
