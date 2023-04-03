'''
data_utils.py

handy functions for converting inputs and outputs

'''

import numpy as np
import jax.numpy as jnp


#===================================
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

    for key,k in zip(keys,range(len(keys))):
        for j in range(sizes[k]):
            if sizes[k]>1:
                out[:, i] = dict[key][:,j]
                out_keys[i] = keys[k]+"_"+str(j)
            else:
                out[:, i] = dict[key][:]
                out_keys[i] = keys[k]
            i+=1

    return(out,out_keys)