import numpyro

#-----------------------

def model(T,Y,E):

    lag = numpyro.sample.uniform