import os
import sys
import numpy as np
#-----------------------------
INDS = [8, 10, 23, 24, 80, 81]
np.random.seed(3)
shuffles_per_source = 2
#-----------------------------

targdirs = []
for cont_ind in INDS:

    for shuffle in range(shuffles_per_source):
        #Select another data folder
        line1_ind = cont_ind
        line2_ind = cont_ind
        targdir = "./%02i-%02i-%02i" %(cont_ind, line1_ind, line2_ind)
            
        while cont_ind == line1_ind or line1_ind == line2_ind or cont_ind==line2_ind or targdir in targdirs:
            line1_ind = INDS[np.random.randint(len(INDS))]
            line2_ind = INDS[np.random.randint(len(INDS))]
            targdir = "./%02i-%02i-%02i" %(cont_ind, line1_ind, line2_ind)
        
        targdirs.append(targdir)

        if not  os.path.isdir(targdir): os.makedirs(targdir)

        #Load data from sources
        CONT  = np.loadtxt("./%02i-%02i-%02i/cont.dat" %(cont_ind, cont_ind, cont_ind))
        LINE1 = np.loadtxt("./%02i-%02i-%02i/MGII.dat" %(line1_ind, line1_ind, line1_ind))
        LINE2 = np.loadtxt("./%02i-%02i-%02i/CIV.dat" %(line2_ind, line2_ind, line2_ind))

        #Save files to new homes
        np.savetxt(X=CONT, fname=targdir+"./cont.dat")
        np.savetxt(X=LINE1, fname=targdir+"./line1.dat")
        np.savetxt(X=LINE2, fname=targdir+"./line2.dat")

        print(targdir)

#-----------------------------
print("Done")
