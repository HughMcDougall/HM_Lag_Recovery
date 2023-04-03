import pandas as pd
import SIMBA

IN = pd.read_csv("./Data/real_data/twoline_index.dat", sep="\t")
 
conturls = ["./Data/real_data/" + url[2:] for url in IN["cont_url"]]
OUT = {"ID"       : IN["ID"],
       "cont_url" : conturls,
       "line1_url": ["./Data/real_data/" + url[2:] for url in IN["line1_url"]],
       "line2_url": ["./Data/real_data/" + url[2:] for url in IN["line2_url"]],
       "out_url"  : ["./Data/real_data/" + url[2:-8] for url in IN["cont_url"]]
       }

SIMBA.make(OUT)
