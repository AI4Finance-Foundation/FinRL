import pickle as pkl
import pandas as pd

fname = 'alpaca_2019-1-1_2022-1-1'
with open(f"{fname}.pkl", "rb") as f:
    object = pkl.load(f)
    
df = pd.DataFrame(object)
df.to_csv(f'{fname}.csv')