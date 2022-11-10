import pickle as pkl
import pandas as pd

fname = 'alpaca_2022-8-24_2022-9-2'
with open(f"{fname}.pkl", "rb") as f:
    object = pkl.load(f)
    
df = pd.DataFrame(object)
df.to_csv(f'{fname}.csv')