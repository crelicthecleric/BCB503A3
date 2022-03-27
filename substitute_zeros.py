import argparse
import pandas as pd
import random

parser = argparse.ArgumentParser(description='Substitute fraction of dataframe with zeros')
parser.add_argument('input', help='input file')
parser.add_argument('output', help='output file')
parser.add_argument('frac', type=float, help='fraction to replace')
args = parser.parse_args()

def substitute_zeros(infile, outfile, frac):
    data = pd.read_csv(infile)
    label = data[["activity"]]
    data.drop(columns=["activity"])
    
    ix = [(row, col) for row in range(data.shape[0]) for col in range(data.shape[1])]
    for row, col in random.sample(ix, int(round(frac*len(ix)))):
        data.iat[row, col] = 0
        
    data["activity"] = label
    data.to_csv(outfile, index=False)
    
if __name__ == "__main__":
    substitute_zeros(args.input, args.output, args.frac)
