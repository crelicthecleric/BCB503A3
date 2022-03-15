import sys
import pandas as pd
infile = sys.argv[1]
label = sys.argv[2]
prop = float(sys.argv[3])
outfile = sys.argv[4]

def stratified_sample(infile, label, prop, outfile):
    df = pd.read_csv(infile)
    sample = df.groupby(label, group_keys=False).apply(lambda x: x.sample(frac=prop))
    sample.to_csv(outfile, index=False)

if __name__ == "__main__":
    stratified_sample(infile, label, prop, outfile)