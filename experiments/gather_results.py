import glob
import json
import os
import pandas as pd
import re
import sys

args = sys.argv
assert len(args) >= 4
pattern = args[1]
out_path = args[2]

result_dirs = sorted([
    f for f in glob.glob(pattern)
    if os.path.isdir(f)
])
experiment_names = [
    #'/'.join(f.split('/')[-2:])
    f.split('/')[-1]
    for f in result_dirs
]


def get_metric(result_dir, metric):
    with open(os.path.join(result_dir, 'eval_results.json')) as f:
        s = f.read()
    metric_dict = json.loads(s)
    return metric_dict[metric]

results = {}

for metric in args[3:]:
    results[metric] = [get_metric(d, metric) for d in result_dirs]

df = pd.DataFrame(data=results, index=experiment_names)
df.to_csv(out_path)
df.loc['mean', :] = df.mean(axis=0)
print(out_path + ':')
print(df)



