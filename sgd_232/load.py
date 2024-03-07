import re
import os
import numpy as np

path = './sgd_232'
files = [os.path.join(path, f) for f in os.listdir(path) if 'log' in f]
for f in files:
    model = re.search('model\d+', f).group(0)
    data = open(f).readlines()
    pattern = 'energy: (-?\d+\.\d+)'
    energy_list = [float(re.search(pattern, line).group(1)) for line in data]
    print(model, min(energy_list))
