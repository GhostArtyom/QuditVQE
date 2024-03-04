import os
from scipy.io import savemat
from collections import defaultdict

fidelity = {
    'num1': defaultdict(list),
    'num2': defaultdict(list),
    'num3': defaultdict(list),
    'num4': defaultdict(list),
    'num5': defaultdict(list)
}
fidelity_20240202 = {
    'num1': {},
    'num2': {},
    'num3': {},
    'num4': {},
    'num5': {}
}
fidelity_20240305 = {
    'num1': {},
    'num2': {},
    'num3': {},
    'num4': {},
    'num5': {}
}

for i in fidelity.keys():
    for k, v in fidelity_20240202[i].items():
        fidelity[i][k].extend(v)
    for k, v in fidelity_20240305[i].items():
        fidelity[i][k].extend(v)
    fidelity[i] = dict(fidelity[i])

for i in fidelity.keys():
    for k, v in fidelity[i].items():
        print(i, k, len(v))
    print()

for num in range(1, 6):
    sub = sorted(os.listdir('./data_322'))[num + 1]
    path = f'./data_322/{sub}'  # path of subfolder
    # print(num, fidelity[f'num{num}'])
    # savemat(f'{path}/fidelity_num{num}_D9_L2.mat', fidelity[f'num{num}'])