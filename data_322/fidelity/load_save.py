import os
from scipy.io import savemat
from importlib import import_module


def module(name: str):
    fidelity = import_module(name).fidelity

    for i in fidelity.keys():
        fidelity_dict = {int(k[3:]): len(v) for k, v in fidelity[i].items()}
        print(i, sum(fidelity_dict.values()), fidelity_dict)

    for num in range(1, 6):
        sub = [i for i in sorted(os.listdir('./data_322')) if f'num{num}' in i][0]
        path = f'./data_322/{sub}'  # path of subfolder
        print(f'{path}/fidelity_num{num}_{name}.mat')
        # savemat(f'{path}/fidelity_num{num}_{name}.mat', fidelity[f'num{num}'])


for D in [5, 6, 7, 8, 9]:
    module(f'D{D}_L2')