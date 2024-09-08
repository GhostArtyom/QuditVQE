import re
from scipy.io import savemat

path = './data_322/Logs'
path_classical = f'{path}/num1~4_classical_D1_L2.log'
path_violation = f'{path}/num1~5_violation_D5~9_L2.log'

classical, violation = {}, {}
for line in open(path_classical).readlines():
    match = re.search('num(\d) D(\d).+Fidelity: (\d.\d+) (\d+)', line)
    if match:
        num = int(match.group(1))
        D = int(match.group(2))
        fidelity = float(match.group(3))
        iteration = int(match.group(4))
        if f'num{num}_D{D}' in classical:
            classical[f'num{num}_D{D}']['fidelity'].append(fidelity)
            classical[f'num{num}_D{D}']['iteration'].append(iteration)
        else:
            classical[f'num{num}_D{D}'] = {'fidelity': [fidelity]}
            classical[f'num{num}_D{D}'].update({'iteration': [iteration]})
for line in open(path_violation).readlines():
    match = re.search('num(\d) D(\d).+Fidelity: (\d.\d+) (\d+)', line)
    if re.search('local minima', line):
        violation.pop(f'num{num}_D{D}')
    if match:
        num = int(match.group(1))
        D = int(match.group(2))
        fidelity = float(match.group(3))
        iteration = int(match.group(4))
        if f'num{num}_D{D}' in violation:
            violation[f'num{num}_D{D}']['fidelity'].append(fidelity)
            violation[f'num{num}_D{D}']['iteration'].append(iteration)
        else:
            violation[f'num{num}_D{D}'] = {'fidelity': [fidelity]}
            violation[f'num{num}_D{D}'].update({'iteration': [iteration]})

fidelity_iteration = {'classical': classical, 'violation': violation}
# savemat(f'./data_322/fidelity_iteration.mat', fidelity_iteration)
