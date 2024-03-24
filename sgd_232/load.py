import re
import os

path = os.getcwd() if 'sgd_232' in os.getcwd() else os.path.join(os.getcwd(), 'sgd_232')
files = [os.path.join(path, f) for f in sorted(os.listdir(path)) if 'log' in f]
for f in files:
    energy_list = []
    model = re.search('model\d+', f).group(0)
    types = re.search('d3_(.+)_model', f).group(1)
    data = open(f).readlines()
    pattern = 'energy: (-?\d+\.\d+)'
    energy_list = [float(re.search(pattern, line).group(1)) for line in data if re.search(pattern, line)]
    min_energy = min(energy_list) if len(energy_list) > 0 else ' No violation'
    print(model, types, min_energy, len(data), sorted(energy_list)[:10])
    print([line[20:] for line in data if str(min_energy) in line][0])

# files = [os.path.join(path, f) for f in sorted(os.listdir(path)) if 'txt' in f]
# for f in files:
#     energy_list = []
#     model = re.search('model\d+', f).group(0)
#     types = re.search('232_(.+)_model', f).group(1)
#     data = open(f).readlines()
#     pattern = '\[\d+, -\d+, \d+, (-?\d+\.\d+).+\],True'
#     energy_list = [float(re.search(pattern, line).group(1)) for line in data if re.search(pattern, line)]
#     min_energy = min(energy_list) if len(energy_list) > 0 else ' No True Label'
#     print(model, types, min_energy, len(energy_list), len(data))

'''
20240324
model1410 POVM -4.038861771619601 11112
model1705 POVM -5.0192710472420945 622
20240319
model1216 promising -9.017427013560935 718
model1410 promising -4.031971471487899 446
model1705 promising -5.041090250419363 1233
model45 promising  No violation 0
20240312
model1216 all_types -9.018750842556233 180 60268
model1410 all_types -4.039285783622273 266 50298
model1705 all_types -5.041619072925202 305 57869
model45 all_types  No True Label 0 5613
model1216 promising -9.018751013802062 3210 18632
model1410 promising -4.039285789849283 1789 6100
model1705 promising -5.041619404551212 5562 16488
model45 promising  No True Label 0 3072
'''