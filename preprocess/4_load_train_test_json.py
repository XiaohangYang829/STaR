import os, sys
import json
import shutil
sys.path.append("./outside-code")
import BVH as BVH

# Specify the path to your JSON file
train_file_path = "./train.json"
test_file_path = "./test.json"

# Open the JSON file and load the data into a dictionary
with open(train_file_path, "r") as json_file:
    train = json.load(json_file)
with open(test_file_path, "r") as json_file:
    test = json.load(json_file)

missing_list = {
    'AJ': [],
    'Big_Vegas': [],
    'Goblin_D_Shareyko': [],
    'Kaya': [],
    'Mousey': [],
    'Peasant_Man': [],
    'Warrok_W_Kurniawan': [],
    'Claire': [],
    'Mutant': [],
    'Sporty_Granny': [],
}

test_path = './test_char_all/'

for dic in test['test_pairs_kk']:
    source_path = test_path + dic['source_character'].replace(' ', '_').replace('Aj', 'AJ') + '/' + dic['source_motion_file']
    target_path = test_path + dic['terget_character'].replace(' ', '_').replace('Aj', 'AJ') + '/' + dic['target_motion_file']
    missing_list[source_path.split('/')[2]].append(source_path)
    missing_list[target_path.split('/')[2]].append(target_path)

for dic in test['test_pairs_ku']:
    source_path = test_path + dic['source_character'].replace(' ', '_').replace('Aj', 'AJ') + '/' + dic['source_motion_file']
    target_path = test_path + dic['terget_character'].replace(' ', '_').replace('Aj', 'AJ') + '/' + dic['target_motion_file']
    missing_list[source_path.split('/')[2]].append(source_path)
    missing_list[target_path.split('/')[2]].append(target_path)

for dic in test['test_pairs_uk']:
    source_path = test_path + dic['source_character'].replace(' ', '_').replace('Aj', 'AJ') + '/' + dic['source_motion_file']
    target_path = test_path + dic['terget_character'].replace(' ', '_').replace('Aj', 'AJ') + '/' + dic['target_motion_file']
    missing_list[source_path.split('/')[2]].append(source_path)
    missing_list[target_path.split('/')[2]].append(target_path)

for dic in test['test_pairs_uu']:
    source_path = test_path + dic['source_character'].replace(' ', '_').replace('Aj', 'AJ') + '/' + dic['source_motion_file']
    target_path = test_path + dic['terget_character'].replace(' ', '_').replace('Aj', 'AJ') + '/' + dic['target_motion_file']
    missing_list[source_path.split('/')[2]].append(source_path)
    missing_list[target_path.split('/')[2]].append(target_path)

# Specify the path to the folder
folder_path = './woskin/'

del_list = []
for i, dic in enumerate(test['test_pairs_kk']):
    source_path = test_path + dic['source_character'].replace(' ', '_').replace('Aj', 'AJ') + '/' + dic['source_motion_file']
    target_path = test_path + dic['terget_character'].replace(' ', '_').replace('Aj', 'AJ') + '/' + dic['target_motion_file']
    anim_s, jname_s, framet_s = BVH.load(source_path.replace('.fbx', '.bvh'))
    anim_t, jname_t, framet_t = BVH.load(target_path.replace('.fbx', '.bvh'))
    if not anim_s.shape[0] == anim_t.shape[0]:
        print('test_pairs_kk not matching: {}'.format(dic))
        del_list.append(i)
for i in del_list:
    del test['test_pairs_kk'][i]

del_list = []
for i, dic in enumerate(test['test_pairs_uk']):
    source_path = test_path + dic['source_character'].replace(' ', '_').replace('Aj', 'AJ') + '/' + dic['source_motion_file']
    target_path = test_path + dic['terget_character'].replace(' ', '_').replace('Aj', 'AJ') + '/' + dic['target_motion_file']
    anim_s, jname_s, framet_s = BVH.load(source_path.replace('.fbx', '.bvh'))
    anim_t, jname_t, framet_t = BVH.load(target_path.replace('.fbx', '.bvh'))
    if not anim_s.shape[0] == anim_t.shape[0]:
        print('test_pairs_uk not matching: {}'.format(dic))
        del_list.append(i)
for i in del_list:
    del test['test_pairs_uk'][i]

del_list = []
for i, dic in enumerate(test['test_pairs_ku']):
    source_path = test_path + dic['source_character'].replace(' ', '_').replace('Aj', 'AJ') + '/' + dic['source_motion_file']
    target_path = test_path + dic['terget_character'].replace(' ', '_').replace('Aj', 'AJ') + '/' + dic['target_motion_file']
    anim_s, jname_s, framet_s = BVH.load(source_path.replace('.fbx', '.bvh'))
    anim_t, jname_t, framet_t = BVH.load(target_path.replace('.fbx', '.bvh'))
    if not anim_s.shape[0] == anim_t.shape[0]:
        print('test_pairs_ku not matching: {}'.format(dic))
        del_list.append(i)
for i in del_list:
    del test['test_pairs_ku'][i]

del_list = []
for i, dic in enumerate(test['test_pairs_uu']):
    source_path = test_path + dic['source_character'].replace(' ', '_').replace('Aj', 'AJ') + '/' + dic['source_motion_file']
    target_path = test_path + dic['terget_character'].replace(' ', '_').replace('Aj', 'AJ') + '/' + dic['target_motion_file']
    anim_s, jname_s, framet_s = BVH.load(source_path.replace('.fbx', '.bvh'))
    anim_t, jname_t, framet_t = BVH.load(target_path.replace('.fbx', '.bvh'))
    if not anim_s.shape[0] == anim_t.shape[0]:
        print('test_pairs_uu not matching: {}'.format(dic))
        del_list.append(i)
for i in del_list:
    del test['test_pairs_uu'][i]

test['test_files'] = missing_list
with open('./test_filtered.json', 'w') as json_file:
    json.dump(test, json_file)
