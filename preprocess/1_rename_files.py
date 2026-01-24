import os
import re
import copy
import json
import pyfbx
import random
import shutil
import itertools


characters = [
    'Aj',
    'Big Vegas',
    'Claire',
    'Goblin D Shareyko',
    'Kaya',
    'Mousey',
    'Mutant',
    'Peasant Man',
    'Sporty Granny',
    'Warrok W Kurniawan',
]

# Define a function to extract the number inside parentheses
def extract_number(filename):
    match = re.search(r'\((\d+)\)', filename)
    return int(match.group(1)) if match else -1


# not work, check if the fbx file is the same
def checkFileContent(file1, file2):
    filename = item.split('  Description:')[0]
    fbx1 = pyfbx.FBX(file1)
    fbx2 = pyfbx.FBX(file2)
    return fbx1 == fbx2


# Specify the path to the folder
folder_path = './woskin/'

# Make folders for each category
category_folder = {}
all_file_mapping = {}

# Get all filenames in the folder
for character in characters:
    fileList = os.listdir(folder_path + character + '_60fps')
    fileList = sorted(fileList, key = extract_number)

    fileList.remove('.DS_Store')
    category_path = character.lower().replace(' ', '_') + '_motion_category.txt'
    error_path = character.lower().replace(' ', '_') + '_motion_category_error.txt'
    if category_path in fileList and error_path in fileList:
        fileList.remove(category_path)
        fileList.remove(error_path)
        with open(folder_path + character + '_60fps/' + category_path, 'r') as file:
            tmp = file.read()
            categoryList = re.split(r',(?!\s)', tmp)
            for item in categoryList:
                if 'NewPage' in item:
                    categoryList.remove(item)

            # compare file list with category list
            print(len(categoryList))
            print(len(fileList))

            fileFolders = {}
            for item in fileList:
                category = item.split('.fbx')[0]
                if '(' in item:
                    category = category.split(' (')[0]
                if category not in fileFolders:
                    fileFolders[category] = []
                fileFolders[category].append(item)

            print('matching')
            matchingDict = {}
            # matchingDict_inv = {}
            unmatched = []
            for item in categoryList:
                category = item.split('  Description:')[0]
                category = category.replace('/', '_')
                if category in fileFolders:
                    try:
                        # tmp = fileFolders[category].pop(0)
                        matchingDict[fileFolders[category].pop(0)] = item
                        # if item not in matchingDict_inv:
                        #     matchingDict_inv[item] = []
                        # matchingDict_inv.append(tmp)
                    except Exception as e:
                        print('Error, Character: {}, Motion: {}'.format(character, category))
                else:
                    unmatched.append(item)
            all_file_mapping[character] = matchingDict

            print('adding all characters to folders')
            for item in categoryList:
                if item not in category_folder:
                    category_folder[item] = []
                category_folder[item].append(character)

    else:
        print('error')

print('sampling')

# train: 1906, test: 477, all: 2383
print('1. sample trainset and testset')
keys_list = list(category_folder.keys())
train_num = int(len(keys_list) / 5 * 4) + 1
test_num = len(keys_list) - train_num
train_category = random.sample(keys_list, train_num)
test_category = [item for item in keys_list if item not in train_category]

print('2. sample kk, ku, uk, uu, 4 testing pairs')
train_characters = [
    'Aj',
    'Big Vegas',
    'Goblin D Shareyko',
    'Kaya',
    'Mousey',
    'Peasant Man',
    'Warrok W Kurniawan',
]
test_characters = [
    'Claire',
    'Mutant',
    'Peasant Man',
    'Sporty Granny',
    'Warrok W Kurniawan',
]
common_characters = [
    'Peasant Man',
    'Warrok W Kurniawan',
]
train_only_characters = [
    'Aj',
    'Big Vegas',
    'Goblin D Shareyko',
    'Kaya',
    'Mousey',
]
test_only_characters = [
    'Claire',
    'Mutant',
    'Sporty Granny',
]
train_itemDict = {}
train_characterDictDesc = {
    'Aj': [],
    'Big Vegas': [],
    'Goblin D Shareyko': [],
    'Kaya': [],
    'Mousey': [],
    'Peasant Man': [],
    'Warrok W Kurniawan': [],
}
train_characterDictFile = {
    'Aj': [],
    'Big Vegas': [],
    'Goblin D Shareyko': [],
    'Kaya': [],
    'Mousey': [],
    'Peasant Man': [],
    'Warrok W Kurniawan': [],
}
test_itemDict = {}
test_characterDictDesc = {
    'Claire': [],
    'Mutant': [],
    'Peasant Man': [],
    'Sporty Granny': [],
    'Warrok W Kurniawan': [],
}
test_characterDictFile = {
    'Claire': [],
    'Mutant': [],
    'Peasant Man': [],
    'Sporty Granny': [],
    'Warrok W Kurniawan': [],
}


def get_filename(character, target):
    keys = [key for key, desc in all_file_mapping[character].items() if desc == target]
    if not keys:
        print('file not found for: {}'.format(target))
        return None
    return keys[0]


def permute(source_list, subset_size):
    # Generate combinations of the subset
    subsets = itertools.combinations(source_list, subset_size)

    # Generate permutations for each combination
    permutations = []
    for subset in subsets:
        subset_permutations = itertools.permutations(subset)
        permutations.extend(subset_permutations)

    return permutations


print('2.0.1. sample trainset, one motion -> one character')
# sample motion for trainset
for key, values in category_folder.items():
    if key in train_category:
        choices = [item for item in values if item in train_characters]
        sampled = random.choice(choices)
        train_itemDict[key] = sampled
        train_characterDictDesc[sampled].append(key)
        # replace the category name with file name
        filename = get_filename(sampled, key)
        train_characterDictFile[sampled].append(filename)
for character in train_characterDictFile.keys():
    print('\t {}: {}'.format(character, len(train_characterDictFile[character])))

print('2.0.2. sample testset, one motion -> one character')
# sample motion for testset
for key, values in category_folder.items():
    if key in test_category:
        choices = [item for item in values if item in test_characters]
        sampled = random.choice(choices)
        test_itemDict[key] = sampled
        test_characterDictDesc[sampled].append(key)
        # replace the category name with file name
        filename = get_filename(sampled, key)
        test_characterDictFile[sampled].append(filename)
for character in test_characterDictFile.keys():
    print('\t {}: {}'.format(character, len(test_characterDictFile[character])))

test_characterDictFile_expand = copy.deepcopy(test_characterDictFile)

print('2.1. sample known motion // known character')
kk_list = []
for character in train_only_characters:
    for file, desc in zip(train_characterDictFile[character], train_characterDictDesc[character]):
        for character_ in common_characters:
            if character_ in category_folder[desc]:
                target_motion = get_filename(character_, desc)
                if target_motion:
                    kk_list.append({
                        'source_character': character,
                        'source_motion_file': file,
                        'terget_character': character_,
                        'target_motion_file': target_motion,
                        'motion description': desc,
                    })
                    # if target_motion not in test_characterDictFile_expand[character_]:
                    #     test_characterDictFile_expand[character_].append(target_motion)
                    # else:
                    #     print('existing motion of {}: {}'.format(character_, target_motion))
print('the length of kk_list: {}'.format(len(kk_list)))

print('2.2. sample known motion // unknown character')
ku_list = []
for character in train_only_characters:
    for file, desc in zip(train_characterDictFile[character], train_characterDictDesc[character]):
        for character_ in test_only_characters:
            if character_ in category_folder[desc]:
                target_motion = get_filename(character_, desc)
                if target_motion:
                    ku_list.append({
                        'source_character': character,
                        'source_motion_file': file,
                        'terget_character': character_,
                        'target_motion_file': target_motion,
                        'motion description': desc,
                    })
                    # if target_motion not in test_characterDictFile_expand[character_]:
                    #     test_characterDictFile_expand[character_].append(target_motion)
                    # else:
                    #     print('existing motion of {}: {}'.format(character_, target_motion))
print('the length of ku_list: {}'.format(len(ku_list)))

print('2.3. sample unknown motion // known character')
uk_list = []
for character in test_only_characters:
    for file, desc in zip(test_characterDictFile[character], test_characterDictDesc[character]):
        for character_ in common_characters:
            if character_ in category_folder[desc]:
                target_motion = get_filename(character_, desc)
                if target_motion:
                    uk_list.append({
                        'source_character': character,
                        'source_motion_file': file,
                        'terget_character': character_,
                        'target_motion_file': target_motion,
                        'motion description': desc,
                    })
                    # if target_motion not in test_characterDictFile_expand[character_]:
                    #     test_characterDictFile_expand[character_].append(target_motion)
                    # else:
                    #     print('existing motion of {}: {}'.format(character_, target_motion))
print('the length of uk_list: {}'.format(len(uk_list)))

print('2.4. sample unknown motion // unknown character')
uu_list = []
permuted_pairs = permute(test_only_characters, 2)
for source, target in permuted_pairs:
    for file, desc in zip(test_characterDictFile[source], test_characterDictDesc[source]):
        if target in category_folder[desc]:
            target_motion = get_filename(target, desc)
            if target_motion:
                uu_list.append({
                    'source_character': source,
                    'source_motion_file': file,
                    'terget_character': target,
                    'target_motion_file': target_motion,
                    'motion description': desc,
                })
                # if target_motion not in test_characterDictFile_expand[target]:
                #     test_characterDictFile_expand[target].append(target_motion)
                # else:
                #     print('existing motion of {}: {}'.format(target, target_motion))
print('the length of uu_list: {}'.format(len(uu_list)))

print('3.0. sample testset')
sample_num = 100
kk_list_sampled = random.sample(kk_list, 100)
ku_list_sampled = random.sample(ku_list, 100)
uk_list_sampled = random.sample(uk_list, 100)
uu_list_sampled = random.sample(uu_list, 100)
for pair in kk_list_sampled:
    if pair['target_motion_file'] not in test_characterDictFile_expand[pair['terget_character']]:
        test_characterDictFile_expand[pair['terget_character']].append(pair['target_motion_file'])
for pair in ku_list_sampled:
    if pair['target_motion_file'] not in test_characterDictFile_expand[pair['terget_character']]:
        test_characterDictFile_expand[pair['terget_character']].append(pair['target_motion_file'])
for pair in uk_list_sampled:
    if pair['target_motion_file'] not in test_characterDictFile_expand[pair['terget_character']]:
        test_characterDictFile_expand[pair['terget_character']].append(pair['target_motion_file'])
for pair in uu_list_sampled:
    if pair['target_motion_file'] not in test_characterDictFile_expand[pair['terget_character']]:
        test_characterDictFile_expand[pair['terget_character']].append(pair['target_motion_file'])

print('Done!')
print('Please manually check the motion clips match with Blender again!')

print('4.0. copy the trainset to a new folder')
trainfile_path = './trainset/'
if not os.path.exists(trainfile_path):
    os.makedirs(trainfile_path)
for character in train_characters:
    if not os.path.exists(trainfile_path + character + '/'):
        os.makedirs(trainfile_path + character + '/')
    for file in train_characterDictFile[character]:
        source_file = folder_path + character + '_60fps/' + file
        if os.path.exists(source_file):
            shutil.copy(source_file, trainfile_path + character + '/')

print('4.1. copy the testset to a new folder')
testfile_path = './testset/'
if not os.path.exists(testfile_path):
    os.makedirs(testfile_path)
for character in test_characters:
    if not os.path.exists(testfile_path + character + '/'):
        os.makedirs(testfile_path + character + '/')
    for file in test_characterDictFile_expand[character]:
        source_file = folder_path + character + '_60fps/' + file
        if os.path.exists(source_file):
            shutil.copy(source_file, testfile_path + character + '/')

print('4. saving the files')
train_dict = {
    'train_files': train_characterDictFile,
    'train_dict': None,
}
test_dict = {
    'test_files': test_characterDictFile_expand,
    'test_dict': None,
    'test_pairs_kk': kk_list_sampled,
    'test_pairs_ku': ku_list_sampled,
    'test_pairs_uk': uk_list_sampled,
    'test_pairs_uu': uu_list_sampled,
}

json_train_path = './train.json'
json_test_path = './test.json'

with open(json_train_path, 'w') as json_file:
    json.dump(train_dict, json_file)
with open(json_test_path, 'w') as json_file:
    json.dump(test_dict, json_file)

print('saved in json files')
