import pickle
import os
import numpy as np
import random
import copy


def load_pickle(path, filename):
    with open(f"{path}/{filename}.pickle", "rb") as f:
        data = pickle.load(f)
        return data


def save_pickle(data, path, filename):
    with open(f"{path}/{filename}.pickle", "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def get_test_train(fold):
    len_fold = len(fold)
    fold_unit = int(len_fold / 5)

    test_list = fold[:fold_unit]
    train_list = fold[fold_unit:]

    return test_list, train_list


PATH = '/home/jinmyeong/code/HEARD'
os.chdir(PATH)

PHEME_path = 'data'
filename = 'PAN_to_PHEME'

PHEME_data = load_pickle(path=PHEME_path, filename=filename)

eid_list = list(PHEME_data.keys())

# train_eid 만 뽑기
train_eid_list = eid_list[:2820]

# test_eid 만 뽑기
test_eid_list = eid_list[2820:]

# # 겹치는 dev_eid 뽑기
# copied_train_eid_list = copy.deepcopy(train_eid_list)
# random.shuffle(copied_train_eid_list)

# len_copied_train_eid_list = len(copied_train_eid_list)
# fold_unit = int(len_copied_train_eid_list / 9)
# val_eid_list = copied_train_eid_list[:fold_unit]

# 안 겹치는 dev_eid 뽑기
len_train_eid_list = len(train_eid_list)
random.shuffle(train_eid_list)

fold_unit = int(len_train_eid_list / 6)
val_eid_list = train_eid_list[:fold_unit]

# # fold 나누기 -> 총 5개의 fold로 나누기
# len_eid_list = len(eid_list)
# fold_unit = int(len_eid_list / 10)

# val_eid_list = eid_list[:fold_unit]

# fold_0 = eid_list[fold_unit:]
# # fold_1 = eid_list[fold_unit * 2:fold_unit * 3]
# # fold_2 = eid_list[fold_unit * 3:fold_unit * 4]
# # fold_3 = eid_list[fold_unit * 4:fold_unit * 5]
# # fold_4 = eid_list[fold_unit * 5:]

# fold_0_test, fold_0_train = get_test_train(fold_0)
# # fold_1_test, fold_1_train = get_test_train(fold_1)
# # fold_2_test, fold_2_train = get_test_train(fold_2)
# # fold_3_test, fold_3_train = get_test_train(fold_3)
# # fold_4_test, fold_4_train = get_test_train(fold_4)

# 겹치는 dev
# data_ids = {"val": val_eid_list, "fold0": {
#     'test': test_eid_list, 'train': train_eid_list}, }

# 원래 HEARD 세팅 값
data_ids = {"val": val_eid_list, "fold0": {
    'test': test_eid_list, 'train': train_eid_list[fold_unit:]}, }

save_pickle(data=data_ids, path=PHEME_path, filename='PAN_data_ids')
