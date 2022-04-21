import os
import csv
import hashlib

path = "/home/beck/Repositories/causaarg/data/semevalt6"
def read(fp, split):
    X = {
        'sentence': [],
        'topic': [],
        'label': [],
        'split': [],
        'id': []
    }
    with open(os.path.join(path, fp), "r", encoding="ISO-8859-1") as infile:
        data = list(csv.reader(infile, delimiter='\t', quotechar = '"', ))
        for dat in data[1:]: #ignore header
            X['id'].append(hashlib.md5(dat[2].encode()).hexdigest())
            X['sentence'].append(dat[2])
            X['topic'].append(dat[1])
            X['label'].append(dat[3])
            X['split'].append(split)
    return X

X_train = read("trainingdata-all-annotations.txt", split="train")
X_val = read("trialdata-all-annotations.txt", split="val")
X_test1 = read("testdata-taskA-all-annotations.txt", split="test")
X_test2 = read("testdata-taskB-all-annotations.txt", split="test")

X = {
    'id': X_train['id'] + X_val['id'] + X_test1['id'] + X_test2['id'],
    'sentence': X_train['sentence'] + X_val['sentence'] + X_test1['sentence'] + X_test2['sentence'],
    'topic': X_train['topic'] + X_val['topic'] + X_test1['topic'] + X_test2['topic'],
    'label': X_train['label'] + X_val['label'] + X_test1['label'] + X_test2['label'],
    'split': X_train['split'] + X_val['split'] + X_test1['split'] + X_test2['split'],
}
