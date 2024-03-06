"""
    This file display length information regarding the training dataset
"""
import os
from collections import OrderedDict


def getSetDir(set):
    curr_path = os.path.dirname(__file__)
    # dirs = [f"{curr_path}/../dataset/archive/dataset/{set}/", f"{curr_path}/../dataset/archive/datasetSmall/{set}/"]
    dirs = [f"{curr_path}/../dataset/archive/dataset/{set}/"]
    return dirs


def getKeyOrdValue(item):
    return ord(item[0])


def printInfo(dict):
    for k, v in dict.items():
        print(f"- [ {k} ] has {v} files.")


def min_max(dict, set="training"):
    # Use the length to compare the keys
    minChar, minValue = min(dict[set].items(), key=lambda x: x[1])
    maxChar, maxValue = max(dict[set].items(), key=lambda x: x[1])

    return minChar, minValue, maxChar, maxValue


def getSetLengthInfo(dict, dirs, set):
    for dir in dirs:
        dataDir = os.listdir(dir)
        for subdir in dataDir:
            data = {
                "char": subdir.split("/")[-1].lower(),
                "data_length": len(os.listdir(dir + subdir))
            }
            if data["char"] in dict[set]:
                dict[set][data["char"]] += data["data_length"]
            else:
                dict[set][data["char"]] = data["data_length"]


def orderDict(dict, getKey):
    return OrderedDict(sorted(dict.items(), key=getKey))


# Information gathering
contentInfo = {"training": {}, "testing": {}, "validation": {}}

getSetLengthInfo(contentInfo, getSetDir("train"), "training")
getSetLengthInfo(contentInfo, getSetDir("test"), "testing")
getSetLengthInfo(contentInfo, getSetDir("validation"), "validation")

contentInfo["training"] = orderDict(contentInfo["training"], getKeyOrdValue)
contentInfo["testing"] = orderDict(contentInfo["testing"], getKeyOrdValue)
contentInfo["validation"] = orderDict(contentInfo["validation"], getKeyOrdValue)

# Display
minChar, minValue, maxChar, maxValue = min_max(contentInfo, "training")
print("-- Training set length information --")
print(f"- [INFO] The character with minimum size is {minChar} and has {minValue} training examples.")
print(f"- [INFO] The character with maximum size is {maxChar} and has {maxValue} training examples.\n")

minChar, minValue, maxChar, maxValue = min_max(contentInfo, "testing")
print("-- Testing set length information --")
print(f"- [INFO] The character with minimum size is {minChar} and has {minValue} training examples.")
print(f"- [INFO] The character with maximum size is {maxChar} and has {maxValue} training examples.\n")

minChar, minValue, maxChar, maxValue = min_max(contentInfo, "validation")
print("-- Validation set length information --")
print(f"- [INFO] The character with minimum size is {minChar} and has {minValue} training examples.")
print(f"- [INFO] The character with maximum size is {maxChar} and has {maxValue} training examples.")
