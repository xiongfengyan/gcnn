# Copyright 2018 @ Wuhan Univeristy. All Rights Reserved.
#
# =====================================================
import json, os
import numpy as np

def shuffle_data(filename):
    print("Shuffling the {} data".format(filename))
    with open(filename,'r',encoding='utf-8') as file:
        data=json.load(file)
        feature_size=len(data['features'])
        import random
        random.shuffle(data['features'])
        with open(filename[:-5]+"r.json", 'w') as json_file:
            json_file.write(json.dumps(data,indent=2))
            json_file.close()
        file.close()
    print('Done!')

shuffle_data('./data/sh1128.json')