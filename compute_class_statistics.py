

"""
This file is responsible for discovering statistics about the datasets we use.
It iterates through the 


"""
from collections import defaultdict
import sys

def compute_statistics(data_filepath):
    """ Accepts a path to a data file that is formatted as the 
        training and val files in this project (i.e. path1 path2).
        It then iterates through each filepath in the files and counts
        the classes. 
        Returns: dict where each entry is class: count
    """

    class_dict = {}
    with open(data_filepath) as f:
        for line in f.readlines():
            _, bbox_fp = line.split(" ")
            file_class_dict = _read_bbox_file(bbox_fp.strip())
            _add_to_dict(file_class_dict, class_dict)            
    return class_dict


def _add_to_dict(from_dict, to_dict):
    """ Adds all the entries in from_dict to to_dict
        If the entry exists already, it sums the values of the two.
    """
    for k,v in from_dict.items():
        if k in to_dict.keys():
            to_dict[k] += from_dict[k]
        else:
            to_dict[k] = from_dict[k]
    

def _read_bbox_file(filepath):
    class_count = defaultdict(int)
    with open(filepath) as f:
        for line in f.readlines():
            class_str = line.split(" ")[0].lower()
            class_count[class_str] += 1
    return class_count




if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(f"--{path}")
        class_dict = compute_statistics(path)
        for key,value in sorted(class_dict.items(), key=lambda x: x[1], reverse=True):
           print("{: >15}: {: >10}".format(key, value))


