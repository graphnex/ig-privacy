import os
import sys
import inspect

# setting path

curr_path = os.path.abspath(inspect.getfile(inspect.currentframe()))

dir_level3 = os.path.dirname(curr_path)
dir_level2 = os.path.dirname(dir_level3)
dir_level1 = os.path.dirname(dir_level2)
dir_level0 = os.path.dirname(dir_level1)

sys.path.insert(0, dir_level0)