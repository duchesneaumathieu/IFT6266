from Functionals import *
import pickle
import sys

argv = sys.argv
if len(argv) != 2:
    print "python cpu_to_gpy.py <pickle name>"
    sys.exit()
name = argv[1]

file = open(name+".pkl", "rb")
map = pickle.load(file)
file.close()

map["model"] = map["model"].copy()

file = open(name+"_cpu.pkl", "wb")
pickle.dump(map["model"].copy(), file)
file.close()
