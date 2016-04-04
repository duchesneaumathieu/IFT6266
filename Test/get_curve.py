import pickle
import sys

name = sys.argv[1]

file = open(name+".pkl", "rb")
#miss code here
map = pickle.load(file)
file.close()

curve = map["curve"]
file = open(name+"_curve.pkl", "wb")
pickle.dump(file)
file.close()

