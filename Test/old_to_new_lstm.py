import pickle
import sys

file = open(sys.argv[1]+".pkl", "rb")
map = pickle.load(file)
file.close()
params = map["params"]

model_parameters = model.get_parameters()

lstm = LSTM(params[0], params[1])
params2 = lstm.get_parameters()

for i in range(len(params2)):
    params2[i].set_value(model_parameters[i].get_value())

file = open("test_copy.pkl", "wb")
map["model"] = lstm
pickle.dump(map, file)
