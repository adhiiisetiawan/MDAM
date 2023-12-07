import pickle

# Specify the path to your pickle file
pickle_file_path = 'tsp_20.pkl'

# Load data from the pickle file
with open(pickle_file_path, 'rb') as f:
    loaded_data = pickle.load(f)

print(type(loaded_data))
print(len(loaded_data))
print(loaded_data[1:3])