import pickle 


with open('yolo_checkpoints/check_pkl/data_1.pkl', 'rb') as f:
    x1 = pickle.load(f)
    
with open('yolo_checkpoints/check_pkl/data_2.pkl', 'rb') as f:
    x2 = pickle.load(f)
    
print("X1: ", x1)
print("X2: ", x2)