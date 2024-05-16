import os
from sklearn.model_selection import KFold

def split_data(data_file, num_fold=10):
    kf = KFold(n_splits=num_fold, shuffle=True)
    data_file_name = os.path.splitext(data_file)[0]
    print("Processing file:", data_file_name)
    
    with open(data_file, "r") as f:
        data = f.readlines()
        assert len(data) % 4 == 0, "The total number of lines must be a multiple of 4."
        n = len(data) // 4
        fold_number = 0
        for train, test in kf.split(list(range(n))):
            train_file_path = f"{data_file_name}_{fold_number}_train.txt"
            test_file_path = f"{data_file_name}_{fold_number}_test.txt"
            
            with open(train_file_path, "w") as target_file:
                for i in train:
                    target_file.writelines(data[4*i:4*i+4])
                    
            with open(test_file_path, "w") as target_file:
                for i in test:
                    target_file.writelines(data[4*i:4*i+4])
            
            fold_number += 1


data_path = '/home/lr/zym/research/VeloBERT/dataset/E.coli.summary.add.pro.SS.txt'
split_data(data_path)
