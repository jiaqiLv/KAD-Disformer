import yaml
import argparse
import os
import random

def dict2namespace(config):
    args = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(args, key, new_value)
    return args

def load_args_from_yaml_all(file_path:str):
    """
    Load config parameters using `yaml` files
    """
    with open(file_path,'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return dict2namespace(config=config)

def train_test_file_split(file_path='/code/model/KAD-Disformer/data/KPI-Anomaly-Detection/Processed_dataset',train_ratio=0.8):
    csv_files = [f for f in os.listdir(file_path) if f.endswith('.csv')]
    random.shuffle(csv_files)
    split_index = int(len(csv_files)*train_ratio)
    train_set = csv_files[:split_index]
    test_set = csv_files[split_index:]
    print(train_set)
    print(test_set)
    return train_set,test_set

TRAIN_SET = ['adb2fde9-8589-3f5b-a410-5fe14386c7af.csv']
# TRAIN_SET = ['adb2fde9-8589-3f5b-a410-5fe14386c7af.csv', '4d2af31a-9916-3d9f-8a8e-8a268a48c095.csv', 
#               '7103fa0f-cac4-314f-addc-866190247439.csv', '9c639a46-34c8-39bc-aaf0-9144b37adfc8.csv', 
#               '42d6616d-c9c5-370a-a8ba-17ead74f3114.csv', 'e0747cad-8dc8-38a9-a9ab-855b61f5551d.csv', 
#               '55f8b8b8-b659-38df-b3df-e4a5a8a54bc9.csv', 'f0932edd-6400-3e63-9559-0a9860a1baa9.csv', 
#               '1c6d7a26-1f1a-3321-bb4d-7a9d969ec8f0.csv', '6efa3a07-4544-34a0-b921-a155bd1a05e8.csv', 
#               '05f10d3a-239c-3bef-9bdc-a2feeb0037aa.csv', 'c69a50cf-ee03-3bd7-831e-407d36c7ee91.csv', 
#               'ab216663-dcc2-3a24-b1ee-2c3e550e06c9.csv', '847e8ecc-f8d2-3a93-9107-f367a0aab37d.csv', 
#               'a07ac296-de40-3a7c-8df3-91f642cc14d0.csv', '6a757df4-95e5-3357-8406-165e2bd49360.csv', 
#               'ba5f3328-9f3f-3ff5-a683-84437d16d554.csv', '0efb375b-b902-3661-ab23-9a0bb799f4e3.csv', 
#               '8723f0fb-eaef-32e6-b372-6034c9c04b80.csv', '431a8542-c468-3988-a508-3afd06a218da.csv', 
#               '57051487-3a40-3828-9084-a12f7f23ee38.csv', '43115f2a-baeb-3b01-96f7-4ea14188343c.csv', 
#               'c02607e8-7399-3dde-9d28-8a8da5e5d251.csv']

TEST_SET = ['6d1114ae-be04-3c46-b5aa-be1a003a57cd.csv', '54350a12-7a9d-3ca8-b81f-f886b9d156fd.csv', 
            'da10a69f-d836-3baa-ad40-3e548ecf1fbd.csv', 'ffb82d38-5f00-37db-abc0-5d2e4e4cb6aa.csv', 
            'a8c06b47-cc41-3738-9110-12df0ee4c721.csv', '301c70d8-1630-35ac-8f96-bc1b6f4359ea.csv']


if __name__ == '__main__':
    train_test_file_split()