import os
import subprocess

def main():
    data_dir = "/home/charles/Data/Dataset/Collected/电芯13-长边-1/NG/height"
    
    file_list = os.listdir(data_dir)
    
    for file_name in file_list:
        file_path = os.path.join(data_dir, file_name)
        if os.path.isfile(file_path):
            subprocess.run(['../build/test/def_test', file_path, 15, 0.08, 5])
    

if __name__ == "__main__":
    main()