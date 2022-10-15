import numpy as np
import argparse
import sys

sys.path.append('.')
from data.data_utils import load_data

def get_args():
    """
    ArgParser function to collect arguments.
    ...
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                           type=str,
                           help="data directory")
    parser.add_argument('--save_dir',
                           type=str,
                           help="save directory")
    parser.add_argument('--chunk_size',
                            default=10000,
                            type=int,
                            help="chunk size for data")
    
    args = parser.parse_args()
    return args

def main():
    """
    Main function to reshape the data if it is the correct size, then it's left alone otherwise it's split into multiple files.
    
    ...
    
    """
    args = get_args()
    data = np.load(args.data_dir, allow_pickle=True)
    
    bs, seq_len, _, _, _ = data.shape
    split_idx = int(bs * seq_len/float(args.chunk_size))
    
    if bs * seq_len > 10000:
        data_split = np.array_split(data, split_idx)
    
    else:
        data_split = [data]
    
    data_split = [x for x in data_split if x.size > 0]
    print(len(data_split))
    for i in range(len(data_split)):
        np.save(args.save_dir + 'chunk-{}.npy'.format(i), data_split[i])
    
    print('\n Finished reorganizing the data...')
    

if __name__ == "__main__":
    main()