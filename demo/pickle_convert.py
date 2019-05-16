import os, sys, pickle
import time

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

import pdb

import argparse

# python pickle_convert.py --in "val2014_feat1024_withboxlist" --out "val2014_feat1024_withboxlist_protocol2"

def parse_args():
    parser = argparse.ArgumentParser(description='convert pickle protocol 3->2')

    parser.add_argument('--in', dest='in_folder',
                        help='input directory (pkl files)',
                        default="train2014_feat1024_withboxlist", 
                        type=str)
    parser.add_argument('--out', dest='out_folder',
                        help='otuput directory (pkl files)',
                        default="train2014_feat1024_withboxlist_protocol2", 
                        type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    base_dir = os.getcwd()

    args = parse_args()

    print('Called with args:')
    print(args)

    in_folder = os.path.join(base_dir, args.in_folder)
    out_folder = os.path.join(base_dir, args.out_folder)

    if not os.path.exists(in_folder):
        raise FileExistsError
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        print('[INFO] Create feature output directory (non exists)')
    
    pkllist = imglist = os.listdir(in_folder)
    num_pkl = len(pkllist)
    print('Loaded files: {}.'.format(num_pkl))

    print('[INFO] Start converting')
    counter = 0
    while (num_pkl > 0):
        num_pkl -= 1

        det_tic = time.time()
        pkl_file = os.path.join(in_folder, pkllist[num_pkl])
        pkl_id = int(pkllist[num_pkl].replace('.pkl',''))
        
        pkl_ = pickle.load(open(pkl_file, 'rb'))
        dest_ = os.path.join(out_folder, pkllist[num_pkl])
        pickle.dump(pkl_, open(dest_,'wb'),protocol=2)
        det_toc = time.time()
        detect_time = det_toc - det_tic

        counter += 1

        sys.stdout.write('pkl_convert: {:d}/{:d} {:.3f}s   \r' \
                    .format(num_pkl + 1, len(pkllist), detect_time))
        sys.stdout.flush()

    print('[DONE]!')