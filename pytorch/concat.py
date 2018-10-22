"""
To run encoder.concatenate
"""


import argparse

from pyt_utils import encoder

def main(args):
    encoder.concatenate(args.enc_dir1, args.enc_dir2, args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enc-dir1", type=str)
    parser.add_argument("--enc-dir2", type=str)
    parser.add_argument("-o", "--output-dir", type=str)
    args = parser.parse_args()
    main(args)

