from src.train import Train
from src.testcase import TestCase
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode',type=int, default=1, help='1: train, 2: test')
    parser.add_argument('-c', '--config', type=str, default='configs/config.yaml', help='config file')
    args = parser.parse_args()
    
    if args.mode == 1:
        Train().train()
    elif args.mode == 2:
        TestCase().test()
    else:
        print('Invalid mode')

if __name__ == '__main__':
    main()