from src.train import Train
import argparse
from configs.config import load_config
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode',type=int, default=1, help='1: train, 2: test')
    parser.add_argument('-c', '--config', type=str, default='configs/config.yaml', help='config file')
    args = parser.parse_args()
    config = load_config(args.config)
    
    if args.mode == 1:
        Train(config).train()
    elif args.mode == 2:
        Train(config).test()
    else:
        print('Invalid mode')

if __name__ == '__main__':
    main()