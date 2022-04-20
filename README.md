# ChromaGan Pytorch Remake
## The Pytorch implementation of ChromaGAN
WIP - The core is trainable, but the project it self need some modification
### How to run
- Create new virtual environment
  ```
    conda create -n chroma-gan python=3.7.6
  ```
- Install dependencies
  ```
    pip install -r requirements.txt
  ```
- Run the code
  ```
    python main.py --config configs/config.yaml --mode 2
  ```
  - --config: path to the config file
  - --mode: 
    - 1: Train
    - 2: Test, coloring the sample images

## References
- [ChromaGAN](https://github.com/pvitoria/ChromaGAN)
