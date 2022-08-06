# ChromaGan Pytorch Remake
## The Pytorch implementation of ChromaGAN
WIP - The core is trainable, but the project it self need some modification
### How to run
- Create new virtual environment
  ```
    conda create -n ChromaGAN python=3.7
  ```
- Install dependencies
  ```
    pip install -r requirements.txt
    pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 torchtext==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html --user
  ```
  I don't know why but that torch install works for me
 
- The sample file
  ```
  https://drive.google.com/file/d/1NahYUHfTxO1eRBfDtXTRn8EKiUV0SEKU/
  ```
- Run the code
  ```
    python main.py --mode 1
  ```
  - --config: path to the config file
  - --mode: 
    - 1: Train
    - 2: Coloring the sample images (weight required)

## References
- [ChromaGAN](https://github.com/pvitoria/ChromaGAN)
