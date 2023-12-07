## Installation

First, install Python >= 3.9 and PyTorch >= 1.9, e.g. using conda:
```
conda create -n fs-xlt python=3.10
conda activate fs-xlt
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
```

Next install `adapter-transformers`, e.g. from pip:
```
pip install adapter-transformers
```

Then download and install composable-sft:
```
git clone https://github.com/cambridgeltl/composable-sft.git
cd composable-sft
pip install -e .
```

Finally install this package itself:
```
# assuming you're in the root directory
pip install -e .
```
