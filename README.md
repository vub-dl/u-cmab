# Individual-treatment-effect optimisation in dynamic environments
## Overview
Code is split in simulation code, found in the folder `simulation-code` and code for the experiments, found in the folder `u-cmab`.

To install `pylift`, we refer to [pylift](https://github.com/wayfair/pylift), for `cs-um` we refer to [cs-um](https://github.com/vub-dl/cs-um)

## Running the code
All code is provided in Python 3.6.6. Before running any experiments, make sure all dependencies are installed (this could take a few minutes):

```shell
pip install -r requirements.txt
```

and for `pylift` specifically:
```shell
git clone https://github.com/wayfair/pylift
cd pylift
pip install .
cd ..
```

After [installation](https://jupyter.readthedocs.io/en/latest/install.html), all experiments can be run in `jupyter notebook`:
```shell
jupyter notebook
```
Every figure in the submitted paper corresponds with a notebook, provided at the root of this repository. Note that all notebooks are jupyter notebooks, with the exception of one Wolfram Mathematica notebook (`Figure~1.nb`).
Due to the anonymisation process, notebooks are converted to `json`. When copying the notebooks, one can save a file as `ipynb` and open with jupyter notebook. For more information, visit [this guide](http://ipython.org/ipython-doc/rel-1.0.0/interactive/nbconvert.html)
