# Uplift Modeling is a Contextual Multi-Armed Bandit
This git repository is complementary to a [SIGKDD 2019](https://www.kdd.org/kdd2019/Calls/view/kdd-2019-call-for-research-papers) submission titled _Uplift Modeling is a Contextual Multi-Armed Bandit_, it includes all code used for experimentation and plotting.

## Overview
Code is split in simulation code, found in the folder `simulation-code` and code for the uplifted contextual multi-armed bandit (U-CMAB), found in the folder `u-cmab`. As `simulation-code` is according to [3], we refer to the appropriate [git repository](https://github.com/vub-dl/cs-um) for further documentation. While no codebase is currently available for the Fourier based approximation method [13], `u-cmab/fourier.py` constitutes our interpretation.

The required packages are listed in the `requirements.txt` file, with `pylift` as an exception. We use `pylift` to measure against the U-CMAB. In order to properly install `pylift`, we refer to its [git repository](https://github.com/wayfair/pylift) and [documentation](https://pylift.readthedocs.io/en/latest/installation.html).

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
Every figure in the submitted paper corresponds with a notebook, provided at the root of this repository. Note that all notebooks are jupyter notebooks, with the exception of one Wolfram Mathematica notebook (`Figure~2.nb`).
Note that due to the anonymisation process, notebooks are converted to `json`. When copying the notebooks, one can save a file as `ipynb` and open with jupyter notebook.

---
[3] Jeroen Berrevoets and Wouter Verbeke. 2019. Causal Simulations for Uplift Modeling. _arXiv preprint arXiv:2560.119_ (2019).

[13] George Konidaris, Sarah Osentoski, and Philip S Thomas. 2011. Value function approximation in reinforcement learning using the Fourier basis.. In _AAAI,_ Vol. 6.7.