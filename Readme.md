# Introduction
This branch contains exepriments for the paper On the Application of Saddle-Point Methods for Combined Equilibrium Transportation Models

To reproduce the experiment
1. Install
2. Run `OFAC_vs_sequential.ipynb`

# Installation
1. Grab bstabler's TransportationNetworks sumbodule: use `git clone  --recurse submodules`
or do `git submodule update --init` after clone
2. [Install](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) conda if not yet
3. Create and activate conda environment
```bash
conda env create -f environment.yml
conda activate tm
```
4. Add this conda environment to your jupyter notebook 
```bash
ipython kernel install --user --name=tm
```
After that you can select `tm` kernel from notebook's kernel menu. 
Alternatively, you can install  jupyter into the environment and run it from there (but it gave me an error while launching the notebook app)
```bash
conda install jupyter -n tm
```
More details about jupyter with conda env [here](https://stackoverflow.com/a/58068850)
