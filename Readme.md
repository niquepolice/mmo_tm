# Introduction
This is the master repository for transportation modelling projects of [MMO lab](https://labmmo.ru/) 
### It is based on
- https://github.com/MeruzaKub/TransportNet
- https://github.com/tamamolis/TransportNet
- https://github.com/Lareton/transport_network_optimization

and also uses code written in other related projects of MMO lab. 

### Content
Repo contains implementations of basic algorithms for the equilibrium traffic assignment problem:
$$\sum_e \sigma_e(f_e) \to \min_{f \in F_d},$$

and the combined travel-demand (twostage) problem:
$$\gamma \sum_e\sigma_e(f_e) + \sum_{ij}d_{ij}\ln d_{ij} \to \min_{\substack{f\in F_d \\\ \sum_j d_{ij}=l_i\\ \sum_i d_{ij}=w_j}}.$$
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

Docker image  might be created on demand to simplify the installation process. We also have remote linux servers for internal use

# Workflow
1. Fork
2. Do whatever you want in the fork
3. Create pull requests to make fixes/improvements in this repo if applicable
4. Create a pull request to include your fork in the list below
5. Strong results should be included in this repo if it does not require overcomplicating the base architecture

# Recognized forks

# TODO
- [ ] Remove sources/targets with zero arrivals or departures before applying sinkhorn algorithm
- [ ] Add experiments for the twostage model with stable dynamics, clean up the ipynb
- [ ] Add requirements.txt or docker
- [ ] Add all necessary `solve_cvxpy` implementations
- [ ] Cover with tests
