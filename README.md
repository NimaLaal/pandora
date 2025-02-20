# PANDORA

## The Project

Painlessly Attempting a Numerical Democratization Of [pulsar timing] Residual Analysis (PANDORA)

## Unique Features
1. Perform astrophysical inference and model selection (product space sampling) on Pulsar Timing Array (PTA) data sets using normalizing flows.
2. [Multi-pulsar Gibbs sampling](https://arxiv.org/pdf/2410.11944) algorithm for the most general, agnostic, per-frequency Bayesian search for a low-frequency (red) noise process in a PTA data.

## Other Features
Perform various real data set GWB detection analyses the fastest way possible given the limitations of JAX, Numpy, and your own hardware!  

## Getting Started


### Installing
To install with enterprise:
```
conda create -n pandora python=3.11
pip install scikit-sparse==0.4.13
pip install git+https://github.com/nanograv/enterprise.git
pip install git+https://github.com/nanograv/enterprise_extensions.git
pip install git+https://github.com/NimaLaal/pandora.git
```
To install without enterprise
```
conda create -n pandora python=3.11
pip install git+https://github.com/NimaLaal/pandora.git
```
### Note
Make sure your JAX install uses GPU if you would like to use your GPU. For more information go [here](https://docs.jax.dev/en/latest/installation.html) or [here](https://developer.apple.com/metal/jax/) for M-series macs.

## Examples
It is highly recommended that you have ```enterprise``` installed in the same environment as ```pandora```. However, for M-series macs, this comes at the price of not utilizing arm64. If you do not have ```enterprise``` installed, you need to supply things manually to some of the pandora functions. Things like, ```Tspan```, ```TNT```, ```TNr```, ```pulsar positions```, etc. Take a look at the data directory for some of these files for the NG15year data set.

Check the examples folder for a few demo notebooks.

## Authors

[Nima Laal](https://ui.adsabs.harvard.edu/search/filter_author_facet_hier_fq_author=AND&filter_author_facet_hier_fq_author=author_facet_hier%3A%221%2FLaal%2C%20N%2FLaal%2C%20Nima%22&fq=%7B!type%3Daqp%20v%3D%24fq_database%7D&fq=%7B!type%3Daqp%20v%3D%24fq_author%7D&fq_author=(author_facet_hier%3A%221%2FLaal%2C%20N%2FLaal%2C%20Nima%22)&fq_database=(database%3Aastronomy%20OR%20database%3Aphysics)&q=%20author%3A%22Nima%20Laal%22&sort=date%20desc%2C%20bibcode%20desc&p_=0)<br/> 
nima.laal@gmail.com
