# Hedonic Games
> Community detection using game theory and alternative approaches

## Full paper (technical report)

Available at: https://lucaslopes.github.io/hedonic/doc/paper.pdf

## To run experiments

1. Run the python file: `$ python compare.py`

If you want to change experiment parameters, change it at the end of the file.

## To visualize results

1. Run the python file: `$ python plot_results.py`

It will generate:

1. [Main Plot](https://lucaslopes.github.io/hedonic/doc/main_plot.png) !!! UPDATED !!!

A bug was fixed in plot (d), this was the previous version (submitted to ICC 2021):

<!-- ![https://lucaslopes.github.io/hedonic/doc/main_plot_submitted.png](./doc/main_plot_submitted.png) -->
<img src="./doc/submitted/main_plot_submitted.png" alt="Main Plot (submitted to ICC 2021)"/>

And here is the current version (bug fixed and `pol_books` changed order with `pol_blogs`):

<!-- ![https://lucaslopes.github.io/hedonic/doc/main_plot.png](./doc/main_plot.png) -->
<img src="./doc/main_plot.png" alt="Main Plot (bug fixed)"/>

2. Fig 1 (a) is considered at multiple noise levels (saved in `./outputs/noises/plots/`)

3. An [animated gif](https://lucaslopes.github.io/hedonic/doc/noises.gif) accounting for all noise levels is shown below

<!-- ![https://lucaslopes.github.io/hedonic/noises.gif](./doc/noises.gif) -->
<img src="./doc/noises.gif" alt="noises" width="500"/>