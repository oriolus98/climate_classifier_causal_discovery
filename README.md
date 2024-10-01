# Climate zones classifier and Causal Discovery with Tigramite
(In progress)

Section 3 of my master thesis (./Master_Thesis_Oriol_Chacon.pdf)

Starting off a climatic dataset, containing different temporal series variables (soil moisture, radiation, temperature...) defined on different Earth locations, the functions and classes defined on this repository can be used to both implement temporal series clustering in order to define geographical climatic zones, and to perform a complet causal analysis of the dataset.

The repository contains:

### Notebooks:
Here some introductory tutorials are presented. They come in handy to get acquainted with different graphical causal discovery theoretical concepts. Some of the Figures present in the master thesis  result from these simple experiments:

* notebooks/partial_correlations.ipynb  -  Notebook 1: Partial Correlations. Familiarization with the concept of linear conditional independence. 
* notebooks/pcmci_intro.ipynb    -   Notebook 2: Introduction to PC-MCI. A simple example on how to use different Tigramite functionalities, as well as a reconstruction of the algorithm steps with stats and sklearn, to better understand it.
* notebooks/causal_discovery_example.ipynb    -   Notebook 3: Identifying climatic causal graphs. A comparison of linear and non-parametric tests for identifying a causal graph from tropical averaged data. It also explains how to include experts assumptions in the process.


### climateclassifier/climateclassy.py
This script contains different custom classes used to simplify and automate the experiments:
* DataLoader: It allows to load our dataset as a numpy array with dimensions pixel x time x climatic variable. It also allows to substract time-series seasonality if needed. It is interesting to note that, if we had to deal with a different dataset, we would just need to adapt this class to load the new data format following the same structure, and all subsequent analysis on this project can be easily automated.
* ClimateClassifier: it receives an object of the class DataLoader, and has different functionalities to implement time-series k-means clustering with DTW metric, save or visualize the results. It also includes the option to train a temporal autoencoder in order to reduce the dimensionality of multi variable time-series, to simplify the task of the clustering algorithm. At the end this particular option has not been included in the master thesis, as four-dimensional clustering results were good enough. However, different tests showed the potential of this method. Therefore, it can be useful in the case we need to classify a significantly bigger dataset.
* GridSearcher: Custom Grid Search k-fold hyperparameter optimization for the autoencoder described in the previous class, as sklearn class is not capable to deal with the data structure necessary for this experiment.

### classification_test.py
This script implements the classes described above in order to cluster the hole dataset presented in the master thesis.

### causef.py
This script automates the hole process of discovering the causal graph and estimating causal effects between all links identified for each different pixel. Results obtained are saved in a csv dataset, where each row contains a different pixel with its coordinates, and columns represent a given possible causal link.

### causef2.py
This script follows the structure of the script above, adapted to the future experiment described in the master thesis with a more complet dataset. However, it struggles with the available computational resources right now.
Efficiency and parallelization modifications need to be explored.

### tests/test_data_loader.py
Preliminar functional tests for the repository, implemented with pytest
