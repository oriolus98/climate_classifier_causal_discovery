# Climate zones classifier and Causal Discovery with Tigramite
(In progress)

Section 3 of my master thesis.

Starting off a climatic dataset, containing different temporal series variables (soil moisture, radiation, temperature...) defined on different Earth locations, the functions and classes defined on this repository have two main goals. 

First, classify all pixels on the planet on different climatic zones using an unsupervised approach (k-means algorithm with a DTW distance). A dimensionality reduction temporal autoencoder is also tried in order to test if it is possible to reduce the temporal complexity of the problem. 

Then, the causal discovery package Tigramite is used in order to retrieve the causal temporal graphs of the system. 