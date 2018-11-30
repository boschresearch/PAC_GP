 # Datasets

The datasets used in the publication "Learning Gaussian Processes by Minimizing PAC-Bayesian Generalization bounds" are from the sources listed below. 
For using the mentioned datasets, please provide loading functions in _experiments/load_dataset.py_.

## Benchmark datasets

### Snelson dataset

URL: http://www.gatsby.ucl.ac.uk/~snelson/

download the SPGP_dist.zip and use the files:
- train_inputs
- test_inputs
- test_outputs

### Boston dataset

The Boston dataset can be downloaded via sklearn.datasets.load_boston.

### Pol and Kin40k dataset

URL: https://github.com/trungngv/fgp/tree/master/data

Datasets are in the folder _pol_ and _kin40k_. We merged the training and test data
test_data.asc/train_data.asc and test_labels.asc/train_labels.asc to one dataset and then
used random 90/10 splits.


### SARCOS large-scale experiment

URL: http://www.gaussianprocess.org/gpml/data/

Filenames:
 - Training data: sarcos_inv.mat
 - Test data: sarcos_inv_test.mat
 
 We again merged the training and test dataset and then used random 90/10 splits.


