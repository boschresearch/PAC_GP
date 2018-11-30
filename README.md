# PAC-GP

This is the companion code for the Gaussian process training method reported in the paper
[Learning Gaussian Processes by Minimizing PAC-Bayesian Generalization Bounds by David Reeb et al., NIPS 2018](https://papers.nips.cc/paper/7594-learning-gaussian-processes-by-minimizing-pac-bayesian-generalization-bounds).
The code allows the users to experiment with the proposed GP training method.
Please cite the above paper when reporting, reproducing or extending the results.

## Purpose of the project

This software is a research prototype, solely developed for and published as
part of the publication cited above. It will neither be
maintained nor monitored in any way.

## Requirements

The PAC-GP core code depends on Tensorflow.
For running the experiments, GPflow and sklearn are also required.

## Usage

The PAC-GP interface is similar to the ones of GPy and GPFlow.
Training and evaluation of PAC-GPs is demonstrated in `pac_gp/pac_gp_example.py` on a small synthetic dataset.

## Reproducing PAC-GP results

The experiments reported in the publication can be reproduced by executing

```
python epsilon_study.py  --run --plot
python sparseGP_study.py --run --plot
```
for the dataset _boston_. Results are saved in the folders _epsilon_ and _ind_points_.

If a similiar experiment shall be repeated on a different dataset, the easiest way is to write a new loading function in _experiments/load_dataset.py_.
If one wishes to use PAC-GP in a different context the wrapper function _compare_ in _experiments/helpers.py_ may be used. It takes care of splitting into train/test,
initialization, training and evaluation.

## License

PAC-GP is open-sourced under the MIT license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in PAC_GP, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).
