## Usage
To generate expected max curves, put a list containing the performance (accuracy, F1, or your measure of choice) of the N trained models on validation data in the main method within plot.py.

The code has a few options for better visualization: 1) log-scale the X-axis, 2) shade the variance (when comparing multiple curves the shading can be distracting), and 3) scaling the X-axis with the average runtime (when comparing approaches with very different run times, it can be more appropriate to scale by total time spent rather than number of trials).


# Show Your Work: Improved Reporting of Experimental Results

This repository contains code for computing expected max validation performance curves as introduced in [_Show Your Work: Improved Reporting of Experimental Information_](https://arxiv.org/abs/1909.03004).

Machine learning and NLP research often involves searching over hyperparameters. Most commonly this is done by training N models on a set of training data, evaluating each of the N models on a held-out validation set, and choosing the best of the N models to evaluate on a test set. Often, this final test number is all that's reported, but there is a lot of useful information in the other experiments. The code in this repository is meant as a way to visualize the N validation results.


## Understanding expected max performance
The X-axis represents the number of hyperparameter trials (or time, if the average time for the experiments is included).

The Y-axis represents the expected max performance for a given X (number of hyperparameter trials). "If I train X models, what is the expected performance of the best one?"

The leftmost point on the curve (X = 1) is the average across the N validation scores.

The shading is +/- the sample standard error (which is similar to the standard deviation), not shaded outside the observed min and max.

If two curves cross, then the best-performing model depends on the budget. Simply saying "Model A outperforms Model B" is ill-defined.


## Assumptions
This was designed as a tool for reporting the expected max performance for budgets n <= N. We leave forecasting performance with larger budgets (n > N) to future work.

This method for computing expected max performance assumes I.I.D. draws. In practice, that means it's appropriate when using random search for hyperparameter optimization, as recommended in general [here](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf). The calculation of the expected max may not be correct if the hyperparameters were chosen using manual search, grid search, or Bayesian optimization.

If you have too few points estimating statistics like the expected max might not be a good idea, and you shoud just report the values.


## Citation

If you use this repository for your research, please cite:

```bibtex
@inproceedings{showyourwork,
 author = {Jesse Dodge and Suchin Gururangan and Dallas Card and Roy Schwartz and Noah A. Smith},
 title = {Show Your Work: Improved Reporting of Experimental Results},
 year = {2019},
 booktitle = {Proceedings of EMNLP},
}
```