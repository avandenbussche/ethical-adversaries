


# Ethical adversaries

Our framework follows the work described in Delobelle et al's paper ["Ethical Adversaries: Towards Mitigating Unfairness with Adversarial Machine Learning" on arXiv »](https://arxiv.org/abs/2005.06852)

# Get started
You can replicate the experiments from our  by following the following steps.
You can install the required dependencies:
install using [Pipenv](https://pipenv.readthedocs.io/en/latest/) *(install by running `pip install pipenv` in your terminal)* by running `pipenv install`.

move to the src dir and run (for COMPAS):
```shell script
%run main.py --epochs 50 --grl-lambda 50 --batch-size 128 --attack-size 10 --dataset compas --iterations 50 --save-dir ../results --optimize-attribute "race" --measure-attribute "sex"  --measure-attribute "sex"
```

We support the COMPAS dataset which is in the repo, with the parameter: 

- COMPAS: `--dataset compas `


# Hyper-Parameters

 We sought to render all of our modifications accessible via the original codebase’s command line interface. For example, to set the protected attributes to optimize and measure against, respectively, the user could set the following arguments:
 
` –optimize-attribute "sex,race"`
 `–measure-attribute "sex"`
` –measure-attribute "race"`
 
 


## Structure
Our framework uses a Feeder and an adversarial Reader which . More info can be found in our paper.



## What can be found in this repo?

The repo is structured as follows:
 - the _data_ folder contains all used data. Inside are subfolders defined by file extensions. In the _csv_ subfolder are the data sample from Propublica for both predicting recidives and violent recidives (compas-scores-two-years*.csv). Other files are "sanitized" versions with some of the columns removed (because considered useless are not used at all by Propublica). Two files can be found one keeping age as values (integers) and one containing age category as defined in the Propublica study. The _arff_ folder contains the transformed version of "sanitized" data into the arff format which is used by Weka, a library containing various machine learning algorithms. Weka is only used as a preliminary study in order to try to answer the first challenge. Then, we plan to use scikit-learn and the Python language. Again, in this sub-folder, we multiplied the files by removing some columns regarding the prediction of COMPAS (one file contains the score between 1 and 10, another one  contains the risk categories: "Low", "Medium" and "High") or the actual groundtruth. Note that one file should exist for each combination of cases (prediction class *AND* age category or value).

 - the _script_ folder contains python scripts that we use to conduct our experiments. Mainly 2 scripts can be found there: the `attack_model.py` and `main.py`. We edited these two scripts in order to execute our vision in the paper. They are both adjusted to allow the optimization and measurement of multiple protected attributes, specifically race and sex.

 Some scripts mention of secML in the filename. These scripts are the one using the secML library which implements the adversarial ML attacks studied by the PRALab. These scripts reimplement the work done in the 3 other scripts (being: a ML model factory, a script to train a model and a script to perform the adversarial attack).






