# Recourse-using-Incomplete-Causal-Graph

The following has been tested with `python 3.10`

## Installation Instructions
```shell
git clone https://github.com/cmaspi/Recourse-using-Incomplete-Causal-Graph.git
pip install virtualenv
cd Recourse-using-Incomplete-Causal-Graph/
source _venv/bin/activate # or with equivalent shell extension
# Making Changes in site-packages
rm -r _venv/lib/python3.10/site-packages/numpy/testing/
svn checkout https://github.com/numpy/numpy/branches/maintenance/1.14.x/numpy/testing _venv/lib/python3.10/site-packages/numpy/testing/
sed -i '4s/.*/from collections.abc import Iterable/' _venv/lib/python3.10/site-packages/causalgraphicalmodels/cgm.py
```

## Performing Causal Discovery
```shell
python causal_discovery/main.py
```

## Finding the recourses and generating the tables
```shell
python recourse/main.py --scm_class diamond --classifier_class lr --lambda_lcb 2. --optimization_approach grad_descent --grad_descent_epochs 1000 --batch_number 0 --sample_count 100 --experimental_setups m0_true m1_alin m1_akrr m1_gaus
```

Here, the argument `scm_class` can take the arguments `diamond`, `lin4v`, or `german-credit`. Change this argument to get the corresponding table.
