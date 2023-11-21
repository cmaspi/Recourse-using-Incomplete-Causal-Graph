# Recourse-using-Incomplete-Causal-Graph

The following has been tested with `python 3.10`

## Installation Instructions
```shell
git clone https://github.com/cmaspi/Recourse-using-Incomplete-Causal-Graph.git
pip install virtualenv
cd Recourse-using-Incomplete-Causal-Graph/
source _venv/bin/activate # or with equivalent shell extension
rm -r _venv/lib/python3.10/site-packages/numpy/testing/
svn checkout https://github.com/numpy/numpy/branches/maintenance/1.14.x/numpy/testing _venv/lib/python3.10/site-packages/numpy/testing/
sed -i '4s/.*/from collections.abc import Iterable/' _venv/lib/python3.10/site-packages/causalgraphicalmodels/cgm.py

```
