from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import kci
from dataset_generator import *
import numpy as np
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphClass import CausalGraph

from sklearn.svm import SVR
from hsic import hsic_gam
from sklearn.model_selection import train_test_split

from utils import is_consistent

# Uncomment the data generator which you would like to use and comment the others

# Diamond
DataGen = Diamond(
    lambda: np.random.uniform(-3, 3),
    lambda: np.random.normal(0, 1),
    lambda: np.random.normal(0, 1),
    lambda: np.random.normal(0, 1),
)

data = DataGen.generate_diamond(5000)

# # Lin4v
# DataGen = Lin4V(
#     lambda: np.random.uniform(-5, 5),
#     lambda: np.random.normal(0, 1),
#     lambda: np.random.normal(0, 1),
#     lambda: np.random.normal(0, 1),
# )

# data = DataGen.generate_lin4v(5000)

# Semi-Synthetic German Credit
# DataGen = GermanCredit(
#     lambda: np.random.binomial(n=1, p=0.5),
#     lambda: np.random.gamma(10, 3.5),
#     lambda: np.random.normal(0, 0.5),
#     lambda: np.random.normal(0, 2),
#     lambda: np.random.normal(0, 3),
#     lambda: np.random.normal(0, 2),
#     lambda: np.random.normal(0, 5),
# )

# data = DataGen.generate_german_credit(5000)

cg = CausalGraph(len(data))
nodes = cg.G.get_nodes()

# add any background knowledge 
# bk = BackgroundKnowledge().add_required_by_node(nodes[0], nodes[2]).add_required_by_node(nodes[2], nodes[0])

cg_with_bk = pc(
    data,
    alpha=0.05,
    indep_test="kci",
    kernelZ="Gaussian",
    # background_knowledge=bk
)

print("Causal Graph obtained after PC:\n", cg_with_bk.find_adj())

cg = cg_with_bk.find_adj()
causal_graph = []
for x, y in cg:
    if is_consistent(data, x, y):
        causal_graph.append((x, y))

print("Causal Graph consistent with data:\n", causal_graph)