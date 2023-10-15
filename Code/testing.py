from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import kci
from dataset_generator import Diamond
import numpy as np
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphClass import CausalGraph

# import os

# os.environ["PATH"] += os.pathsep+

DataGen = Diamond(lambda: np.random.uniform(-3, 3),
                      lambda: np.random.uniform(-1, 1),
                      lambda: np.random.uniform(-1, 1),
                      lambda: np.random.uniform(-1, 1)
                      )

data = DataGen.generate_diamond(1000)

cg = CausalGraph(4)
nodes = cg.G.get_nodes()

bk = BackgroundKnowledge().add_required_by_node(nodes[0],nodes[2]).add_required_by_node(nodes[2],nodes[0])
cg_with_bk = pc(data,alpha=0.05,indep_test="kci",kernelZ="Gaussian",background_knowledge=bk)
print(cg_with_bk.find_adj())
print(cg_with_bk.G.get_graph_edges())