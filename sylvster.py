import numpy as np
from scipy import linalg

def Solve_sylvester(G0,G):
    return linalg.solve_sylvester(G0,G0,2*G)