################################## IMPORTS ##################################

from hybrid import Hybrid
from Data.RecSys2022 import RecSys2022URMType
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from run_algorithm import run_algorithm

model_class = Hybrid
model_params = {}
run_algorithm(
    model_class, urm_type=RecSys2022URMType.ONE_INTERACTED, dummies=False)

################################ PRODUCE CSV ################################
