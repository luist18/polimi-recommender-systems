from Data.RecSys2022 import RecSys2022URMType
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from run_algorithm import run_algorithm

model_class = RP3betaRecommender
model_params = {'topK': 90, 'alpha': 0.6820374074955758,
                'beta': 0.3613672986607047, 'implicit': True}

run_algorithm(model_class, model_params, RecSys2022URMType.ONE_INTERACTED)
