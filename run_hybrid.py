from Data.RecSys2022 import RecSys2022URMType
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from run_algorithm import test_algorithm, run_algorithm

model_class = SLIMElasticNetRecommender
model_params = {'topK': 1000, 'l1_ratio': 1.0, 'alpha': 2e-4}

test_algorithm(model_class, model_params,
               urm_type=RecSys2022URMType.ONE_INTERACTED, dummies=False)

model_class = RP3betaRecommender
model_params = {'topK': 90, 'alpha': 0.6820374074955758,
                'beta': 0.3613672986607047, 'implicit': True}

test_algorithm(model_class, model_params,
               urm_type=RecSys2022URMType.ONE_INTERACTED, dummies=False)
