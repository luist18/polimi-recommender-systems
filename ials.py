import scipy.sparse as sps
from numpy import linalg as LA

import Utils.not_random
import Utils.submission as submission
from Data.RecSys2022 import RecSys2022, RecSys2022URMType
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.ImplicitAlternatingLeastSquares import IALSRecommender

dataset = RecSys2022(feature_dummies=False)
dataset.build(type=RecSys2022URMType.ONE_INTERACTED)

urm = dataset.get_urm()
icm = dataset.get_icm()
urm = sps.vstack([urm, icm.T])

urm_type = dataset.get_urm_type()

print('Dataset loaded successfully with URM type:', urm_type.name)

urm_train, urm_test = split_train_in_two_percentage_global_sample(
    urm, train_percentage=0.8)

icm_train, icm_test = split_train_in_two_percentage_global_sample(
    icm, train_percentage=0.8)

# urm_test = sps.vstack([urm_test, icm.T])

evaluator_test = EvaluatorHoldout(urm_test, cutoff_list=[10])

ials = IALSRecommender(urm_train)
ials.fit(iterations=10, factors=48, regularization=1.0, alpha=10)

result_df, _ = evaluator_test.evaluateRecommender(ials)
print("MAP 10:", result_df.loc[10]["MAP"])
