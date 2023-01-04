import Utils.not_random
from Data_manager.split_functions.split_train_validation_random_holdout import (
    split_train_in_two_percentage_global_sample,
    split_train_in_two_percentage_user_wise)
from Evaluation.Evaluator import EvaluatorHoldout, Evaluator, EvaluatorNegativeItemSample
import Utils.submission as submission
from Data.RecSys2022 import RecSys2022, RecSys2022URMType
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.ImplicitAlternatingLeastSquares import IALSRecommender

dataset = RecSys2022()
dataset.build(type=RecSys2022URMType.ONE_INTERACTED)

urm = dataset.get_urm()
icm = dataset.get_icm()

urm_type = dataset.get_urm_type()

print('Dataset loaded successfully with URM type:', urm_type.name)

urm_train, urm_test = split_train_in_two_percentage_global_sample(
    urm, train_percentage=0.75)
# urm_train, urm_validation = split_train_in_two_percentage_global_sample(urm, train_percentage = 0.80)

# evaluator_validation = EvaluatorHoldout(urm_validation, cutoff_list=[10])
evaluator_test = EvaluatorHoldout(urm_test, cutoff_list=[10])

ials = IALSRecommender(urm_train)
ials.fit()

result_df, _ = evaluator_test.evaluateRecommender(ials)
print("MAP 10:", result_df.loc[10]["MAP"])
