# importing this library so the seed stays the same
import Utils.not_random
from Data.RecSys2022 import RecSys2022, RecSys2022URMType
from Data_manager.split_functions.split_train_validation_random_holdout import (
    split_train_in_two_percentage_global_sample,
    split_train_in_two_percentage_user_wise)
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender

# building the dataset
# later we can call the build method to get different URM
dataset = RecSys2022()
dataset.build(type=RecSys2022URMType.ONE_INTERACTED)

urm = dataset.get_urm()
icm = dataset.get_icm()

urm_type = dataset.get_urm_type()

print('Dataset loaded successfully with URM type:', urm_type.name)

# create the model with the parameters found with hyper parameter tuning
# model accuracy was 0.03729794851803506
# parameters {'topK': 90, 'alpha': 0.6820374074955758, 'beta': 0.3613672986607047}

rp3beta_recommender = RP3betaRecommender(urm)
rp3beta_recommender.fit(
    topK=90, alpha=0.6820374074955758, beta=0.3613672986607047, implicit=True)


urm_train, urm_test = split_train_in_two_percentage_global_sample(
    urm, train_percentage=0.75)
# urm_train, urm_validation = split_train_in_two_percentage_global_sample(urm, train_percentage = 0.80)

# evaluator_validation = EvaluatorHoldout(urm_validation, cutoff_list=[10])
evaluator_test = EvaluatorHoldout(urm_test, cutoff_list=[10])

result_df, _ = evaluator_test.evaluateRecommender(rp3beta_recommender)

print("MAP 10:", result_df.loc[10]["MAP"])
print(result_df)
