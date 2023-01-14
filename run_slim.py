# importing this library so the seed stays the same
import inspect

import Utils.not_random
import Utils.submission as submission
from Data.RecSys2022 import RecSys2022, RecSys2022URMType
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.SLIM.SLIMElasticNetRecommender import \
    MultiThreadSLIM_SLIMElasticNetRecommender

dataset = RecSys2022(feature_dummies=False)
dataset.build(type=RecSys2022URMType.ONE_INTERACTED)

urm = dataset.get_urm()
icm = dataset.get_icm()

urm_type = dataset.get_urm_type()

urm_train, urm_validation = split_train_in_two_percentage_global_sample(
    urm, train_percentage=0.8)

evaluator_validation = EvaluatorHoldout(urm_validation, cutoff_list=[10])

print('Dataset loaded successfully with URM type:', urm_type.name)

model = MultiThreadSLIM_SLIMElasticNetRecommender(urm_train)

fit_params = {'topK': 1000, 'l1_ratio': 1.0, 'alpha': 2e-4, 'workers': 4}

print(f'Fitting model {model.__class__.__name__}...')
print(f'Fit params: {fit_params}')
model.fit(**fit_params)

print('Computing evaluation...')

result_df, _ = evaluator_validation.evaluateRecommender(model)

map_value = result_df.loc[10]["MAP"]
print(f'MAP@10: {map_value:.7f}')
