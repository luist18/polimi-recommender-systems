# importing this library so the seed stays the same
import inspect

import Utils.not_random
import Utils.submission as submission
from Data.RecSys2022 import RecSys2022, RecSys2022URMType
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout


def __has_icm(model_class):
    # if any of the keys contains the string 'icm' then it has icm
    return any('icm' in key.lower() for key in inspect.signature(model_class).parameters.keys())


def run_algorithm(model_class, fit_params={}, urm_type=RecSys2022URMType.DEFAULT, dummies=True):
    dataset = RecSys2022(feature_dummies=dummies)
    dataset.build(type=urm_type)

    urm = dataset.get_urm()
    icm = dataset.get_icm()

    urm_type = dataset.get_urm_type()

    print('Dataset loaded successfully with URM type:', urm_type.name)

    model = None
    if __has_icm(model_class):
        model = model_class(urm, icm)
    else:
        model = model_class(urm)

    print(f'Fitting model {model.__class__.__name__}...')
    print(f'Fit params: {fit_params}')
    model.fit(**fit_params)

    # create submission
    submission.save_submission(
        dataset, model, f'{model.__class__.__name__}.csv')


def test_algorithm(model_class, fit_params={}, urm_type=RecSys2022URMType.DEFAULT, dummies=True):
    dataset = RecSys2022(feature_dummies=dummies)
    dataset.build(type=RecSys2022URMType.ONE_INTERACTED)

    urm = dataset.get_urm()
    icm = dataset.get_icm()

    urm_type = dataset.get_urm_type()

    urm_train, urm_validation = split_train_in_two_percentage_global_sample(
        urm, train_percentage=0.8)

    evaluator_validation = EvaluatorHoldout(urm_validation, cutoff_list=[10])

    print('Dataset loaded successfully with URM type:', urm_type.name)

    model = None
    if __has_icm(model_class):
        model = model_class(urm_train, icm)
    else:
        model = model_class(urm_train)

    print(f'Fitting model {model.__class__.__name__}...')
    print(f'Fit params: {fit_params}')
    model.fit(**fit_params)

    print('Computing evaluation...')

    result_df, _ = evaluator_validation.evaluateRecommender(model)

    map_value = result_df.loc[10]["MAP"]
    print(f'MAP@10: {map_value:.7f}')
