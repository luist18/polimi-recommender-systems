# importing this library so the seed stays the same
import os
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from skopt.space import Real, Integer
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample, split_train_in_two_percentage_user_wise
import Utils.not_random
from Data.RecSys2022 import RecSys2022, RecSys2022URMType
from Data_manager.split_functions.split_train_validation_random_holdout import (
    split_train_in_two_percentage_global_sample,
    split_train_in_two_percentage_user_wise)
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender

# building the dataset
# later we can call the build method to get different URM
dataset = RecSys2022()
dataset.build(type=RecSys2022URMType.ONE_INTERACTED)

urm = dataset.get_urm()
icm = dataset.get_icm()

urm_train, urm_test = split_train_in_two_percentage_global_sample(
    urm, train_percentage=0.8)
# urm_train, urm_validation = split_train_in_two_percentage_global_sample(urm, train_percentage = 0.80)

# evaluator_validation = EvaluatorHoldout(urm_validation, cutoff_list=[10])
evaluator_test = EvaluatorHoldout(urm_test, cutoff_list=[10])

recommender_class = MultiThreadSLIM_SLIMElasticNetRecommender

hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                           evaluator_validation=evaluator_test)

hyperparameters_range_dictionary = {
    "topK": Integer(32, 1024),
    "l1_ratio": Real(0.0001, 1.),
    "alpha": Real(1, 256)
}

recommender_input_args = SearchInputRecommenderArgs(
    # For a CBF model simply put [URM_train, ICM_train]
    CONSTRUCTOR_POSITIONAL_ARGS=[urm_train],
    CONSTRUCTOR_KEYWORD_ARGS={},
    FIT_POSITIONAL_ARGS=[],
    FIT_KEYWORD_ARGS={
        "implicit": True,
    },
    EARLYSTOPPING_KEYWORD_ARGS={},
)

recommender_input_args_last_test = SearchInputRecommenderArgs(
    # For a CBF model simply put [URM_train_validation, ICM_train]
    CONSTRUCTOR_POSITIONAL_ARGS=[urm_test],
    CONSTRUCTOR_KEYWORD_ARGS={},
    FIT_POSITIONAL_ARGS=[],
    FIT_KEYWORD_ARGS={
        "implicit": True,
    },
    EARLYSTOPPING_KEYWORD_ARGS={},
)

output_folder_path = "result_experiments/"

# If directory does not exist, create
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

n_cases = 30
n_random_starts = int(n_cases*0.3)
metric_to_optimize = "MAP"
cutoff_to_optimize = 10

hyperparameterSearch.search(recommender_input_args,
                            recommender_input_args_last_test=recommender_input_args_last_test,
                            hyperparameter_search_space=hyperparameters_range_dictionary,
                            n_cases=n_cases,
                            n_random_starts=n_random_starts,
                            save_model="last",
                            output_folder_path=output_folder_path,  # Where to save the results
                            output_file_name_root=recommender_class.RECOMMENDER_NAME,  # How to call the files
                            metric_to_optimize=metric_to_optimize,
                            cutoff_to_optimize=cutoff_to_optimize,
                            )
