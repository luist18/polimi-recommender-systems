import numpy as np
from tqdm import tqdm
import scipy.sparse as sps

from Evaluation.metrics import MAP


def _remove_item_interactions(URM, item_list):
    URM = sps.csc_matrix(URM.copy())

    for item_index in item_list:

        start_pos = URM.indptr[item_index]
        end_pos = URM.indptr[item_index+1]

        URM.data[start_pos:end_pos] = np.zeros_like(
            URM.data[start_pos:end_pos])

    URM.eliminate_zeros()
    URM = sps.csr_matrix(URM)

    return URM


def _prune_users(URM_test, ignore_items_ID, min_ratings_per_user):
    """
    Remove users with a number of ratings lower than min_ratings_per_user, excluding the items to be ignored in the evaluation
    :param URM_test:
    :param ignore_items_ID:
    :param min_ratings_per_user:
    :return:
    """

    users_to_evaluate_mask = np.zeros(URM_test.shape[0], dtype=np.bool)

    URM_test = _remove_item_interactions(URM_test, ignore_items_ID)
    URM_test = sps.csr_matrix(URM_test)

    rows = URM_test.indptr
    n_user_ratings = np.ediff1d(rows)
    new_mask = n_user_ratings >= min_ratings_per_user

    users_to_evaluate_mask = np.logical_or(users_to_evaluate_mask, new_mask)

    return URM_test, users_to_evaluate_mask


class Evaluator:

    def __init__(self, urm):
        self.urm = urm

        self.urm, users_to_evaluate_mask = _prune_users(
            self.urm, np.array([]), 1)

        if not np.all(users_to_evaluate_mask):
            print("Ignoring {} ({:4.1f}%) Users that have less than {} test interactions".format(len(users_to_evaluate_mask) - np.sum(users_to_evaluate_mask),
                                                                                                 100*np.sum(np.logical_not(users_to_evaluate_mask))/len(users_to_evaluate_mask), 1))
        self.n_users, self.n_items = self.urm.shape

        self.users_to_evaluate = np.arange(
            self.n_users)[users_to_evaluate_mask]

        self.users_to_evaluate = list(self.users_to_evaluate)

    def get_user_relevant_items(self, user_id):
        return self.urm.indices[self.urm.indptr[user_id]:self.urm.indptr[user_id+1]]

    def calculate_map(self, recommender):
        mapobj = MAP()

        for user_id in tqdm(self.users_to_evaluate):
            relevant_items = self.get_user_relevant_items(user_id)

            recommended_items = recommender.recommend(
                [user_id], cutoff=10)[0]

            is_relevant = np.in1d(
                recommended_items, relevant_items, assume_unique=True)
            is_relevant_current_cutoff = is_relevant[0:10]
            mapobj.add_recommendations(
                is_relevant_current_cutoff, relevant_items)

        return mapobj.get_metric_value()
