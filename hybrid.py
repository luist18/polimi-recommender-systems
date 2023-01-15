################################# IMPORT RECOMMENDERS #################################

from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from Recommenders.BaseRecommender import BaseRecommender

################################## IMPORT LIBRARIES ##################################

from numpy import linalg as LA
import scipy.sparse as sps
from tqdm import tqdm
import numpy as np

#################################### HYBRID CLASS ####################################


class Hybrid(BaseRecommender):

    RECOMMENDER_NAME = "Hybrid"

    def __init__(self, urm, icm):
        super(Hybrid, self).__init__(urm)
        self.icm = icm

        self.epsilon = 0.5
        self.gamma = 0.3
        
        self.Beta = RP3betaRecommender(self.URM_train)
        self.Slim = MultiThreadSLIM_SLIMElasticNetRecommender(self.URM_train)

        # Fit the recommenders
        self.Beta.fit(alpha=0.719514, beta=0.229898, min_rating=0, topK=80, implicit=True, normalize_similarity=True)
        self.Slim.fit(topK=25000, l1_ratio=1.0, alpha=2e-4, workers=4)

    def fit(self, norm=2, epsilon=0.5, interactions_count=10):
        # Instantiate the recommenders
        self.norm = norm
        self.epsilon = epsilon
        self.interactions_count = interactions_count
    
    def _compute_item_score(self, user_id_array, items_to_compute = None):
        n_items = self.URM_train.shape[1]
        
        item_weights = np.empty([len(user_id_array), n_items])
        
        for i in range(len(user_id_array)):
            interactions_count = len(self.URM_train[user_id_array[i],:].indices)
            
            # In a simple extension this could be a loop over a list of pretrained recommender objects
            item_weights_1 = self.Beta._compute_item_score(user_id_array[i], items_to_compute)
            item_weights_2 = self.Slim._compute_item_score(user_id_array[i], items_to_compute)

            norm_item_weights_1 = LA.norm(item_weights_1, self.norm)
            norm_item_weights_2 = LA.norm(item_weights_2, self.norm)

            if norm_item_weights_1 == 0:
                # print("Warning: Norm {} of item weights for recommender 1 is zero. Avoiding division by zero".format(self.norm))
                norm_item_weights_1 = 1
                # raise ValueError("Norm {} of item weights for recommender 1 is zero. Avoiding division by zero".format(self.norm))
            
            if norm_item_weights_2 == 0:
                # print("Warning: Norm {} of item weights for recommender 2 is zero. Avoiding division by zero".format(self.norm))
                norm_item_weights_2 = 1
                # raise ValueError("Norm {} of item weights for recommender 2 is zero. Avoiding division by zero".format(self.norm))
            
            if interactions_count < self.interactions_count:
                item_weights[i,:] = item_weights_1 / norm_item_weights_1 * self.epsilon + item_weights_2 / norm_item_weights_2 * (1-self.epsilon)
            else:
                item_weights[i,:] = item_weights_1 / norm_item_weights_1 * (1-self.epsilon) + item_weights_2 / norm_item_weights_2 * self.epsilon

        return item_weights
    
    
    