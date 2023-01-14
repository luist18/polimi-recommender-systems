################################# IMPORT RECOMMENDERS #################################

from Recommenders.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
from Recommenders.BaseRecommender import BaseRecommender

################################## IMPORT LIBRARIES ##################################

from numpy import linalg as LA
import scipy.sparse as sps
from tqdm import tqdm
import numpy as np
import similaripy

#################################### HYBRID CLASS ####################################


class Hybrid(BaseRecommender):

    RECOMMENDER_NAME = "Hybrid"

    def __init__(self, urm, icm):
        super(Hybrid, self).__init__(urm)
        self.icm = icm

        self.epsilon = 0.5
        
        self.Beta = RP3betaRecommender(self.URM_train)
        self.Slim = MultiThreadSLIM_SLIMElasticNetRecommender(self.URM_train)

        # Fit the recommenders
        self.Beta.fit(alpha=0.719514, beta=0.229898, min_rating=0, topK=80, implicit=True, normalize_similarity=True)
        self.Slim.fit(topK=25000, l1_ratio=1.0, alpha=2e-4, workers=8)

    def fit(self, epsilon=0.5):
        # Instantiate the recommenders     
        self.epsilon = epsilon
    
    def _compute_item_score(self, user_id_array, items_to_compute = None):
        item_weights_1 = self.Beta._compute_item_score(user_id_array, items_to_compute)
        item_weights_2 = self.Beta._compute_item_score(user_id_array, items_to_compute)

        item_weights = item_weights_1*self.epsilon + item_weights_2*(1-self.epsilon)
        
        return item_weights
    
    
    