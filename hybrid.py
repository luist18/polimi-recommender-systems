################################# IMPORT RECOMMENDERS #################################

from Recommenders.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
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

    def fit(self):

        # Stack and normalize URM and ICM
        icm_bm25 = similaripy.normalization.bm25plus(self.icm.copy())
        
        # Instantiate the recommenders     
        self.ItemCF1 = ItemKNNCFRecommender(self.URM_train, icm_bm25)
        self.Beta = RP3betaRecommender(self.URM_train)
        self.Slim = SLIM_BPR_Cython(self.URM_train)

        # Fit the recommenders
        self.ItemCF1.fit(2048, 2000, "rp3beta", "bm25plus", "bm25")
        self.Beta.fit(alpha=0.6820374074955758, beta=0.3613672986607047, min_rating=0, topK=90, implicit=True, normalize_similarity=False)
        self.Slim.fit(epochs=20)
        
    
    def _compute_item_score(self, user_id_array, items_to_compute = None):

        item_weights = np.empty([len(user_id_array), 19630])
        for i in tqdm(range(len(user_id_array))):

            w1 = self.Slim._compute_item_score(user_id_array[i], items_to_compute)  
            w2 = self.ItemCF1._compute_item_score(user_id_array[i], items_to_compute)  
            w3 = w1 * 1.25 + w2
            w3 /= LA.norm(w3, 2) 
            w4 = self.Beta._compute_item_score(user_id_array[i], items_to_compute)
            w4 /= LA.norm(w4, 2)
            
            print(w3.shape)
            print(w4.shape)
            w = w3 * 1.25 + w4 * 1
            item_weights[i,:] = w 

        return item_weights
    
    
    