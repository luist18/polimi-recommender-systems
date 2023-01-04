import os
import shutil
import zipfile
from enum import Enum

import pandas as pd
import scipy.sparse as sps

from Data import RecSys2022Utils as utils


class RecSys2022URMType(Enum):
    DEFAULT = 0,
    SUM = 1,
    ONE_INTERACTED = 2,
    NEGATIVE_CHECKING_POSITIVE_VIEWING = 3,
    NEGATIVE_CHECKING_ZERO_TO_ONE_VIEWING = 4,
    ONE_CHECKING_ONE_TO_TWO_VIEWING = 5


class RecSys2022:

    # relative path
    DATASET_DIR = 'assets'
    DATASET_NAME = 'recommender-system-2022-challenge-polimi.zip'

    def __init__(self):
        # converts to absolute path
        self.dataset_dir = os.path.join(
            os.path.dirname(__file__), RecSys2022.DATASET_DIR)
        self.dataset_path = os.path.join(
            self.dataset_dir, RecSys2022.DATASET_NAME)

        # assigned to a value during the _unzip() method
        self.extracted_path = None

        self._unzip()

        self.interactions = self._load_interactions()
        self.features = self._load_features()
        self.target_ids = self._load_target_ids()

        # assigned to a value during the build method
        self.item_original_ID_to_index = None
        self.user_original_ID_to_index = None
        self.icm = None
        self.urm = None
        self.urm_type = None

        self._cleanup()

    def _unzip(self):
        print('Unzipping dataset...')

        basename = os.path.basename(self.dataset_path)

        # later to be used in the _cleanup() method
        self.extracted_path = os.path.join(
            self.dataset_dir, f'{basename}-unzipped')

        with zipfile.ZipFile(self.dataset_path, 'r') as zip_ref:
            zip_ref.extractall(self.extracted_path)

    def _cleanup(self):
        print('Cleaning up...')
        shutil.rmtree(self.extracted_path)

    def _load_interactions(self):
        print('Loading interactions...')
        interactions_path = os.path.join(
            self.extracted_path,
            'interactions_and_impressions.csv'
        )

        interactions = utils.load_interactions(interactions_path)

        return interactions

    def _load_features(self):
        print('Loading features...')

        length_path = os.path.join(
            self.extracted_path,
            'data_ICM_length.csv'
        )

        type_path = os.path.join(
            self.extracted_path,
            'data_ICM_type.csv'
        )

        length_features, length_archive = utils.load_length_feature(
            length_path)
        type_features = utils.load_type_feature(type_path)

        features = pd.merge(length_features,
                            type_features, on='item_id')

        features_dict = {
            'length': length_features,
            'length_archive': length_archive,
            'type': type_features,
            'all': features
        }

        return features_dict

    def _load_target_ids(self):
        print('Loading target ids...')

        target_ids_path = os.path.join(
            self.extracted_path,
            'data_target_users_test.csv'
        )

        target_ids = utils.load_target_ids(target_ids_path)

        return target_ids

    def _get_interactions_processing(self, type=RecSys2022URMType.DEFAULT, **kwargs):
        length_archive = self.features['length_archive']

        if type == RecSys2022URMType.DEFAULT:
            return self.interactions
        elif type == RecSys2022URMType.SUM:
            return utils.sum_all_interactions(self.interactions, **kwargs)
        elif type == RecSys2022URMType.ONE_INTERACTED:
            return utils.one_interacted(self.interactions, **kwargs)
        elif type == RecSys2022URMType.NEGATIVE_CHECKING_POSITIVE_VIEWING:
            return utils.negative_checking_positive_viewing(self.interactions)
        elif type == RecSys2022URMType.NEGATIVE_CHECKING_ZERO_TO_ONE_VIEWING:
            return utils.negative_checking_zero_to_one_viewing(self.interactions, length_archive, **kwargs)
        elif type == RecSys2022URMType.ONE_CHECKING_ONE_TO_TWO_VIEWING:
            return utils.one_checking_one_to_two_viewing(self.interactions, length_archive, **kwargs)
        else:
            raise ValueError(f'Invalid type: {type}')

    def build(self, type=RecSys2022URMType.DEFAULT, **kwargs):
        previous_mode = pd.options.mode.chained_assignment
        pd.options.mode.chained_assignment = None

        print(f'Building URM and ICM with criteria {type.name}...')
        interactions = self._get_interactions_processing(type, **kwargs)
        features = self.features['all']

        # id mapping
        features_filter = features[features['item_id'].isin(
            interactions['item_id'])]
        interactions_filter = interactions[interactions['item_id'].isin(
            features_filter['item_id'])]

        mapped_id, original_id = pd.factorize(
            features_filter['item_id'].unique())
        item_original_ID_to_index = pd.Series(mapped_id, index=original_id)

        features_filter['item_id'] = features_filter['item_id'].map(
            item_original_ID_to_index)
        interactions_filter['item_id'] = interactions_filter['item_id'].map(
            item_original_ID_to_index)

        mapped_id, original_id = pd.factorize(
            interactions_filter['user_id'].unique())
        user_original_ID_to_index = pd.Series(mapped_id, index=original_id)

        interactions_filter['user_id'] = interactions_filter['user_id'].map(
            user_original_ID_to_index)

        features_filter = features_filter.set_index('item_id')

        # save for later mapping
        self.item_original_ID_to_index = item_original_ID_to_index
        self.user_original_ID_to_index = user_original_ID_to_index

        # build icm
        icm = sps.csr_matrix(features_filter.values)

        # build urm
        number_of_users = interactions_filter['user_id'].nunique()
        number_of_items = len(features_filter)

        urm = sps.csr_matrix(
            (interactions_filter['data'].values,
             (interactions_filter['user_id'].values, interactions_filter['item_id'].values)),
            shape=(number_of_users, number_of_items))

        self.icm = icm
        self.urm = urm
        self.urm_type = type

        print(interactions.head(5))

        pd.options.mode.chained_assignment = previous_mode

    def get_icm(self):
        if self.icm is None:
            raise ValueError('ICM not built yet. Call build() first.')

        return self.icm

    def get_urm(self):
        if self.urm is None:
            raise ValueError('URM not built yet. Call build() first.')

        return self.urm

    def get_urm_type(self):
        return self.urm_type

    def get_target_ids(self):
        return self.target_ids
