from Data.RecSys2022 import RecSys2022
import pandas as pd


def create_submission(dataset: RecSys2022, model):
    user_ids = dataset.get_target_ids()
    item_original_ID_to_index = dataset.item_original_ID_to_index
    user_original_ID_to_index = dataset.user_original_ID_to_index

    predicts = []
    item_index_to_original_ID = dict(
        (v, k) for k, v in item_original_ID_to_index.items())

    for user_id in user_ids:
        masked_user_id = user_original_ID_to_index[user_id]
        recommended_items = model.recommend(masked_user_id, cutoff=10)

        unmasked_items = []
        for item_id in recommended_items:
            unmasked_items.append(item_index_to_original_ID[item_id])

        value = " ".join(map(str, unmasked_items))
        predicts.append(value)

    # dataframe with user_ids and recommended items
    return pd.DataFrame({'user_id': user_ids, 'item_list': predicts})
