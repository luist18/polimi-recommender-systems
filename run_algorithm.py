# importing this library so the seed stays the same
import Utils.not_random
import Utils.submission as submission
from Data.RecSys2022 import RecSys2022, RecSys2022URMType


def run_algorithm(model_class, fit_params, urm_type=RecSys2022URMType.DEFAULT):
    dataset = RecSys2022()
    dataset.build(type=RecSys2022URMType.ONE_INTERACTED)

    urm = dataset.get_urm()
    icm = dataset.get_icm()

    urm_type = dataset.get_urm_type()

    print('Dataset loaded successfully with URM type:', urm_type.name)

    print(f'Fitting model {model.__class__.__name__}...')
    print(f'Fit params: {fit_params}')
    model = model_class(urm)
    model.fit(**fit_params)

    # create submission
    submission.save_submission(
        dataset, model, f'{model.__class__.__name__}.csv')
