import os

PROJECT_PATH = os.path.abspath(os.path.join(__file__, *(os.path.pardir,) * 2))
DATASETS_PATH = os.path.join(PROJECT_PATH, 'dataset')


class Paths:
    summaries_folder = os.path.join(PROJECT_PATH, 'summaries')
    models_folder = os.path.join(PROJECT_PATH, 'model_checkpoints')
    inference_results_folder = os.path.join(PROJECT_PATH, 'inference_results')

    class MAMe:
        images_path = os.path.join(DATASETS_PATH, 'data')
        csv_path = os.path.join(DATASETS_PATH, 'MAMe_dataset.csv')
        toy_csv_path = os.path.join(DATASETS_PATH, 'MAMe_toy_dataset.csv')
