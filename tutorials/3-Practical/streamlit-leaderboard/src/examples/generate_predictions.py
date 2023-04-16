import json
from pathlib import Path

import numpy as np

NUM_PREDS = 100
GROUND_TRUTH_DATA = {str(i): int((i % 3) == 0) for i in range(NUM_PREDS)}

PREDICTIONS_DIR = Path(__file__).parent.absolute() / 'predictions'

def dump_predictions(prediction_dict, prediction_file):
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    with PREDICTIONS_DIR.joinpath(prediction_file).open('w') as f:
        json.dump(prediction_dict, f)

def generate_random_predictions(p):
    return {str(idx): int(value) for idx, value in enumerate(np.random.binomial(1, p, NUM_PREDS))}

if __name__ == '__main__':
    predictions_name_to_predictions = {
        'all_ones': {str(i): 1 for i in range(NUM_PREDS)},
        'all_zeros': {str(i): 0 for i in range(NUM_PREDS)},
        'perfect': GROUND_TRUTH_DATA,
        'worst': {k: 1 - v for k, v in GROUND_TRUTH_DATA.items()},
        'empty_prediction': {},
        'random_0.5': generate_random_predictions(0.5),
        'random_0.25': generate_random_predictions(0.25),
        'random_0.75': generate_random_predictions(0.75)
    }
    for prediction_name, predictions in predictions_name_to_predictions.items():
        dump_predictions(predictions, prediction_name + '.json')
