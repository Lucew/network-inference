import evaluateSPI as evspi
from glob import glob
import os
import pandas as pd
import pyspi


def main(result_path: str):

    # get the dataset from the result folder and make sure there is only one
    dataset_path = glob(os.path.join(result_path, '*.parquet'))
    assert len(dataset_path) == 1, f'There is more than one dataset in the result path: {dataset_path}.'
    dataset_path = dataset_path.pop()
    dataset = pd.read_parquet(dataset_path)
    print(f'We have {dataset.shape[1]} signals with {dataset.shape[0]} samples per signal.')

    # start the PySPI package once (so all JVM and octave are active)
    with evspi.HiddenPrints():
        calc = pyspi.calculator.Calculator(subset='fast')

    # load the data
    result_df, measures, timing_dict, defined, terminated, undefined = evspi.find_and_load_results(result_path, dataset)
    timing_dict = {k: v for k, v in sorted(timing_dict.items(), key=lambda item: item[1])}  # works for python > 3.6
    print(timing_dict)


if __name__ == '__main__':
    _datasets = ['rotary', 'keti', 'soda', 'plant1', 'plant2', 'plant3', 'plant4']
    _datasets = ['plant1']
    for data in _datasets:
        main(rf'measurements\all_spis\spi_{data}')
