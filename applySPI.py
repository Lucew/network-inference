import argparse
import os.path

import numpy as np
from pyspi.calculator import Calculator
import loadData as ld
import time
import numpy
from sklearn.preprocessing import StandardScaler
import dill


def main(path: str, dataset: str = 'keti', subset: str = 'fast'):
    # get the dataset
    if dataset == 'keti':
        dataset, _, _ = ld.load_keti(os.path.join(path, 'KETI'), sample_rate='1min')
    elif dataset == 'soda':
        dataset, _, _ = ld.load_soda(os.path.join(path, 'Soda'), sample_rate='1min')
    else:
        raise ValueError('Did not recognize the specified dataset.')
    print(f'From data set {dataset}, we have {dataset.shape[1]} signals with {dataset.shape[0]} samples per signal.')

    # get rid of rooms that have constant signals
    std = dataset.std()
    rooms = {key.split('_')[0] for key in std[std == 0].index.tolist()}
    dataset = dataset.loc[:, [column for column in dataset.columns if column.split('_')[0] not in rooms]]

    # set up the calculation of dependency measures
    tt = time.perf_counter()
    allowed_subsets = ['fast', 'all', 'sonnet', 'fabfour']
    if subset not in allowed_subsets:
        raise ValueError(f'Subset {subset} not allowed. Only {allowed_subsets} supported.')
    calc = Calculator(dataset=dataset.to_numpy().T, subset=subset)

    # get the results
    calc.compute()
    print('Computation took', time.perf_counter()-tt, 'Seconds')

    # save the calculator object as a .pkl
    with open(f'saved_calculator_{dataset}_{subset}.pkl', 'wb') as f:
        dill.dump(calc, f)
    return calc


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument('-p', '--path', type=str, default=r"C:\Users\Lucas\Data")
    _parser.add_argument('-d', '--dataset', type=str, required=True)
    _args = _parser.parse_args()
    main(_args.path, _args.dataset)
