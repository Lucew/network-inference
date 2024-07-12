import numpy as np
from pyspi.calculator import Calculator
import loadData as ld
import time
import numpy
from sklearn.preprocessing import StandardScaler
import dill


def main():
    # get the dataset
    dataset, _, _ = ld.load_data(sample_rate='1min')
    print(f'We have {dataset.shape[1]} signals with {dataset.shape[0]} samples per signal.')

    # get rid of rooms that have constant signals
    std = dataset.std()
    rooms = {key.split('_')[0] for key in std[std == 0].index.tolist()}
    dataset = dataset.loc[:, [column for column in dataset.columns if column.split('_')[0] not in rooms]]

    # set up the calculation of dependency measures
    tt = time.perf_counter()
    subset = 'fast'
    calc = Calculator(dataset=dataset.to_numpy().T, subset=subset)

    # get the results
    calc.compute()
    print('Computation took', time.perf_counter()-tt, 'Seconds')

    # save the calculator object as a .pkl
    with open(f'saved_calculator_{subset}.pkl', 'wb') as f:
        dill.dump(calc, f)
    return calc


if __name__ == '__main__':
    main()
