import functools
import typing
import argparse
import os

from tqdm import tqdm

import numpy as np
import pandas as pd

import evaluateSPI as evspi


class InformationFlow:

    def __init__(self, time_diff: int = 1, dt: float = 1.0, normalize: bool = True):
        """
        The object that will compute the information flow.

        All operations are defined in:

        San Liang X. Unraveling the cause-effect relation between time series[J].
        Physical Review E, 2014, 90(5): 052150. doi:10.1103/PhysRevE.90.052150
        (Initital Information Flow Measure)

        San Liang X. Normalizing the causality between time series[J].
        Physical Review E, 2015, 92(2): 022126. doi:10.1103/PhysRevE.92.022126
        (Normalization of the information flow and error estimation)

        :param time_diff: The time difference for the euler scheme derivation necessary
        :param dt: The time distance between the samples (inverse of the sampling rate)
        """

        # set the parameters
        self.time_diff = time_diff
        self.dt = dt
        self.normalize = normalize

        # set the data sources as placeholders (created in fit)
        self.dataset = None
        self.vars = None
        self.cov_diffs = None

        # create a placeholder for the final results after fit
        self.result = None

    @staticmethod
    def cov(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        assert x1.shape == x2.shape and x1.ndim == 1, 'The arrays have not the expected shape.'
        return np.sum((x2 - np.mean(x2)) * (x1 - np.mean(x1)))/(x1.shape[0]-1)

    def euler_diff(self, x1: np.ndarray) -> np.ndarray:
        assert x1.ndim == 1 and x1.shape[0] > self.time_diff, 'The arrays have not the expected shape.'

        # create and empty diff so we keep the dimension size
        dx1 = np.zeros_like(x1)

        # make the euler difference
        dx1[:-self.time_diff] = (x1[self.time_diff:] - x1[:-self.time_diff]) / (self.time_diff * self.dt)

        # set the last few values equal as they are not computable
        dx1[-self.time_diff:] = dx1[-self.time_diff-1]
        return dx1

    def information_flow(self, name1: str, name2: str):
        """
        Compute the information flow from signal two to signal one.
        T(2->1)

        :param name1: The signal the information flows to
        :param name2: The origin of the information
        :return:
        """

        # we do it differently, but compare with: https://github.com/Koni2020/pyLiang

        # get the signals from the dataframe
        signal1 = self.dataset[name1].to_numpy()
        signal2 = self.dataset[name2].to_numpy()
        d_signal1 = self.euler_diff(signal1)

        # get the corresponding correlations we need for information flow, which we pre-computed
        c_11 = self.vars[name1]
        c_22 = self.vars[name2]
        c_1d1 = self.cov_diffs[name1]

        # compute the missing correlations
        c_12 = self.cov(signal1, signal2)
        c_2d1 = self.cov(signal2, d_signal1)

        # compute the information flow
        t_21 = (c_11 * c_12 * c_2d1-c_12**2 * c_1d1)/(c_11**2 * c_22 - c_11 * c_12**2)

        # check whether we want to normalize
        z_21 = 1
        if self.normalize:
            # compute the necessary variables for the normalization
            c = np.cov(signal1, signal2)
            det_c = c[0, 0]*c[1, 1]-c[1, 0]*c[0, 1]  # see https://en.wikipedia.org/wiki/Determinant
            p = (c_22 * c_1d1 - c_12 * c_2d1)/det_c
            q = (-c_12 * c_1d1 + c_11 * c_2d1)/det_c
            c_d1d1 = self.cov(d_signal1, d_signal1)

            # compute the intermediate elements for the normalization
            dh_1 = p
            fact = self.dt/(2 * c_11)
            dh_1_noise = fact * (c_d1d1 + p**2 * c_11 + q**2 * c_22 - 2 * p * c_1d1 - 2 * q * c_2d1 + 2 * p * q * c_12)

            # compute the normalization factor
            z_21 = np.abs(t_21) + np.abs(dh_1) + np.abs(dh_1_noise)

        return t_21/z_21

    def fit(self, dataset: pd.DataFrame, progress: bool = False):
        """
        The function to call to fit the information flow to a dataset.

        :param dataset: The dataframe, where the index is the time, and the columns are the signals/processes
        :param progress: Whether to display progress bar
        :return:
        """

        # copy the dataset
        self.dataset = dataset

        # first compute all the standard derivations for all signals (the C_ii in the paper)
        self.vars = pd.Series({name: np.var(self.dataset[name].to_numpy()) for name in self.dataset.columns})

        # secondly, compute all the difference standard derivations (the C_idi in the paper)
        self.cov_diffs = pd.Series({name: self.cov(self.dataset[name].to_numpy(),
                                                   self.euler_diff(self.dataset[name].to_numpy()))
                                    for name in self.dataset.columns})

        # create the result dataframe
        self.result = pd.DataFrame(columns=self.dataset.columns, index=self.dataset.columns, dtype=float)

        # create the progress bar or no progress bar
        if progress:
            wrapper = functools.partial(tqdm, desc='Computing Information Flow', total=len(self.dataset.columns))
        else:
            def wrapper(input_iterable: typing.Iterable):
                for __item in input_iterable:
                    yield __item

        # Iterate over all pairs of names and compute the directed information flow
        # notation wise each column tells how much information flows from the column to each index signal
        #
        # we could do this using a list expression. However, the overhead by the python loops is minimal
        for idx1, name1 in wrapper(enumerate(self.dataset.columns)):
            for idx2, name2 in enumerate(self.dataset.columns):

                # check whether they are both the same and set to NaN
                # comparison on indices is way faster that comparing (potential long) column strings
                if idx1 == idx2:
                    self.result.loc[name1, name2] = np.NAN
                    continue

                # compute the normalized information flow from name2 to name1 (column to index)
                tau_12 = self.information_flow(name1, name2)

                # save the result in the dataframe
                self.result.loc[name1, name2] = tau_12
        return self

    def score(self, symmetric: bool = False, absolute: bool = False):
        """
        Receive the computed information flow. For symmetric the absolute value is taken, and the similarities in both
        directions are added. For "absolute" we return the absolute value.

        Negative values indicate that the information flow decreases chaos, and positive values indicate that the
        information flow increases chaos.
        :param symmetric: Whether to symmetrize the information flow (induces absolute=True)
        :param absolute: Whether to receive the absolute values
        :return:
        """

        # check that we have called fit first
        if self.dataset is None or self.vars is None or self.cov_diffs is None:
            raise ValueError("The Information Flow has not been computed yet. Please call fit() first.")

        result = self.result.copy()
        if symmetric:
            np_values = np.abs(result.to_numpy())
            result = pd.DataFrame(data=np_values+np_values.T, index=self.result.columns, columns=self.result.columns)
        elif absolute:
            result = result.abs()
        print(result)
        return result


def main(dataset_path: str, save_path: str, normalize: bool, symmetric: bool, absolute: bool, progress: bool):

    # read the dataset
    dataset = pd.read_parquet(dataset_path)

    # find the rooms
    rooms = {col.split('_', 1)[0] for col in dataset.columns}

    # create the information flow estimator
    estimator = InformationFlow(normalize=normalize)

    # make the fit and extract the score
    score = estimator.fit(dataset, progress=progress).score(symmetric=symmetric, absolute=absolute)

    # save the score
    score.to_parquet(os.path.join(save_path, f'infoflow_sym-{symmetric}_abs-{absolute}.parquet'))

    # make the result_df
    result_df = {'infoflow': score}
    measures = list(result_df.keys())

    # create dataframe to save results
    results = pd.DataFrame(index=measures,
                           columns=["gross accuracy", "Mean Reciprocal Rank", "pairwise auroc", "Adjusted Rand Index",
                                    "Normalized Mutual Information", "Adjusted Mutual Information", "Homogeneity",
                                    "Completeness", "V-Measure", 'Triplet Accuracy', "Mean Average Precision",
                                    "Normalized Discount Cumulative Gain"],
                           dtype=float)

    # compute the results
    evspi.compute_triplet_accuracy(result_df, measures, results)
    evspi.compute_gross_accuracy(result_df, measures, results)
    evspi.compute_reciprocal_rank(result_df, measures, results)
    evspi.compute_pairwise_auroc(result_df, measures, results)
    evspi.compute_clustering(result_df, measures, results, len(rooms))
    evspi.compute_map(result_df, measures, results)
    evspi.compute_normalized_discounted_gain(result_df, measures, results)
    print(results)

    # print the results
    return
    for col in score.columns:
        print(col)
        print(score[col].nlargest(5))
        print(score[col].nsmallest(5))
        print('\n\n')


def parse_bool(input_arg: str) -> bool:
    input_arg = input_arg.lower()
    if input_arg == 'true':
        return True
    elif input_arg == 'false':
        return False
    else:
        raise ValueError(f'{input_arg} is not True or False.')


"""
Created on 08 Jun 2023
Update on 08 Jun 2023
@author: Hanyu Jin
version: 1.0.0
Citation: X. San Liang, 2015: Normalizing the causality between time series. Phys. Rev. E 92, 022126.
"""

import numpy as np
from scipy.stats import norm
from collections import namedtuple

def causality_est(xx1: np.ndarray, xx2: np.ndarray, n=2, alpha=0.95) -> tuple[float, float, float, float]:
    '''
    Estimate T21, the information transfer from series X2 to series X1
    dt is taken to be 1.
    :param xx1: the series1
    :param xx2: the series2
    :param n: integer >=1, time advance in performing Euler forward
    :return: T21:  info flow from X2 to X1	(Note: Not X1 -> X2!)
             err90: standard error at 90% significance level
             err95: standard error at 95% significance level
             err99: standard error at 99% significance level
    '''

    res = namedtuple('Liang_Causality_Test', ['info', 'h', 'err', 'alpha'])

    dt = 1

    nm = xx1.size

    dx1 = (xx1[n:nm] - xx1[:(nm - n)]) / (n * dt)
    x1 = xx1[:(nm - n)]

    dx2 = (xx2[n:nm] - xx2[:(nm - n)]) / (n * dt)
    x2 = xx2[:(nm - n)]

    N = nm - n

    C = np.cov(x1, x2)

    dC = np.zeros((2, 2))
    dC[0, 0] = np.sum((x1 - np.mean(x1)) * (dx1 - np.mean(dx1)))
    dC[0, 1] = np.sum((x1 - np.mean(x1)) * (dx2 - np.mean(dx2)))
    dC[1, 0] = np.sum((x2 - np.mean(x2)) * (dx1 - np.mean(dx1)))
    dC[1, 1] = np.sum((x2 - np.mean(x2)) * (dx2 - np.mean(dx2)))
    dC = dC / (N - 1)

    C_infty = C

    detc = np.linalg.det(C)

    a11 = C[1, 1] * dC[0, 0] - C[0, 1] * dC[1, 0]
    a12 = -C[0, 1] * dC[0, 0] + C[0, 0] * dC[1, 0]

    a11 = a11 / detc
    a12 = a12 / detc

    f1 = np.mean(dx1) - a11 * np.mean(x1) - a12 * np.mean(x2)

    R1 = dx1 - (f1 + a11 * x1 + a12 * x2)

    Q1 = np.sum(R1 * R1)

    b1 = np.sqrt(Q1 * dt / N)

    NI = np.zeros((4, 4))
    NI[0, 0] = N * dt / (b1 * b1)
    NI[1, 1] = dt / (b1 * b1) * np.sum(x1 * x1)
    NI[2, 2] = dt / (b1 * b1) * np.sum(x2 * x2)
    NI[3, 3] = 3 * dt / (b1 * b1 * b1 * b1) * np.sum(R1 * R1) - N / (b1 * b1)
    NI[0, 1] = dt / (b1 * b1) * np.sum(x1)
    NI[0, 2] = dt / (b1 * b1) * np.sum(x2)
    NI[0, 3] = 2 * dt / (b1 * b1 * b1) * np.sum(R1)
    NI[1, 2] = dt / (b1 * b1) * np.sum(x1 * x2)
    NI[1, 3] = 2 * dt / (b1 * b1 * b1) * np.sum(R1 * x1)
    NI[2, 3] = 2 * dt / (b1 * b1 * b1) * np.sum(R1 * x2)

    NI[1, 0] = NI[0, 1]
    NI[2, 0] = NI[0, 2]
    NI[2, 1] = NI[1, 2]
    NI[3, 0] = NI[0, 3]
    NI[3, 1] = NI[1, 3]
    NI[3, 2] = NI[2, 3]

    invNI = np.linalg.inv(NI)
    var_a12 = invNI[2, 2]

    T21 = C_infty[0, 1] / C_infty[0, 0] * (-C[1, 0] * dC[0, 0] + C[0, 0] * dC[1, 0]) / detc

    dH1_star = a11

    dH1_noise = b1**2 / (2. * C[0, 0])

    Z = np.abs(T21) + np.abs(dH1_star) + np.abs(dH1_noise)

    var_T21 = (C_infty[0, 1] / C_infty[0, 0]) ** 2 * var_a12

    z_alpha = norm.ppf((1 + alpha) / 2)
    err = np.sqrt(var_T21) * z_alpha
    err1 = T21 - err
    err2 = T21 + err
    if err2 < 0 or err1 > 0:
        h = True
    else:
        h = False

    return res(T21, h, err, alpha), Z


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument('-pp', '--parquet_path', type=str, default=r"./")
    _parser.add_argument('-sp', '--save_path', type=str, default='soda')
    _parser.add_argument('-norm', '--normalize', type=parse_bool, default=True)
    _parser.add_argument('-sym', '--symmetric', type=parse_bool, default=False)
    _parser.add_argument('-abs', '--absolute', type=parse_bool, default=True)
    _parser.add_argument('-prog', '--progress', type=parse_bool, default=True)
    _args = _parser.parse_args()
    main(_args.parquet_path, _args.save_path, _args.normalize, _args.symmetric, _args.absolute, _args.progress)
