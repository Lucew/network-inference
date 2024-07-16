# we use this nice article by
# https://alexandra-zaharia.github.io/posts/kill-subprocess-and-its-children-on-timeout-python/
# WARNING: This only works on LINUX platforms.

import os
import signal
import subprocess
import multiprocessing as mp
import loadData as ld
import yaml
import argparse
import time
import functools
from tqdm import tqdm


def run_calculator(input_tuple: tuple[str, str], parquet_path: str, timeout_s: int):

    # unpack the tuple we received as input
    save_path, subset = input_tuple

    # check whether we are on a posix system
    if os.name != 'posix':
        raise EnvironmentError('This program can only be run on linux machines!')

    # create the command we want to run
    cmd = ['python', 'parallelSPIScript', '--parquet_path', parquet_path, '--save_path', save_path, '--subset', subset]

    # create the file we want to write the output in
    filet = open(os.path.join(save_path, 'output_file'), 'w')

    # start running the process and put a timeout on it
    try:
        p = subprocess.Popen(cmd, start_new_session=True, stderr=filet, stdout=filet)
        p.wait(timeout=timeout_s)

    # check whether we reached a timeout and kill the process
    except subprocess.TimeoutExpired:
        print(f'\n\n\nTimeout for {cmd} ({timeout_s}s) expired', file=filet)
        print('Terminating the whole process group...', file=filet)
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    finally:
        filet.close()


def iterate_config(config_path: str):
    # load the config we want to use for the pyspi run
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # go through the config and write single configs out of there
    for spi_type, spi_names in config.items():
        for spi_name, spi_configs in spi_names.items():

            # create the dict we want to keep
            keep_dict = {key: val for key, val in spi_configs.items() if key != 'config'}

            # check whether we have configs then iterate over those
            spi_configs = spi_configs.get('configs', None)
            if spi_configs is None:
                keep_dict['configs'] = None
                yield keep_dict
            else:
                for config in spi_configs:
                    keep_dict['configs'] = config
                    yield {spi_type: {spi_name: keep_dict}}


def main(path: str, dataset_name: str, sampling_rate: str, timeout_s: int = None):

    # get the dataset
    dataset_name = dataset_name.lower()
    if dataset_name == 'keti':
        dataset, _, _ = ld.load_keti(os.path.join(path, 'KETI'), sample_rate=sampling_rate)
    elif dataset_name == 'soda':
        dataset, _, _ = ld.load_soda(os.path.join(path, 'Soda'), sample_rate=sampling_rate, sensor_count=2)
    else:
        raise ValueError(f'Did not recognize the specified dataset: [{dataset_name}].')
    print(f'For data set {dataset_name}, we have {dataset.shape[1]} signals with {dataset.shape[0]} samples/signal.')

    # save the dataset into the current working directory
    parquet_path = f'{dataset_name}.parquet'
    dataset.to_parquet(parquet_path)

    # make a folder for the current run
    curr_path = f'spi_{int(time.time())}'
    os.mkdir(curr_path)

    # flatten the config into a list of things to do and create the corresponding folders
    config_paths = []
    for idx, config in enumerate(iterate_config('config.yaml')):

        # create the new folder
        result_path = os.path.join(curr_path, str(idx))
        os.mkdir(result_path)

        # save the new config
        config_path = os.path.join(result_path, 'config.yaml')
        with open(config_path, 'w') as filet:
            yaml.dump(config, filet, default_flow_style=False)
        config_paths.append((result_path, config_path))

    # now make the multiprocessing over all the metrics
    spi_computing = functools.partial(run_calculator, parquet_path=parquet_path, timeout_s=timeout_s)
    with mp.Pool(mp.cpu_count()//2) as pool:

        # do this to have a progress bar
        result_iterator = tqdm(pool.imap_unordered(spi_computing, config_paths),
                               desc='Computing SPIs', total=len(config_paths))
        for _ in result_iterator:
            pass


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument('-p', '--path', type=str, default=r"C:\Users\Lucas\Data")
    _parser.add_argument('-d', '--dataset', type=str, default='keti')
    _parser.add_argument('-s', '--sampling_rate', type=str, default='1min')
    _parser.add_argument('-t', '--timeout_s', type=int, default=10)
    _args = _parser.parse_args()
    main(_args.path, _args.dataset, _args.sampling_rate, _args.timeout_s)
