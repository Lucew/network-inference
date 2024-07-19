import pandas as pd
import os
from glob import glob
from tqdm import tqdm
import functools
import multiprocessing as mp


def load_data_from_folder(path: str) -> dict[str: pd.DataFrame]:

    # find all the csv files in the folder
    files = glob(os.path.join(path, '*.parquet'))

    # load them into a dictionary
    files = {os.path.split(os.path.splitext(file)[0])[-1]: pd.read_parquet(file) for file in files}

    # convert the index into time stamps
    start_timestamp = pd.to_datetime(0)
    end_timestamp = pd.Timestamp.now()
    for file in files.values():
        start_timestamp = max(start_timestamp, file.index.min())
        end_timestamp = min(end_timestamp, file.index.max())

    # sanity check the folders
    assert len(files) in {93, 94, 156, 157}, f'Folder {path} has {len(files)}.'
    return path, files, start_timestamp, end_timestamp


def load_plant(path: str, sample_rate: str):

    # check the sample rate as we only ever expect one (data is already sampled)
    if sample_rate != '10min':
        raise ValueError(f'We only accept 10min for sample rate. You gave: {sample_rate}.')

    # find all the folders in the directory
    folders = glob(os.path.join(path, f'*{os.path.sep}'))

    # make the wrapper for the tqdm progress bar
    wrapper = functools.partial(tqdm, desc=f'Loading Data', total=len(folders))

    # go through all the folders and load the files include a loading bar
    start_timestamp = pd.to_datetime(0)
    end_timestamp = pd.Timestamp.now()
    data = dict()
    with mp.Pool(min(mp.cpu_count() // 2, 3)) as pool:
        for path, files, start_date, end_date in wrapper(pool.imap_unordered(load_data_from_folder, folders)):
            data[os.path.split(os.path.split(path)[0])[-1]] = files
            start_timestamp = max(start_timestamp, start_date)
            end_timestamp = min(end_timestamp, end_date)

    # check that all sensors have the same length
    length = set(df.shape[0] for room in tqdm(data.values(), desc='Check length') for df in room.values())
    start = set((df.index.min(), df.index.max()) for room in tqdm(data.values(), desc='Check index')
                for df in room.values())
    assert len(length) == 1, 'Length is different for some values.'
    assert len(start) == 1, 'Index is different for some values.'
    length = length.pop()
    start = start.pop()

    # combine into one dataframe
    reformated = []
    for room, room_data in tqdm(data.items(), desc='Rename and Reformat'):
        for sensor, df in room_data.items():
            df = df.rename(columns={'value': f'{room}_{sensor}'})
            reformated.append(df)

    # put into one dataframe
    df = pd.concat(reformated, axis=1)

    # check whether there are any NaN
    assert not df.isna().any(axis=1).any(), 'There are NaN values.'

    # make information print
    print(f'Loaded and resampled all with sampling rate {sample_rate} signals from {start} with length {length}.')
    return df, length, start


if __name__ == "__main__":
    _path = r"C:\Users\Lucas\Data\Plant_@1"
    load_plant(_path, sample_rate='10min')
