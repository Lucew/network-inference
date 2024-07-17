import pandas as pd
import os
from glob import glob
from tqdm import tqdm
import functools
import multiprocessing as mp
import loadData as ld


def prepare_rotary_data(path: str):

    # create the path to the file we want to save
    data_path = os.path.join(path, "all data from database (exported from KNIME workflow).csv")

    # check whether the file already exists
    target_path = os.path.join(path, "rotary_data.parquet")
    if os.path.isfile(target_path):
        print(f"Prepared rotary data already existing at [{target_path}].")
        return

    # load the data and correct some known imprecise time stamps
    df = pd.read_csv(data_path, sep=';', header=0)
    df.loc[1147115, "TIMESTAMP"] = f'{df.loc[1147115, "TIMESTAMP"]}.000000'
    df.loc[3327757, "TIMESTAMP"] = f'{df.loc[3327757, "TIMESTAMP"]}.000000'
    df.loc[3433369, "TIMESTAMP"] = f'{df.loc[3433369, "TIMESTAMP"]}.000000'

    # filter the machines whose data is clean and complete
    with open(r"C:\Users\Lucas\Data\RotaryMachines\allowed_machines.txt", 'r') as filet:
        allowed_machines = {int(ele) for ele in filet.readlines()}
    df = df.loc[df['mcmachine'].isin(allowed_machines), :]

    # filter the signals for the interesting ones
    allowed_alias = {'MachineStatus', 'SlipwayTemperatureAct', 'SpeedAct'}
    df = df.loc[df['alias'].isin(allowed_alias), :]

    # convert the time columns
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])

    # make a quick sanity check that for all machines the same signals exist
    signal_set = None
    for machine in allowed_machines:
        if signal_set is None:
            signal_set = set(df[df['mcmachine'] == machine]["alias"].unique())
        else:
            assert len(signal_set ^ set(
                df[df['mcmachine'] == machine]["alias"].unique())) == 0, f'There are differences for machine {machine}.'

    # define a short time span so the data is appropriately large
    start = pd.to_datetime("2023-08-15 00:00:00")
    end = pd.to_datetime("2023-10-15 00:00:00")
    df = df[(df['TIMESTAMP'] >= start) & (df['TIMESTAMP'] <= end)]

    # write the output to the folder
    df.to_parquet(target_path)


def separate_rotary_data(path: str):

    # load the data into memory
    flat_df = pd.read_parquet(os.path.join(path, "rotary_data.parquet"))

    # go through all the machines
    result_path = os.path.join(path, "separated_data")

    # check whether we already have it
    if os.path.isdir(result_path):
        print(f"The separated data already exists at: [{result_path}].")
        return
    os.mkdir(result_path)

    # go through the machines and make the separation
    signals = flat_df['alias'].unique()
    for idx, machine in enumerate(flat_df['mcmachine'].unique()):

        # make a directory for the machine
        dir_path = os.path.join(result_path, f'machine_{idx}')
        os.mkdir(dir_path)

        # go through the signals
        for signal in signals:

            # filter the signal and drop the filtered columns
            machine_signal_df = flat_df[(flat_df['mcmachine'] == machine) & (flat_df['alias'] == signal)]
            machine_signal_df = machine_signal_df[['value', 'TIMESTAMP']]
            machine_signal_df = machine_signal_df.set_index('TIMESTAMP')

            # save the separated file
            file_path = os.path.join(dir_path, f'{signal.lower()}.parquet')
            machine_signal_df.to_parquet(file_path)


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
    assert len(files) == 3, f'Folder {path} has {len(files)}.'
    return path, files, start_timestamp, end_timestamp


def load_rotary(path: str, sample_rate: str):

    # find all the folders in the directory
    folders = glob(os.path.join(path, 'separated_data', f'*{os.path.sep}'))

    # make the wrapper for the tqdm progress bar
    wrapper = functools.partial(tqdm, desc=f'Loading Data', total=len(folders))

    # go through all the folders and load the files include a loading bar
    start_timestamp = pd.to_datetime(0)
    end_timestamp = pd.Timestamp.now()
    data = dict()
    with mp.Pool(mp.cpu_count() // 2) as pool:
        for path, files, start_date, end_date in wrapper(pool.imap_unordered(load_data_from_folder, folders)):
            data[os.path.split(os.path.split(path)[0])[-1]] = files
            start_timestamp = max(start_timestamp, start_date)
            end_timestamp = min(end_timestamp, end_date)

    # convert the maximum timestamps
    start_timestamp = start_timestamp.ceil(sample_rate)
    end_timestamp = end_timestamp.floor(sample_rate)

    # make the wrapper for the tqdm progress bar
    wrapper = functools.partial(tqdm, desc=f'Resampling Data', total=len(data))

    # make a function we can work with for resampling
    resampler = functools.partial(ld.aggregate_room, start_date=start_timestamp, end_date=end_timestamp,
                                  sample_rate=sample_rate)
    with mp.Pool(mp.cpu_count() // 2) as pool:
        data = {room: room_data for room, room_data in wrapper(pool.imap_unordered(resampler, data.items()))}

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
    _path = r"C:\Users\Lucas\Data\Rotary"
    prepare_rotary_data(_path)
    separate_rotary_data(_path)
    load_rotary(_path, sample_rate='5min')
