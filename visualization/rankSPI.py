import functools
import pandas as pd
from glob import glob
import os
import matplotlib.pyplot as plt
import makeCDID as mkd
import seaborn as sns


def load_results(path: str):
    # get all the results into memory
    results = {os.path.split(cp)[-1].split('_', 2)[-1].split('.')[0]: (pd.read_parquet(cp))
               for cp in glob(os.path.join(path, '..', 'result_*.parquet'))}
    # results.update({f'{name}_2': data for name, data in results.items()})
    return results


def make_create_ranks(path: str = './', metric_subset: str = 'all', dataset_subset: str = 'all'):

    # load the files
    results = load_results(path)

    # define the metrics to select
    if metric_subset == 'ir':
        metrics = ["Mean Reciprocal Rank", "Mean Average Precision", "Normalized Discount Cumulative Gain", "Triplet Accuracy"]
    elif metric_subset == 'cluster':
        metrics = ["Adjusted Rand Index", "Adjusted Mutual Information", "V-Measure"]
    elif metric_subset == 'all':
        metrics = ["Mean Reciprocal Rank", "Mean Average Precision", "Normalized Discount Cumulative Gain", "Triplet Accuracy", "Adjusted Rand Index", "Adjusted Mutual Information", "V-Measure"]
    else:
        raise ValueError(f'Metric subset {metric_subset} is not valid.')

    # only keep selected datasets
    if dataset_subset == 'all':
        pass
    elif dataset_subset == 'building':
        buildings_datasets = {'soda', 'keti'}
        results = {name: data for name, data in results.items() if name in buildings_datasets}
    elif dataset_subset == 'plant':
        results = {name: data for name, data in results.items() if name.startswith('plant')}
    else:
        raise ValueError(f'Dataset subset {metric_subset} is not valid.')

    # go through the results and compute the means as well as a set of metrics that finished for all
    spis = list(functools.reduce(lambda x, y: x & y, (set(data.index) for data in results.values())))
    print(len(spis), spis)
    ranks = dict()
    for name, data in results.items():
        # get the average ranks over the different metrics for each dataset
        ranks[name] = data.loc[spis, metrics].rank(ascending=False).mean(axis=1)

    # get the metrics that are always in the best 20
    # in_best_20 = list(functools.reduce(lambda x, y: x & y, (set(data.nsmallest(30).index) for data in ranks.values())))

    # filter all the dataframes for these metrics
    # ranks = {name: df.loc[in_best_20] for name, df in ranks.items()}

    # create a dataframe that contains all the plots

    # now create the set of metrics that is the same for all of them
    ranks = pd.DataFrame(ranks)
    weighted_ranks = ranks.copy()
    weights = {'keti': 235, 'soda': 404, 'plant1': 405, 'plant2': 406, 'plant3': 406, 'plant4': 405, 'rotary': 42}
    for col in weighted_ranks:
        weighted_ranks[col] = weighted_ranks[col]*weights[col]

    # compute the mean over all datasets
    # overall_rank = weighted_ranks.sum(axis=1)/sum(weights.values())
    overall_rank = ranks.mean(axis=1)
    overall_rank = pd.DataFrame(overall_rank, columns=["Performance"])

    # create a dataframe that contains the ranks for each dataset and metric
    complete_ranks = dict()
    for dataset, information in results.items():
        information = information.rank(ascending=False)
        information = information.loc[spis, metrics]
        for metric in information.columns:
            complete_ranks[(dataset, metric)] = information[metric]
        complete_ranks[(dataset, 'Mean Rank')] = information.mean(axis=1)
    complete_ranks = pd.DataFrame(complete_ranks)

    # sort the datasets by their inverted string
    column_order = sorted(set(complete_ranks.columns.get_level_values(0)), key=lambda x: x[::-1])
    complete_ranks = complete_ranks[column_order]

    return ranks, results, overall_rank, complete_ranks


def create_cdi_diagram(ranks: pd.DataFrame):

    # we need to melt the ranks
    melted_ranks = pd.melt(ranks.reset_index(), id_vars='index', value_vars=ranks.columns)
    melted_ranks.rename(columns={'index': 'classifier_name', 'variable': 'dataset_name', 'value': 'accuracy'}, inplace=True)
    mkd.draw_cd_diagram(df_perf=melted_ranks, title='Ranks', labels=True)
    plt.show()


def make_group_plot(overall_rank: pd.DataFrame):
    overall_rank['group'] = [ele.split('_', 1)[0] for ele in overall_rank.index]
    overall_rank = overall_rank.groupby('group').min()
    make_plot(overall_rank, 10)


def make_plot(df: pd.DataFrame, amount: int = 5):

    # compute the mean
    df.sort_values(inplace=True, by="Performance")
    if "cov-sq_EmpiricalCovariance" in df.index and "cov_EmpiricalCovariance" in df.index:
        df = pd.concat((df.nsmallest(amount, columns="Performance"), df.loc[["cov-sq_EmpiricalCovariance"]], df.loc[["cov_EmpiricalCovariance"]]))
    else:
        df = df.nsmallest(amount, columns="Performance")

    # Plotting
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.set_xlim([1, int(df["Performance"].max()*(1.2 if amount < 10 else 1.05))])

    # Add a horizontal arrow
    plt.annotate(
        '',
        xy=(-0.1, 0.5), xycoords='axes fraction', xytext=(1.1, 0.5),
        arrowprops=dict(arrowstyle='<|-', lw=1.5, color='black'),
        zorder=1
    )

    plt.scatter(df['Performance'], [1] * df.shape[0], marker='o', zorder=2, edgecolors='black', color='white')

    # Adding labels
    last_val_even = -5
    last_val_odd = -5
    for idx, value in enumerate(df['Performance']):
        if idx & 1:
            va_text = 'bottom'
            ha_text = 'left'
            text = f'-  {df.index[idx]} - {value: 0.1f}'
            offset = +40 if abs(last_val_even-value) < 0.5 else 0
            last_val_even = value
        else:
            va_text = 'top'
            ha_text = 'right'
            text = f'{value: 0.1f} - {df.index[idx]}        '
            offset = -30 if abs(last_val_odd-value) < 0.5 else 0
            last_val_odd = value
        print(45+offset, text)
        plt.text(value, 1, text, ha=ha_text, va=va_text, rotation=45+offset, fontsize='large')

    # Customizing the plot
    plt.xlabel('Average Rank', x=1.025, fontsize='large', fontweight='bold')
    plt.yticks([])  # Hide the y-axis
    plt.grid(False)  # Remove the grid

    # Remove all spines
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')

    # Invert the x-axis
    plt.gca().invert_xaxis()

    # Adjust the bottom spine to be at y=0
    ax.spines['bottom'].set_position(('data', 1))

    # Show plot
    plt.tight_layout()
    plt.show()


def make_parallel_coordinates(ranks: pd.DataFrame, overall_rank: pd.DataFrame):

    # get the best 10
    best_ten = list(overall_rank.nsmallest(7, columns='Performance').index)
    ranks = ranks.loc[best_ten, :]

    # get the amount of different metrics
    metrics = set(ranks.columns.get_level_values(1))

    # get the datasets in the correct order
    datasets = list(ranks.columns.get_level_values(0))
    datasets = [ele for idx, ele in enumerate(datasets) if idx == 0 or datasets[idx-1] != ele]

    # make the plot
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    with plt.style.context('ggplot'):

        # reset the index and melt the dataframe
        ranks = ranks.reset_index(inplace=False)
        melted_df = ranks.melt(id_vars=[('index', '')], value_vars=list(ranks.columns)[1:])
        melted_df = melted_df.rename(columns={('index', ''): "spi"})
        melted_df['Type'] = [ele.split('_', 1)[0] for ele in melted_df["spi"]]
        melted_df["variable_0"] = melted_df["variable_0"] + ', ' + melted_df["variable_1"]
        melted_df = melted_df.drop(columns="variable_1")
        melted_df = melted_df.rename(columns={"variable_0": "Metric"})
        sns.lineplot(melted_df, x='Metric', y='value', hue="spi", style="spi")

        # get the xticks and change them
        new_labels = []
        plt.grid(True)
        for ele in ax.get_xticklabels():
            ele.set_text(ele.get_text().split(', ')[1])
            if ele.get_text() == 'Mean Rank':
                ele.set_fontweight('bold')
            new_labels.append(ele)
        # new_labels = [ele.get_text().split(', ')[1][1:-2] for ele in ax.get_xticklabels()]
        ax.set_xticklabels(new_labels)

        # make multilevel x-axis
        sec = ax.secondary_xaxis(location=1)
        positions = [ele-1 for ele in list(range(len(metrics), len(metrics)*len(datasets)+1, len(metrics)))]
        sec.set_xticks([ele-len(metrics)/2 for ele in positions], labels=datasets)

        # make the gridlines for the datasets bold
        gridlines = ax.xaxis.get_gridlines()
        for pos in positions:
            gridlines[pos].set_color("k")
            gridlines[pos].set_linewidth(2.5)

        # invert the axis and show the plot
        ax.invert_yaxis()
        ax.set_ylabel("Rank")
        plt.xticks(rotation=45, ha='right')
        ax.legend(loc='best', fancybox=True, shadow=True)
        plt.show()


def make_parallel_vertical_coordinates(ranks: pd.DataFrame, overall_rank: pd.DataFrame):
    # get the best 10
    best_ten = list(overall_rank.nsmallest(7, columns='Performance').index)
    ranks = ranks.loc[best_ten, :]

    # get the amount of different metrics
    metrics = set(ranks.columns.get_level_values(1))

    # make the plot
    fig, ax = plt.subplots(figsize=(6, 12), constrained_layout=True)
    with plt.style.context('ggplot'):

        # reset the index and melt the dataframe
        ranks = ranks.reset_index(inplace=False)
        melted_df = ranks.melt(id_vars=[('index', '')], value_vars=list(ranks.columns)[1:])
        melted_df = melted_df.rename(columns={('index', ''): "spi"})
        melted_df['Type'] = [ele.split('_', 1)[0] for ele in melted_df["spi"]]
        melted_df["variable_2"] = melted_df["variable_0"] + ', ' + melted_df["variable_1"]
        melted_df = melted_df.rename(columns={"variable_2": "Metric"})

        # get the datasets in the correct order
        datasets = list(melted_df['variable_0'])
        datasets = [ele for idx, ele in enumerate(datasets) if idx == 0 or datasets[idx - 1] != ele]

        # make some clean-ups
        melted_df.drop(columns=['variable_0', 'variable_1'], inplace=True)

        # make the plot
        sns.lineplot(melted_df, x='value', y='Metric', hue="spi", style="spi", orient='y')

        # get the xticks and change them
        new_labels = []
        plt.grid(True)
        for ele, gridline in zip(ax.get_yticklabels(), ax.yaxis.get_gridlines()):
            ele.set_text(ele.get_text().split(', ')[1])
            if ele.get_text() == 'Mean Rank':
                gridline.set_color("k")
                gridline.set_linewidth(2.5)
                ele.set_fontweight('bold')
            if ele.get_text() in ["Adjusted Rand Index", "Adjusted Mutual Information", "V-Measure"]:
                ele.set_color('#072140')
            elif ele.get_text() == 'Mean Rank':
                pass
            else:
                ele.set_color('#0099FF')
            new_labels.append(ele)
        ax.set_yticklabels(new_labels)
        ax.set_ylabel("")

        # make multilevel x-axis
        sec = ax.secondary_yaxis(location=1)
        positions = [ele - 1 for ele in list(range(len(metrics), len(metrics) * len(datasets) + 1, len(metrics)))]
        sec.set_yticks([ele - len(metrics) / 2 for ele in positions], labels=datasets)

        # invert the axis and show the plot
        ax.invert_xaxis()
        ax.set_xlabel("Rank")
        # plt.xticks(rotation=90)
        # plt.yticks(rotation=90)
        ax.legend(loc='best', fancybox=True, shadow=True)
        plt.show()


def main():
    ranks, results, overall_rank, complete_ranks = make_create_ranks(metric_subset='all', dataset_subset='all')
    make_plot(overall_rank)
    make_parallel_coordinates(complete_ranks, overall_rank)
    make_parallel_vertical_coordinates(complete_ranks, overall_rank)
    make_group_plot(overall_rank)


if __name__ == '__main__':
    main()