import functools
import pandas as pd
from glob import glob
import os
import matplotlib.pyplot as plt
import makeCDID as mkd
import seaborn as sns
import networkx as nx

def load_results(path: str):
    # get all the results into memory
    results = {os.path.split(cp)[-1].split('_', 2)[-1].split('.')[0]: (pd.read_parquet(cp))
               for cp in glob(os.path.join(path, 'result_*.parquet'))}
    # results.update({f'{name}_2': data for name, data in results.items()})
    return results


def make_create_ranks(path: str = './'):

    # load the files
    results = load_results(path)

    # define the metrics to select
    metrics = ["Mean Reciprocal Rank", "pairwise auroc", "Adjusted Rand Index", "Adjusted Mutual Information", "V-Measure", 'Triplet Accuracy', "Mean Average Precision", "Normalized Discount Cumulative Gain"]
    # metrics = ["Mean Reciprocal Rank", "Mean Average Precision", "Normalized Discount Cumulative Gain", "Triplet Accuracy"]
    # metrics = ["Adjusted Rand Index", "V-Measure"]
    # metrics = ["Mean Reciprocal Rank", "Adjusted Rand Index", "V-Measure", 'Triplet Accuracy', "Mean Average Precision", "Normalized Discount Cumulative Gain"]

    # go through the results and compute the means as well as a set of metrics that finished for all
    spis = list(functools.reduce(lambda x, y: x & y, (set(data.index) for data in results.values())))
    print(spis)
    ranks = dict()
    for name, data in results.items():
        if not name.startswith('plant'):
            continue
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
    ax.set_xlim([1, int(df["Performance"].max()*1.2)])

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
            offset = +20 if abs(last_val_odd-value) < 0.5 else 0
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
    best_ten = list(overall_rank.nsmallest(5, columns='Performance').index)
    best_ten.append("cov-sq_EmpiricalCovariance")
    best_ten.append("cov_EmpiricalCovariance")
    best_ten.append("gc_gaussian_k-1_kt-1_l-1_lt-1")
    ranks = ranks.loc[best_ten, :]

    # get the amount of different metrics
    metrics = set(ranks.columns.get_level_values(1))

    # get the datasets in the correct order
    datasets = list(ranks.columns.get_level_values(0))
    datasets = [ele for idx, ele in enumerate(datasets) if idx == 0 or datasets[idx-1] != ele]

    # make the plot
    fig, ax = plt.subplots(layout="constrained", figsize=(12, 5))
    with plt.style.context('ggplot'):
        ranks = ranks.reset_index(inplace=False)
        pd.plotting.parallel_coordinates(ranks, marker='x', linestyle='solid', class_column="index")

        # get the xticks and change them
        new_labels = []
        for ele in ax.get_xticklabels():
            ele.set_text(ele.get_text().split(', ')[1][1:-2])
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
        plt.xticks(rotation=90)
        ax.legend(loc='upper center', bbox_to_anchor=(0.55, 0.25), fancybox=True, shadow=True, ncol=(len(best_ten)+1)//2)
        plt.show()


if __name__ == '__main__':
    _ranks, _results, _overall_rank, _complete_ranks = make_create_ranks()
    make_plot(_overall_rank)
    make_parallel_coordinates(_complete_ranks, _overall_rank)
    make_group_plot(_overall_rank)
