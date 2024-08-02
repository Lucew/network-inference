import graphviz as gv
import evaluateSPI as evspi
import rankSPI as rspi
import os
import functools
import collections
import dot2tex


def main():

    # get the category information
    _, categories = evspi.parse_spi_information(os.path.join('..', 'distance_or_similarity.txt'))

    # reverse the categories so we can make lookups
    reversed_categories = {specifier: category for category, specifiers in categories.items() for specifier in specifiers}

    # load the results
    results = rspi.load_results('./')
    spis = list(functools.reduce(lambda x, y: x & y, (set(data.index) for data in results.values())))
    categories = {name: [] for name in categories.keys()}
    for spi in spis:
        spi = spi.split('_', 1)[0].split('-')[0]
        categories[reversed_categories[spi]].append(spi)

    # make a graph
    dot = gv.Digraph('spis', comment='The different spis')
    dot.engine = 'dot'

    # add the first node as the root
    root = ('root', f'Terminated\nRelationship Measures: {len(spis)}')
    dot.node(*root, fontsize='5', width="1")

    # get the first four categories
    cat3 = set(sorted(categories.keys())[:3])

    # add the first level categories
    for category, items in categories.items():
        cat_name = "\n".join(category.split(" "))
        dot.node(category, label=f'{cat_name}: {len(items)}', fontsize='5', width="0.01")
        if category in cat3:
            dot.edge(category, 'root', minlen='1')
        else:
            dot.edge('root', category, minlen='1')
        item_cn = collections.Counter(items)
        for item in set(items):
            dot.node(item, label=f'{item}: {item_cn[item]}', fontsize='5', width="0.01")
            if category in cat3:
                dot.edge(item, category, minlen='1')
            else:
                dot.edge(category, item, minlen='1')

    # render and write the tex file
    dot.render(directory='doctest-output').replace('\\', '/')
    with open('graph.tex', 'w') as filet:

        # write the main file into the code
        filet.write(f'% !TeX root = main.tex\n')

        # create the string we want to modify
        texcode = dot2tex.dot2tex(dot.source, format='tikz', figonly=True, autosize=True, figpreamble='\large')

        # rewrite some of the arrows
        texcode = "\n".join(line.replace('[->]', '[<-]') if any(cat in line for cat in cat3) else line.replace(',]', ', scale=0.95]') if line.startswith(r'\begin{tikz') else line for line in texcode.split('\n'))
        filet.write(texcode)


if __name__ == '__main__':
    main()
