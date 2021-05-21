import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

# List
# ----

# Save list to CSV file
x = np.arange(0.0, 5.0, 1.0)
np.savetxt('test.csv', x, delimiter=',', fmt='%s')

# Class
# -----


# Check for class inheritance relationships
class BaseClass:
    pass


class SubClass(BaseClass):
    pass


print(issubclass(SubClass, BaseClass))  # True

# Figure
# ------
plt.tight_layout()
plt.savefig(fname='figure.pdf', dpi=300, transparent=True, bbox_inches='tight')

# Graph
# -----


def graph_visualize(G):
    """
    Visualize a graph while setting timestamp as edge labels
    """
    # Define layout
    # pos = nx.random_layout(G)
    # pos = nx.spring_layout(G)
    # pos = nx.shell_layout(G)
    pos = nx.kamada_kawai_layout(G)

    # Create canvas
    plt.figure()

    # Turn off axis
    plt.axis('off')

    # Draw graph
    nx.draw(
        G,
        pos,
        edge_color='grey',
        width=1,
        linewidths=1,
        node_size=300,
        node_color='pink',
        alpha=0.9,
        labels={node: node
                for node in G.nodes()}
    )

    # Draw edge labels
    edge_labels = {(e[0], e[1]): str(e[2]['t']) for e in G.edges.data()}
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_color='black',
    )

    # Show or save
    plt.show()
