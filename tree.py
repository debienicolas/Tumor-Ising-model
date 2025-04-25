import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

class kTree:
    """
    A k tree is a tree where each node has k-children.

    The class should have the following attributes:
    - k: the degree of the tree
    - depth: the depth of the tree
    - nodes: the nodes of the tree
    - neighbors: a 2D numpy array of shape (num_nodes, k) where neighbors[i, j] is the j-th neighbor of node i and k is the degree of the tree
    """

    def __init__(self, k=2, depth=4):
        self.k = k
        self.depth = depth
        self.nodes = None
        self.neighbors = None

    
    def construct_tree_recursive(self):
        """
        Construct the tree using recursion.
        with the notion of widht. 
        """
        # Calculate the number of nodes
        num_nodes = sum(self.k**d for d in range(self.depth + 1)) * self.breadth
        
        # Initialize nodes with random spin values (-1 or 1)
        self.nodes = np.random.choice([-1, 1], size=num_nodes)
        
        # Initialize neighbors array with -1 (no neighbor)
        self.neighbors = -np.ones((num_nodes, self.k+1), dtype=int)
        
        def construct_tree_recursive_helper(node_id, level):
            if level == self.depth:
                return
            
            for i in range(self.k):
                child_id = node_id * self.k + i + 1
                
                # Skip if we've reached the end of our nodes
                if child_id >= num_nodes:
                    break
                
                # Connect parent to child
                self.neighbors[node_id, i+1] = child_id
                
                # Connect child to parent (first slot in child's neighbors)
                self.neighbors[child_id, 0] = node_id
                
                # Recursively process child
                construct_tree_recursive_helper(child_id, level+1)
        
        # Start recursion from root node (id=0)
        construct_tree_recursive_helper(0, 0)
        
        return self.nodes, self.neighbors
    
    def construct_tree(self):
        """
        Construct the tree using the given depth and degree.
        """
        # Calculate the number of nodes
        num_nodes = 0
        for d in range(self.depth + 1):
            num_nodes += self.k**d
        
        # Initialize nodes with random spin values (-1 or 1)
        self.nodes = np.random.choice([-1, 1], size=num_nodes)
        
        # Initialize neighbors array with -1 (no neighbor)
        # Increase array size to k+1 to accommodate parent and k children
        self.neighbors = -np.ones((num_nodes, self.k+1), dtype=int)
        
        # If tree has only one node, return early
        if num_nodes == 1:
            return self.nodes, self.neighbors
        
        # Build tree level by level using BFS approach
        node_id = 0  # Start with root
        
        for level in range(self.depth):
            # Calculate the range of nodes at this level
            level_start = sum(self.k**i for i in range(level))
            level_end = sum(self.k**i for i in range(level+1))
            
            # Process each node at this level
            for current in range(level_start, level_end):
                # Skip if we've reached the end of our nodes
                if current >= num_nodes:
                    break
                
                # Calculate children indices
                children_start = level_end + (current - level_start) * self.k
                
                # Add up to k children
                for child_offset in range(self.k):
                    child_id = children_start + child_offset
                    
                    # Skip if we've reached the end of our nodes
                    if child_id >= num_nodes:
                        break
                    
                    # Connect parent to child - Using index 1+ for children
                    self.neighbors[current, child_offset+1] = child_id
                    
                    # Connect child to parent (first slot in child's neighbors)
                    self.neighbors[child_id, 0] = current
        
        return self.nodes, self.neighbors
    
    def show_tree(self, show=False, file_path=None):
        """
        Show the tree using NetworkX.
        """
        G = nx.Graph()  # Use undirected graph to avoid cycles
        
        # Add nodes with spin values as attributes
        for i, spin in enumerate(self.nodes):
            G.add_node(i, spin=int(spin))
        
        # Add edges from the neighbors array (avoiding duplicates)
        edges_added = set()
        for i in range(len(self.nodes)):
            for j in range(self.k):
                neighbor = self.neighbors[i, j]
                if neighbor != -1:
                    # Only add edge if we haven't seen it before
                    edge = tuple(sorted([i, neighbor]))
                    if edge not in edges_added:
                        G.add_edge(i, neighbor)
                        edges_added.add(edge)
                
        pos = graphviz_layout(G, prog="twopi") # circo or twopi
        # Draw nodes with colors based on spin values
        node_colors = ['blue' if spin == 1 else 'red' for spin in self.nodes]
        
        # Draw the network
        plt.figure(figsize=(10, 10))
        nx.draw(G, pos,
                node_color=node_colors,
                node_size=500,
                with_labels=True,
                font_color='white',
                font_weight='bold')
        
        # Add a legend
        blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                                markersize=10, label='Spin = +1')
        red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                                markersize=10, label='Spin = -1')
        plt.legend(handles=[blue_patch, red_patch], loc='upper right')
        
        plt.title(f"k Tree (k={self.k}, depth={self.depth})")
        if file_path:
            plt.savefig(file_path)
        if show:
            plt.show()
        

        




# Example usage
if __name__ == "__main__":
    
    tree = kTree(k=2, depth=3)
    nodes, neighbors = tree.construct_tree_recursive()
    print(nodes.shape)
    print(neighbors.shape)
    print(neighbors)
    tree.show_tree(show=True)
    
    