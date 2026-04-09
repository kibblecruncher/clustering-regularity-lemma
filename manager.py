import networkx as nx
import numpy as np
import queue as q

class DTreeManager(object):

    def savePartition(self, vertex:int,direction:str,step:str, A:np.array, B:np.array):
        """Saves the two partitions to disk."""
    
    def loadPartition(self, vertex:int, direction:str):
        """Loads a partition from disk."""


class GraphManager(object):
    def __init__(self, G:nx.Graph):
        self.G = G

    def makeTripartite(self):
        """Generates the tripartite cover of the graph."""

    def makeLinks(self):
        """Generates the link graphs of tripartite cover"""

    def saveLinkGraph(self):
        """Saves the link graph to disk."""

    def loadLinkGraph(self):
        """Loads the link graph from disk."""

class Manager(object):
    def __init__(self, G:nx.Graph, eps:float):
        self.G = G
        self.eps = eps

    

    def iterate(self):
        """Main loop of the algorithm. Iteratively updates the partition until convergence."""

        #initialize matrix and partition
        

        q = q.Queue()
        q.put("0")
        while q.not_empty():
            pLabel = q.get()

            for v in self.G.nodes():
                #load file for vertex v and partition labeled pLabel
                #if loading fails, skip vertex
                #if succeed, store this vertex in relevantVertices

            for v in relevantVertices:
                #compute deviation, irregularity, split type, and new partitions
                #record deviation, irregularity, and split type for this vertex

            #do we have a problem step
            # compare irregularity to 
            # compuare


            
        #compute clustering coefficient and path data


        #pop a partition from the queue

        #load all vertices with this partition

        #compute irregular vertices

        #compute deviations

        #determine decision for this partition

        #update partition

        