from importlib.resources import path
from unicodedata import name
import json
import os
import networkx as nx
import numpy as np
import queue as q

class FileManager(object):

    def __init__(self,partition_dir:str,graph_dir:str):
        # Create the target directories if they don't exist
        os.makedirs(partition_dir, exist_ok=True)
        self.target_dir = partition_dir
        os.makedirs(graph_dir, exist_ok=True)
        self.graph_dir = graph_dir

    def partitionFileName(self, vertex:int, direction:str)->str:
        """Generates a file name for the given vertex, direction, and step."""
        return os.path.join(self.target_dir, f"partition_{vertex}_{direction}.json")

    def savePartition(self, vertex:int,direction:str, A:np.array, B:np.array)->None:
        """Saves the two partitions to disk."""
        j = json.dumps({"vtx":vertex,"dir":direction,"A":A.tolist(),"B":B.tolist()})
        with open(self.partitionFileName(vertex, direction), "w", encoding="utf-8") as f:            
            f.write(j)

    
    def loadPartition(self, vertex:int, direction:str)->tuple[bool, dict]:
        """Loads the partition from disk."""
        try:
            with open(self.partitionFileName(vertex, direction), "r", encoding="utf-8") as f:
                data = f.read()  # or json.load(f), whatever format you need
        except OSError:
            return False, None
        else:
            return True, data
    

    
    def graphFileName(self, vertex:int)->str:
        """Generates a file name for the given vertex."""
        return os.path.join(self.graph_dir, f"linkGraph_{vertex}.json")

    def saveLinkGraph(self, vertex:int, G:nx.Graph)->None:
        """Saves the link graph to disk."""
        json_data = nx.node_link_data(G)    
        with open(self.graphFileName(vertex), "w", encoding="utf-8") as f:            
            json.dump(json_data, f) 


    def loadLinkGraph(self, v:int)->nx.Graph:
        """Loads the link graph from disk."""
        file_path = self.graphFileName(v)
        try:
            # Load the JSON data from the file
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except OSError:
            return False, None
        else:
            return True, nx.node_link_graph(data)


class GraphManager(object):
    def __init__(self, G:nx.Graph):
        self.H = self.makeTripartite(G)
        self.makeLinks()
        

    def makeTripartite(self,G:nx.Graph)->nx.Graph:
        """Generates the tripartite cover of the graph."""
        H = nx.Graph()
        V1 = [(node, 0) for node in G]
        V2 = [(node, 1) for node in G]
        V3 = [(node, 2) for node in G]

        H.add_nodes_from(V1, part=0)
        H.add_nodes_from(V2, part=1)
        H.add_nodes_from(V3, part=2)
        for i, j in [(0, 1), (1, 2), (2, 0),(1, 0), (2, 1), (0, 2)]:
            H.add_edges_from(((u, i), (v, j)) for u, v in G.edges())
        return H
        
    def getV2(self)->list[int]:
        """Returns the vertices in the second part of the tripartite cover."""
        return [n for n, d in self.H.nodes(data=True) if d.get("part") == 1]    


    def makeLinkGraph(self, vertex:int)->nx.Graph:
        """Generates the link graph of the given vertex."""
        neighbors = list(self.H.neighbors((vertex, 1)))
        N = self.H.subgraph(neighbors).copy()
        return N
    
    def makeLinkPartition(self, vertex:int)->tuple[np.array, np.array]:
        """Generates the link partition of the given vertex."""
        N = self.makeLinkGraph(vertex)
        A = np.array([n for n, d in N.nodes(data=True) if d.get("part") == 0])
        B = np.array([n for n, d in N.nodes(data=True) if d.get("part") == 2])
        return A, B




    


class Manager(object):
    def __init__(self, G:nx.Graph, eps:float):
        self.eps = eps
        self.graph_manager = GraphManager(G)
        self.partition_manager = FileManager("partitions", "graphs")
        self.q = q.Queue()
        self.q.put("")


        #make and save the link graphs and partitions for all vertices in V2
        self.V2 = self.graph_manager.getV2()
        for v in self.V2:
            A, B = self.graph_manager.makeLinkPartition(v)
            self.partition_manager.savePartition(v, "", A, B)
            N = self.graph_manager.makeLinkGraph(v)
            self.partition_manager.saveLinkGraph(v, N)


    def launchThread(self, link:nx.Graph, partition:tuple[np.array, np.array], v:int, dir:str)->None:
        raise NotImplementedError

    def iterate(self):
        """Main loop of the algorithm. Iteratively updates the partition until convergence."""

        #initialize matrix and partition
        while q.not_empty():
            dir = q.get()

            relevantVertices = np.array([False] * len(self.V2)) #mask of vertices in V2 that we will update this step
            for i, v in enumerate(self.V2):
                #load file for vertex v and partition labeled pLabel
                link = self.partition_manager.loadLinkGraph(v)
                success,partition = self.partition_manager.loadPartition(v, dir)
                if not success:
                    #if loading fails, skip vertex
                    continue

                #if succeed, store this vertex in relevantVertices
                relevantVertices[i] = True
                self.launchThread(link, partition, v, dir) #launch a thread to compute the deviation, irregularity, split type, and new partitions for this vertex


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