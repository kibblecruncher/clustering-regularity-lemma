from importlib.resources import path
from unicodedata import name
import json
import os
import networkx as nx
import numpy as np
import queue as q
from clustering_task import Task

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
        
    def deletePartition(self, vertex:int, direction:str)->None:
        """Deletes the partition file from disk."""
        try:
            os.remove(self.partitionFileName(vertex, direction))
        except OSError:
            pass
    

    
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
        #initialize the graph manager and file manager, and compute the initial partitions and link graphs for all vertices in V2
        self.graph_manager = GraphManager(G)
        self.partition_manager = FileManager("partitions", "graphs")
       
       #initialize the queue for the iterations of the algorithm, and add the initial direction to the queue
        self.q = q.Queue()
        self.q.put("")

        self.eps = eps
        self.irreg_vtx_threshold = eps**5 / 90
        self.dev_vtx_threshold = eps
        self.irreg_vtx_count_threshold = 0.1 #this is a parameter we can tune, it is the threshold for the number of irregular vertices in a partition that we consider to be a problem step
        self.dev_threshold = 0.1 #this is a parameter we can tune, it is the threshold for the total deviation of a partition that we consider to be a problem step
        self.irreg_threshold = eps
        self.pathweight_threshold = eps
       
       


        #make and save the link graphs and partitions for all vertices in V2
        self.V2 = self.graph_manager.getV2()
        for v in self.V2:
            A, B = self.graph_manager.makeLinkPartition(v)
            self.partition_manager.savePartition(v, "", A, B)
            N = self.graph_manager.makeLinkGraph(v)
            self.partition_manager.saveLinkGraph(v, N)


    def compute_path_data(self,dir: str) -> tuple[int,int,float]:
        """Computes the pathweight of the current partition."""
        pathweight = 0
        triangle_count = 0
        for v in self.V2:
            link = self.partition_manager.loadLinkGraph(v)
            success,partition = self.partition_manager.loadPartition(v, dir)
            if not success:
                #if loading fails, skip vertex
                continue
            task = Task(link, partition, self.eps)
            pathweight += task.pathweight
            triangle_count += task.edges

        if pathweight == 0:
            gamma = 0.0
        else:
            gamma = triangle_count / pathweight
        return pathweight, triangle_count, gamma

    def iterate(self):
        """Main loop of the algorithm. Iteratively updates the partition until convergence."""

        #initialize matrix and partition
        while q.not_empty():
            dir = q.get() #get the next direction from the queue

            #compute pathweight and gamma for this partition
            pathweight, triangle_count, gamma = self.compute_path_data(dir)

            irreg_weight = 0.0
            dev_weight = 0.0

            dev_vertices = np.array([False] * len(self.V2)) #mask of vertices in V2 that we will update this step
            irreg_vertices = np.array([False] * len(self.V2)) #mask of vertices in V2 that we will update this step
            for i, v in enumerate(self.V2):
                #load file for vertex v and partition labeled pLabel
                link = self.partition_manager.loadLinkGraph(v)
                success,partition = self.partition_manager.loadPartition(v, dir)
                if not success:
                    #if loading fails, skip vertex
                    continue

                #if succeed, compute the irregular vertices and deviation for this vertex and partition

                task = Task(link, partition, self.eps)
                irreg_v, irreg_count = task.compute_irregular_vertices(gamma)
                dev =task.compute_local_deviation(gamma)
                if irreg_count > self.irreg_vtx_count_threshold * pathweight:
                    irreg_vertices[i] = True
                    #add pathweight of this vertex to the total irregularity of this partition
                    irreg_weight += irreg_count * task.deg_B_v 
                elif dev > self.dev_vtx_threshold:
                    dev_vertices[i] = True
                    #add pathweight of this vertex to the total deviation of this partition
                    dev_weight += task.deg_A_v * task.deg_B_v

            #check if we have a problem step
            if irreg_weight > self.irreg_threshold * pathweight:
                #add the new directions to queue for the next iterations
                q.put(dir + "i0")
                q.put(dir + "i1")
                for i, v in enumerate(self.V2):
                    
                    if irreg_vertices[i]:
                        link = self.partition_manager.loadLinkGraph(v)
                        success,partition = self.partition_manager.loadPartition(v, dir)
                        if not success:
                            #if loading fails, throw an error, since this should not happen
                            raise ValueError(f"Failed to load partition for vertex {v} and direction {dir} when setting irregularity partition")
                        task2 = Task(link, partition, self.eps)
                        irreg_v, irreg_count = task2.compute_irregular_vertices(gamma)
                        self.partition_manager.savePartition(v, dir + "i0", irreg_v, task2.B) #save
                        self.partition_manager.savePartition(v, dir + "i1", ~irreg_v, task2.B) #save
                        self.partition_manager.deletePartition(v, dir) #delete old partition to save space
                
            elif dev_weight > self.dev_threshold * pathweight:
                 #add the new directions to queue for the next iterations
                 q.put(dir + "d0")
                 q.put(dir + "d1")
                 q.put(dir + "d2")
                 q.put(dir + "d3")
                 for i, v in enumerate(self.V2):
                    if dev_vertices[i]:
                        link = self.partition_manager.loadLinkGraph(v)
                        success,partition = self.partition_manager.loadPartition(v, dir)
                        if not success:
                            #if loading fails, throw an error, since this should not happen
                            raise ValueError(f"Failed to load partition for vertex {v} and direction {dir} when setting deviation partition")
                        task3 = Task(link, partition, self.eps)
                        L,R = task3.produce_new_masks(gamma)
                        self.partition_manager.savePartition(v, dir + "d0", L, R) #save
                        self.partition_manager.savePartition(v, dir + "d1", ~L, R) #save
                        self.partition_manager.savePartition(v, dir + "d2", L, ~R) #save
                        self.partition_manager.savePartition(v, dir + "d3", ~L, ~R) #save
                        self.partition_manager.deletePartition(v, dir) #delete old partition to save space
            else:
                #if we are not in a problem step, go to the next value on the queue
                continue
        