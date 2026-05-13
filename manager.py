from importlib.resources import path
from typing import Any, Tuple, List
from unicodedata import name
import json
import os
import networkx as nx
import numpy as np
import queue as q

from numpy._typing._array_like import NDArray
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

    
    def loadPartition(self, vertex:int, direction:str)->Tuple[bool, dict]:
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
    

    def deleteAllPartitions(self):
        """deletes all partitions in the partition directory"""
        for filename in os.listdir(self.target_dir):
            file_path = os.path.join(self.target_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except OSError:
                pass   

    def deleteAllLinkGraphs(self):
        """deletes all link graphs to clean up space"""
        for filename in os.listdir(self.graph_dir):
            file_path = os.path.join(self.graph_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
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
        
    def getV2(self)->List[int]:
        """Returns the vertices in the second part of the tripartite cover."""
        return [n[0] for n, d in self.H.nodes(data=True) if d.get("part") == 1]    


    def makeLinkGraph(self, vertex:int)->nx.Graph:
        """Generates the link graph of the given vertex."""
        neighbors = list(self.H.neighbors((vertex, 1)))
        N = self.H.subgraph(neighbors).copy()
        return N
    
    def linkGraphIndex(self,vertex:int):
        """Maps neighbors of vertex to the edge index in the link graph"""
        neighbors = list(self.H.neighbors((vertex, 1)))
        N = self.H.subgraph(neighbors).copy()
        return {n: i for i, n in enumerate(N.nodes())}
    
    def makeLinkPartition(self, vertex:int)->Tuple[np.ndarray, np.ndarray]:
        """Generates the link partition of the given vertex."""
        N = self.makeLinkGraph(vertex)
        A = np.array([n for n, d in N.nodes(data=True) if d.get("part") == 0])
        B = np.array([n for n, d in N.nodes(data=True) if d.get("part") == 2])
        return A, B


class Manager(object):
    def __init__(self, 
                 G:nx.Graph,
                 eps:float,
                 irreg_vtx_threshold:float,
                 dev_vtx_threshold:float,
                 irreg_vtx_count_threshold:float,
                 dev_threshold:float,
                 irreg_threshold:float,
                 clustering_threshold:float,
                 max_depth:int=float('inf')
                ) -> None:
        #initialize the graph manager and file manager, and compute the initial partitions and link graphs for all vertices in V2
        self.graph_manager = GraphManager(G)
        self.partition_manager = FileManager("partitions", "graphs")
       
        self.max_depth = max_depth
        #parameters for the algorithm, which can be tuned for better performance
        self.eps = eps
        self.irreg_vtx_threshold = irreg_vtx_threshold #threshold which defines an irregular vertex
        self.dev_vtx_threshold = dev_vtx_threshold #threshold for local deviation 
        self.irreg_vtx_count_threshold = irreg_vtx_count_threshold #threshold for irregular vertex count at a vertex
        self.dev_threshold = dev_threshold #threshold for pathweight of deviation vertices
        self.irreg_threshold = irreg_threshold #threshold for pathweight irregular vertex set
        self.clustering_threshold = clustering_threshold #threshold for smallest clustering coefficient
       
       
    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        """executes the main algorithm, and returns the final partition labels for the edges in E12 and E23."""
        #make and save the link graphs and partitions for all vertices in V2
        self.V2 = self.graph_manager.getV2()
        for v in self.V2:
            A, B = self.graph_manager.makeLinkPartition(v)
            self.partition_manager.savePartition(v, "", A, B)
            N = self.graph_manager.makeLinkGraph(v)
            self.partition_manager.saveLinkGraph(v, N)

        #initialize the queue for the iterations of the algorithm, and add the initial direction to the queue
        self.q = q.Queue()
        self.q.put("")
        
        #generate the final set of directions to consider
        dirs = self.iterate()

        #now generate global bitmasks from link graph data
        masks_A = []
        masks_B = []
       
        for dir in dirs:
            mask_A, mask_B = self.assemble_partition(dir)
            masks_A.append(mask_A)
            masks_B.append(mask_B)
        
        #compute partition labels
        self.partition_labels_A = self.partitionLabels(masks_A)
        self.partition_labels_B = self.partitionLabels(masks_B)

        #cleanup step
        #delete intermediate data in partition and graph directories
        self.partition_manager.deleteAllPartitions()
        self.partition_manager.deleteAllLinkGraphs()

        return (self.partition_labels_A, self.partition_labels_B)

    def assemble_partition(self, dir:str)->Tuple[np.ndarray, np.ndarray]:
        """Assembles the partition for all vertices in V2 for the given direction."""
        part_A_iter = []
        part_B_iter = []
        for v in self.V2:
            success, partition_str = self.partition_manager.loadPartition(v, dir)
            if success:
                partition_dict = json.loads(partition_str)
                A = np.array(partition_dict["A"])
                B = np.array(partition_dict["B"])
                part_A_iter.append(A)
                part_B_iter.append(B)

        part_A = np.concatenate(part_A_iter)
        part_B = np.concatenate(part_B_iter)
        return part_A, part_B
    
    def partitionLabels(self,bitmasks:List[np.ndarray]) -> np.ndarray:
        num_indices = bitmasks[0].shape[0]
        labels = np.zeros(num_indices, dtype=int)

        for i, bitmask in enumerate(bitmasks):
            labels[bitmask] = i
        
        return labels

    def compute_direction_code_length(self, direction: str) -> int:
        """Computes the length of a direction code."""
        return len(direction)
    
    def compute_path_data(self,dir: str) -> Tuple[int,int,float]:
        """Computes the pathweight of the current partition."""
        pathweight = 0
        triangle_count = 0
        for v in self.V2:
            link = self.partition_manager.loadLinkGraph(v)
            success,partition = self.partition_manager.loadPartition(v, dir)
            if not success:
                #if loading fails, skip vertex
                continue
            # Parse the JSON partition string into A, B arrays
            partition_dict = json.loads(partition)
            A = np.array(partition_dict["A"])
            B = np.array(partition_dict["B"])
            task = Task(link, (A, B), self.eps)
            pathweight += task.pathweight
            triangle_count += task.edges

        if pathweight == 0:
            gamma = 0.0
        else:
            gamma = triangle_count / pathweight
        return pathweight, triangle_count, gamma

    def iterate(self) -> List[str]: 
        """Main loop of the algorithm. Iteratively updates the partition until convergence.
        
        Uses self.max_depth to enforce maximum allowed length of direction code.
        If exceeded, cleans up and raises an error.
        """

        #final output list of good directions
        out = []

        #initialize matrix and partition
        while not self.q.empty():
            dir = self.q.get() #get the next direction from the queue
            
            #check if direction code depth exceeds maximum
            if self.compute_direction_code_length(dir) > self.max_depth:
                #clean up all auxiliary files and report error
                self.partition_manager.deleteAllPartitions()
                self.partition_manager.deleteAllLinkGraphs()
                raise ValueError(f"Direction code '{dir}' (length {self.compute_direction_code_length(dir)}) exceeds maximum depth {self.max_depth}")

            #compute pathweight and gamma for this partition
            pathweight, triangle_count, gamma = self.compute_path_data(dir)

            if gamma < self.clustering_threshold:
                #if gamma is small, we are done with this partition, save partition in output list
                out.append(dir)
                continue

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
                partition_dict = json.loads(partition)
                A = np.array(partition_dict["A"])
                B = np.array(partition_dict["B"])

                task = Task(link, (A, B), self.eps)
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
                self.q.put(dir + "i0")
                self.q.put(dir + "i1")
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
                 self.q.put(dir + "d0")
                 self.q.put(dir + "d1")
                 self.q.put(dir + "d2")
                 self.q.put(dir + "d3")
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
                #if we are not in a problem step, save the partition in the output list
                out.append(dir)
        return out

        