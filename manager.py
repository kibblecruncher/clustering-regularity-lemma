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

    def savePartition(self, vertex:int, direction:str, mask_A:np.array, mask_B:np.array, 
                       neighbors_A:List=None, neighbors_B:List=None)->None:
        """Saves the partition with bitmasks and neighbor identifiers to disk.
        
        Args:
            vertex: The vertex index in V2
            direction: The direction string
            mask_A: Bitmask over neighbors in part 0 (V1)
            mask_B: Bitmask over neighbors in part 2 (V3)
            neighbors_A: The actual neighbor nodes in part 0 (optional, for legacy compatibility)
            neighbors_B: The actual neighbor nodes in part 2 (optional, for legacy compatibility)
        """
        # Convert bitmasks to Python lists (ensure numpy types are converted to native Python types)
        try:
            mask_A_arr = np.asarray(mask_A, dtype=bool)
            mask_B_arr = np.asarray(mask_B, dtype=bool)
            mask_A_list = [int(x) for x in mask_A_arr]  # Convert to int (0 or 1) for JSON compatibility
            mask_B_list = [int(x) for x in mask_B_arr]
        except Exception:
            # Fallback if anything goes wrong
            mask_A_list = mask_A.tolist() if hasattr(mask_A, 'tolist') else list(mask_A)
            mask_B_list = mask_B.tolist() if hasattr(mask_B, 'tolist') else list(mask_B)
        
        #if mask_A or mask_B are all False, we do not need to save anything
        if not np.any(mask_A) and not np.any(mask_B):
            return


        # Convert neighbors to lists for JSON serialization
        # Handle both lists and numpy arrays properly
        if neighbors_A is not None:
            if isinstance(neighbors_A, np.ndarray):
                neighbors_A = neighbors_A.tolist()  # Convert numpy array to list first
            # Convert tuples to lists and ensure native Python types
            neighbors_A_list = []
            for n in neighbors_A:
                if isinstance(n, (tuple, list, np.ndarray)):
                    neighbors_A_list.append([int(x) for x in n])
                else:
                    neighbors_A_list.append(int(n))
        else:
            neighbors_A_list = None
            
        if neighbors_B is not None:
            if isinstance(neighbors_B, np.ndarray):
                neighbors_B = neighbors_B.tolist()  # Convert numpy array to list first
            # Convert tuples to lists and ensure native Python types
            neighbors_B_list = []
            for n in neighbors_B:
                if isinstance(n, (tuple, list, np.ndarray)):
                    neighbors_B_list.append([int(x) for x in n])
                else:
                    neighbors_B_list.append(int(n))
        else:
            neighbors_B_list = None
        
        data = {
            "vtx": vertex,
            "dir": direction,
            "mask_A": mask_A_list,
            "mask_B": mask_B_list,
            "neighbors_A": neighbors_A_list,
            "neighbors_B": neighbors_B_list
        }
        j = json.dumps(data)
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
    
    def makeLinkPartition(self, vertex:int)->Tuple[List, List]:
        """Generates the link partition of the given vertex."""
        N = self.makeLinkGraph(vertex)
        A = [n for n, d in N.nodes(data=True) if d.get("part") == 0]
        B = [n for n, d in N.nodes(data=True) if d.get("part") == 2]
        return A, B


class Manager(object):

    @staticmethod
    def _convert_json_to_nodes(json_list: List) -> List:
        """Convert JSON lists back to tuples for node references (e.g., from JSON serialization)."""
        result = []
        for item in json_list:
            if isinstance(item, list):
                # Convert list to tuple for node references
                result.append(tuple(item))
            else:
                result.append(item)
        return result
    
    def _get_edge_lists(self) -> Tuple[List[Tuple], List[Tuple]]:
        """Get all E12 and E23 edges from the tripartite graph, sorted consistently.
        
        Returns:
            (E12_edges, E23_edges) - Lists of edges from part 0-1 and 1-2 respectively
        """
        H = self.graph_manager.H
        E12_edges = []
        E23_edges = []
        
        for u, v in H.edges():
            u_part = H.nodes[u]['part']
            v_part = H.nodes[v]['part']
            
            # Normalize edge direction (smaller part first)
            if u_part > v_part:
                u, v, u_part, v_part = v, u, v_part, u_part
            
            if u_part == 0 and v_part == 1:
                E12_edges.append((u, v))
            elif u_part == 1 and v_part == 2:
                E23_edges.append((u, v))
        
        # Sort for consistency
        E12_edges.sort()
        E23_edges.sort()
        return E12_edges, E23_edges
    
    def _map_vertex_partition_to_edges(self, v: int, mask_A: np.ndarray, mask_B: np.ndarray,
                                       neighbors_A: List, neighbors_B: List) -> Tuple[np.ndarray, np.ndarray]:
        """Map a vertex's local partition to contributions to global E12 and E23 edge partitions.
        
        Args:
            v: The vertex index in V2
            mask_A: Bitmask indicating which A-neighbors are marked
            mask_B: Bitmask indicating which B-neighbors are marked
            neighbors_A: List of neighbor nodes in part 0
            neighbors_B: List of neighbor nodes in part 2
        
        Returns:
            (E12_contribution, E23_contribution) - Boolean arrays indicating marked edges
        """
        E12_edges, E23_edges = self._get_edge_lists()
        
        # Create sets for fast lookup
        E12_set = {e: i for i, e in enumerate(E12_edges)}
        E23_set = {e: i for i, e in enumerate(E23_edges)}
        
        E12_contrib = np.zeros(len(E12_edges), dtype=bool)
        E23_contrib = np.zeros(len(E23_edges), dtype=bool)
        
        v_node = (v, 1)  # The vertex in part 1 (V2)
        
        # For each marked A-neighbor, mark the corresponding E12 edge
        for idx, neighbor_A in enumerate(neighbors_A):
            if mask_A[idx]:
                # Edge from part 0 to part 1
                edge = (neighbor_A, v_node)
                if edge in E12_set:
                    E12_contrib[E12_set[edge]] = True
        
        # For each marked B-neighbor, mark the corresponding E23 edge
        for idx, neighbor_B in enumerate(neighbors_B):
            if mask_B[idx]:
                # Edge from part 1 to part 2
                edge = (v_node, neighbor_B)
                if edge in E23_set:
                    E23_contrib[E23_set[edge]] = True
        
        return E12_contrib, E23_contrib



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
        """Initializes the manager with the given graph and parameters.
            Standard hyperparameters for the algorithm include:
            eps < 1/16 : parameter for the clustering task, which determines the approximation quality
           
            irreg_vtx_threshold <= eps**5/90: threshold for local irregularity of a vertex within a link graph
            irreg_vtx_count_threshold <= eps**(5/2)/9: threshold for the number of irregular vertices at a vertex
            
            irreg_threshold  <= 2*eps**(5/2)/5: threshold for the total irregularity weight of a partition
            
            dev_vtx_threshold  <= eps**2/9 : threshold for local deviation of a vertex
            dev_threshold <= 2*eps**2/5 : threshold for the total deviation weight of a partition
            
            clustering_threshold <= eps : threshold for smallest allowed clustering coefficient of a partition
        
        """
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
       
       
    def _load_partition_with_mask(self, vertex: int, direction: str) -> Tuple[bool, Any]:
        """Load a partition and apply the mask to select neighbors.
        
        Returns:
            Tuple[bool, tuple]: (success, (A, B)) where A and B have masks applied and converted to tuples
        """
        success, partition_str = self.partition_manager.loadPartition(vertex, direction)
        if not success:
            return False, ([], [])
        
        try:
            partition_dict = json.loads(partition_str)
            mask_A = np.array(partition_dict["mask_A"], dtype=bool)
            mask_B = np.array(partition_dict["mask_B"], dtype=bool)
            
            neighbors_A_full = self._convert_json_to_nodes(partition_dict["neighbors_A"])
            neighbors_B_full = self._convert_json_to_nodes(partition_dict["neighbors_B"])
            
            # Apply masks to select subset of neighbors
            neighbors_A_array = np.array(neighbors_A_full)
            neighbors_B_array = np.array(neighbors_B_full)
            
            # Filter by mask and convert back to list of tuples
            if len(neighbors_A_array) > 0:
                A_filtered = neighbors_A_array[mask_A]
                A = [tuple(node) if isinstance(node, (list, np.ndarray)) else node for node in A_filtered]
            else:
                A = []
            
            if len(neighbors_B_array) > 0:
                B_filtered = neighbors_B_array[mask_B]
                B = [tuple(node) if isinstance(node, (list, np.ndarray)) else node for node in B_filtered]
            else:
                B = []
            
            return True, (A, B)
        except Exception as e:
            import traceback
            print(f"Error in _load_partition_with_mask: {e}")
            print(traceback.format_exc())
            return False, ([], [])
    
    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        """executes the main algorithm, and returns the final partition labels for the edges in E12 and E23."""
        #make and save the link graphs and partitions for all vertices in V2
        self.V2 = self.graph_manager.getV2()
        for v in self.V2:
            A, B = self.graph_manager.makeLinkPartition(v)
            # Initial partitions: mark all neighbors as in partition class 0
            mask_A = np.ones(len(A), dtype=bool)
            mask_B = np.ones(len(B), dtype=bool)
            self.partition_manager.savePartition(v, "", mask_A, mask_B, A, B)
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
        """Assembles the global edge partitions for all vertices in V2 for the given direction.
        
        Maps local vertex partitions (bitmasks over neighbors) to global edge partitions
        (bitmasks over E12 and E23 edges in the tripartite graph).
        """
        E12_edges, E23_edges = self._get_edge_lists()
        
        E12_partition = np.zeros(len(E12_edges), dtype=bool)
        E23_partition = np.zeros(len(E23_edges), dtype=bool)
        
        for v in self.V2:
            success, partition_str = self.partition_manager.loadPartition(v, dir)
            if not success:
                continue
                
            partition_dict = json.loads(partition_str)
            
            # Load bitmasks
            mask_A = np.array(partition_dict["mask_A"], dtype=bool)
            mask_B = np.array(partition_dict["mask_B"], dtype=bool)
            
            # Load neighbor identifiers, converting JSON lists to tuples
            neighbors_A = self._convert_json_to_nodes(partition_dict["neighbors_A"])
            neighbors_B = self._convert_json_to_nodes(partition_dict["neighbors_B"])
            
            # Map this vertex's partition to edge contributions
            E12_contrib, E23_contrib = self._map_vertex_partition_to_edges(v, mask_A, mask_B, neighbors_A, neighbors_B)
            
            # Combine with global partition (OR operation)
            E12_partition |= E12_contrib
            E23_partition |= E23_contrib
        
        return E12_partition, E23_partition
    
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
            success, link = self.partition_manager.loadLinkGraph(v)
            success_partition, (A, B) = self._load_partition_with_mask(v, dir)
            if not success or not success_partition:
                #if loading fails, skip vertex
                continue
            task = Task(link, (A, B), self.eps)
            # pathweight is the sum of deg_A_v * deg_B_v over all vertices in v2, so we add len(A)*len(B) for this vertex to the total pathweight, and add the number of edges in the link graph to the triangle count
            pathweight += len(task.A) * len(task.B)
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
        
        Logs detailed information about each partition iteration including gamma values and failure reasons.
        """

        #final output list of good directions
        out = []
        
        #track all directions considered for logging
        self.directions_considered = []
        self.partition_logs = []  # Track detailed logs for each partition

        #initialize matrix and partition
        while not self.q.empty():
            dir = self.q.get() #get the next direction from the queue
            self.directions_considered.append(dir)
            
            #check if direction code depth exceeds maximum
            if self.compute_direction_code_length(dir) > self.max_depth:
                #clean up all auxiliary files and report error
                self.partition_manager.deleteAllPartitions()
                self.partition_manager.deleteAllLinkGraphs()
                raise ValueError(f"Direction code '{dir}' (length {self.compute_direction_code_length(dir)}) exceeds maximum depth {self.max_depth}")

            #compute pathweight and gamma for this partition
            pathweight, triangle_count, gamma = self.compute_path_data(dir)
            
            # Log this partition
            partition_info = {
                'direction': dir,
                'pathweight': int(pathweight) if isinstance(pathweight, (int, np.integer)) else float(pathweight),
                'triangle_count': int(triangle_count) if isinstance(triangle_count, (int, np.integer)) else float(triangle_count),
                'gamma': float(gamma),
                'clustering_threshold': float(self.clustering_threshold),
                'failure_reason': None,
                'irreg_weight': None,
                'dev_weight': None,
            }

            if gamma < self.clustering_threshold:
                #if gamma is small, we are done with this partition, save partition in output list
                partition_info['failure_reason'] = 'PASSED_GAMMA_CHECK'
                out.append(dir)
                self.partition_logs.append(partition_info)
                continue

            irreg_weight = 0.0
            dev_weight = 0.0

            dev_vertices = np.array([False] * len(self.V2)) #mask of vertices in V2 that we will update this step
            irreg_vertices = np.array([False] * len(self.V2)) #mask of vertices in V2 that we will update this step
            for i, v in enumerate(self.V2):
                #load file for vertex v and partition labeled pLabel
                success_g, link = self.partition_manager.loadLinkGraph(v)
                success_p, (A, B) = self._load_partition_with_mask(v, dir)
                if not success_g or not success_p:
                    #if loading fails, skip vertex
                    continue

                #if succeed, compute the irregular vertices and deviation for this vertex and partition
                task = Task(link, (A, B), self.eps)
                irreg_v, irreg_count = task.compute_irregular_vertices(gamma)
                dev =task.compute_local_deviation(gamma)
                if irreg_count > self.irreg_vtx_count_threshold * pathweight:
                    irreg_vertices[i] = True
                    #add pathweight of this vertex to the total irregularity of this partition
                    # Sum over the irregular vertices' pathweights
                    irreg_weight += np.sum(irreg_v * task.pathweight) 
                elif dev > self.dev_vtx_threshold:
                    dev_vertices[i] = True
                    #add pathweight of this vertex to the total deviation of this partition
                    dev_weight += np.sum(task.pathweight)


            # Store weights in partition log
            partition_info['irreg_weight'] = float(irreg_weight)
            partition_info['dev_weight'] = float(dev_weight)
            partition_info['irreg_threshold'] = float(self.irreg_threshold * pathweight)
            partition_info['dev_threshold'] = float(self.dev_threshold * pathweight)

            #check if we have a problem step
            if irreg_weight > self.irreg_threshold * pathweight:
                partition_info['failure_reason'] = 'FAILED_IRREGULARITY_CHECK'
                self.partition_logs.append(partition_info)
                #add the new directions to queue for the next iterations
                self.q.put(dir + "i0")
                self.q.put(dir + "i1")
                for i, v in enumerate(self.V2):
                    
                    if irreg_vertices[i]:
                        success_g, link = self.partition_manager.loadLinkGraph(v)
                        success_p, (A, B) = self._load_partition_with_mask(v, dir)
                        if not success_g or not success_p:
                            #if loading fails, throw an error, since this should not happen
                            raise ValueError(f"Failed to load partition for vertex {v} and direction {dir} when setting irregularity partition")
                        task2 = Task(link, (A, B), self.eps)
                        irreg_v, irreg_count = task2.compute_irregular_vertices(gamma)
                        self.partition_manager.savePartition(v, dir + "i0", np.array(irreg_v), np.ones(len(B), dtype=bool), A, B) #save
                        self.partition_manager.savePartition(v, dir + "i1", ~np.array(irreg_v), np.ones(len(B), dtype=bool), A, B) #save
                        self.partition_manager.deletePartition(v, dir) #delete old partition to save space
                
            elif dev_weight > self.dev_threshold * pathweight:
                partition_info['failure_reason'] = 'FAILED_DEVIATION_CHECK'
                self.partition_logs.append(partition_info)
                #add the new directions to queue for the next iterations
                self.q.put(dir + "d0")
                self.q.put(dir + "d1")
                self.q.put(dir + "d2")
                self.q.put(dir + "d3")
                for i, v in enumerate(self.V2):
                    if dev_vertices[i]:
                        success_g, link = self.partition_manager.loadLinkGraph(v)
                        success_p, (A, B) = self._load_partition_with_mask(v, dir)
                        if not success_g or not success_p:
                            #if loading fails, throw an error, since this should not happen
                            raise ValueError(f"Failed to load partition for vertex {v} and direction {dir} when setting deviation partition")
                        task3 = Task(link, (A, B), self.eps)
                        L,R = task3.produce_new_masks(gamma)
                        self.partition_manager.savePartition(v, dir + "d0", np.array(L), np.array(R), A, B) #save
                        self.partition_manager.savePartition(v, dir + "d1", ~np.array(L), np.array(R), A, B) #save
                        self.partition_manager.savePartition(v, dir + "d2", np.array(L), ~np.array(R), A, B) #save
                        self.partition_manager.savePartition(v, dir + "d3", ~np.array(L), ~np.array(R), A, B) #save
                        self.partition_manager.deletePartition(v, dir) #delete old partition to save space
            else:
                #if we are not in a problem step, save the partition in the output list
                partition_info['failure_reason'] = 'PASSED_ALL_CHECKS'
                self.partition_logs.append(partition_info)
                out.append(dir)
        return out

        