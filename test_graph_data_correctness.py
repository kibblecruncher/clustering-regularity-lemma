import numpy as np
import networkx as nx

from manager import Manager


def manager_for_graph(graph, tmp_path):
    return Manager(
        G=graph,
        eps=0.5,
        irreg_vtx_threshold=0.5**5 / 90,
        dev_vtx_threshold=0.5,
        irreg_vtx_count_threshold=0.1,
        dev_threshold=0.1,
        irreg_threshold=0.5,
        clustering_threshold=0.5,
        partition_dir=str(tmp_path / "partitions"),
        graph_dir=str(tmp_path / "graphs"),
    )


def node_id(node):
    return node[0]


def full_masks(A, B, _v):
    return np.ones(len(A), dtype=bool), np.ones(len(B), dtype=bool)


def even_neighbor_masks(A, B, _v):
    return (
        np.array([node_id(node) % 2 == 0 for node in A], dtype=bool),
        np.array([node_id(node) % 2 == 0 for node in B], dtype=bool),
    )


def asymmetric_neighbor_masks(A, B, v):
    return (
        np.array([node_id(node) < v for node in A], dtype=bool),
        np.array([node_id(node) > v for node in B], dtype=bool),
    )


def save_masked_direction(manager, direction, mask_fn):
    manager.V2 = manager.graph_manager.getV2()
    for v in manager.V2:
        A, B = manager.graph_manager.makeLinkPartition(v)
        mask_A, mask_B = mask_fn(A, B, v)
        if np.any(mask_A) and np.any(mask_B):
            manager.partition_manager.savePartition(v, direction, mask_A, mask_B, A, B)
            manager.partition_manager.saveLinkGraph(v, manager.graph_manager.makeLinkGraph(v))


def load_masked_records(manager, direction):
    records = []
    for v in manager.V2:
        success, record = manager.partition_manager.loadPartitionRecord(v, direction)
        if success:
            records.append(record)
    return records


def oracle_graph_data(manager, direction):
    pathweight = 0
    triangle_count = 0
    diagonal_pair_count = 0
    open_pair_count = 0

    for record in load_masked_records(manager, direction):
        A, B = record.masked_neighbors()
        b_set = set(B)
        pathweight += len(A) * len(B)

        for a in A:
            for b in B:
                if a == (b[0], 0):
                    diagonal_pair_count += 1
                elif manager.graph_manager.H.has_edge(a, b):
                    triangle_count += 1
                else:
                    open_pair_count += 1

    gamma = 0.0 if pathweight == 0 else triangle_count / pathweight
    return pathweight, triangle_count, gamma, diagonal_pair_count, open_pair_count


def assert_graph_data_matches_oracle(manager, direction):
    expected_pathweight, expected_triangles, expected_gamma, _, _ = oracle_graph_data(
        manager,
        direction,
    )
    pathweight, triangle_count, gamma = manager.compute_graph_data(direction)

    assert pathweight == expected_pathweight
    assert triangle_count == expected_triangles
    assert gamma == expected_gamma


def test_compute_graph_data_empty_graph(tmp_path):
    manager = manager_for_graph(nx.empty_graph(8), tmp_path)
    save_masked_direction(manager, "full", full_masks)

    pathweight, triangle_count, gamma = manager.compute_graph_data("full")

    assert pathweight == 0
    assert triangle_count == 0
    assert gamma == 0.0


def test_compute_graph_data_complete_graph_full_masks_has_expected_values(tmp_path):
    n = 40
    graph = nx.complete_graph(n)
    manager = manager_for_graph(graph, tmp_path)
    save_masked_direction(manager, "full", full_masks)

    pathweight, triangle_count, gamma = manager.compute_graph_data("full")

    assert pathweight == n * (n - 1) ** 2
    assert triangle_count == n * (n - 1) * (n - 2)
    assert pathweight == triangle_count + 2 * graph.number_of_edges()
    assert gamma == (n - 2) / (n - 1)
    assert gamma > 0.97
    assert_graph_data_matches_oracle(manager, "full")


def test_compute_graph_data_complete_graph_with_varying_masks(tmp_path):
    n = 20
    manager = manager_for_graph(nx.complete_graph(n), tmp_path)

    for direction, mask_fn in [
        ("even", even_neighbor_masks),
        ("asym", asymmetric_neighbor_masks),
    ]:
        save_masked_direction(manager, direction, mask_fn)
        assert_graph_data_matches_oracle(manager, direction)

        pathweight, triangle_count, gamma, diagonal_count, open_count = oracle_graph_data(
            manager,
            direction,
        )
        assert pathweight == triangle_count + diagonal_count
        assert open_count == 0
        assert manager.compute_graph_data(direction) == (pathweight, triangle_count, gamma)


def test_compute_graph_data_random_graph_gamma_tracks_edge_density_full_masks(tmp_path):
    n = 160
    density = 0.35
    graph = nx.fast_gnp_random_graph(n, density, seed=20260517)
    manager = manager_for_graph(graph, tmp_path)
    save_masked_direction(manager, "full", full_masks)

    pathweight, triangle_count, gamma = manager.compute_graph_data("full")

    assert pathweight > 0
    assert triangle_count > 0
    assert abs(gamma - nx.density(graph)) < 0.06
    assert_graph_data_matches_oracle(manager, "full")


def test_compute_graph_data_random_graph_with_varying_masks(tmp_path):
    graph = nx.fast_gnp_random_graph(90, 0.4, seed=20260518)
    manager = manager_for_graph(graph, tmp_path)

    for direction, mask_fn in [
        ("even", even_neighbor_masks),
        ("asym", asymmetric_neighbor_masks),
    ]:
        save_masked_direction(manager, direction, mask_fn)
        assert_graph_data_matches_oracle(manager, direction)

        pathweight, triangle_count, gamma, diagonal_count, open_count = oracle_graph_data(
            manager,
            direction,
        )
        assert pathweight == triangle_count + diagonal_count + open_count
        assert manager.compute_graph_data(direction) == (pathweight, triangle_count, gamma)
        assert 0.0 <= gamma <= 1.0
