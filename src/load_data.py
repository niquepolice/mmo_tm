from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import xmltodict

from src.commons import Correspondences

FLOAT = np.float32


def read_metadata_networks_tntp(filename: Path) -> dict:
    with open(filename, "r") as file:
        zones = int(file.readline()[len("<NUMBER OF ZONES>") :].strip())
        nodes = int(file.readline()[len("<NUMBER OF NODES>") :].strip())
        can_pass_through_zones = int(file.readline()[len("<FIRST THRU NODE>") :].strip()) == 1
    return dict(zones=zones, nodes=nodes, can_pass_through_zones=can_pass_through_zones)


def read_graph_transport_networks_tntp(filename: Path) -> tuple[nx.DiGraph, dict]:
    # Made on the basis of
    # https://github.com/bstabler/TransportationNetworks/blob/master/_scripts/parsing%20networks%20in%20Python.ipynb

    """If centroids are separated from regular nodes, the ordering of nodes is (sources, through_nodes, targets)"""

    metadata = read_metadata_networks_tntp(filename)

    net = pd.read_csv(filename, skiprows=8, sep="\t")
    net.columns = [col.strip().lower() for col in net.columns]
    net.loc[:, ["init_node", "term_node"]] -= 1

    graph = nx.DiGraph()
    graph.add_nodes_from(range(metadata["nodes"] + (0 if metadata["can_pass_through_zones"] else metadata["zones"])))

    for row in net.iterrows():
        init_node = row[1].init_node
        term_node = row[1].term_node
        if not metadata["can_pass_through_zones"] and term_node < metadata["zones"]:
            term_node += metadata["nodes"]
        graph.add_edge(
            init_node,
            term_node,
            free_flow_times=FLOAT(row[1].free_flow_time),
            capacities=FLOAT(row[1].capacity),
            rho=FLOAT(row[1].b),
            mu=1 / FLOAT(row[1].power),
        )

    return graph, metadata


def read_traffic_mat_transport_networks_tntp(filename: Path, metadata: dict) -> Correspondences:
    # Made on the basis of
    # https://github.com/bstabler/TransportationNetworks/blob/master/_scripts/parsing%20networks%20in%20Python.ipynb

    with open(filename, "r") as file:
        blocks = file.read().split("Origin")[1:]
    matrix = {}
    for block in blocks:
        demand_data_for_origin = block.split("\n")
        orig = int(demand_data_for_origin[0])
        destinations = ";".join(demand_data_for_origin[1:]).split(";")
        matrix[orig] = {}
        for dest_str in destinations:
            if len(dest_str.strip()) == 0:
                continue
            dest, demand = dest_str.split(":")
            matrix[orig][int(dest)] = FLOAT(demand)

    zones = metadata["zones"]
    traffic_mat = np.zeros((zones, zones))
    for i in range(zones):
        for j in range(zones):
            traffic_mat[i, j] = matrix.get(i + 1, {}).get(j + 1, 0)

    num_nodes = metadata["nodes"]

    sources = np.arange(zones)
    print(f'{metadata["can_pass_through_zones"]=}')
    targets = sources if metadata["can_pass_through_zones"] else num_nodes + sources

    num_nodes += 0 if metadata["can_pass_through_zones"] else metadata["zones"]
    node_traffic_mat = np.zeros((num_nodes, num_nodes), dtype=FLOAT)
    if metadata["can_pass_through_zones"]:
        node_traffic_mat[:zones, :zones] = traffic_mat
    else:
        node_traffic_mat[:zones, -zones:] = traffic_mat

    return Correspondences(
        traffic_mat=traffic_mat,
        node_traffic_mat=node_traffic_mat,
        sources=sources,
        targets=targets,
    )


def update_node_coordinates(node_coords: dict, metadata: dict):
    if not metadata["can_pass_through_zones"]:
        for key in range(metadata["zones"]):
            node_coords[key + metadata["nodes"]] = node_coords[key].copy()


def read_node_coordinates_transport_networks_tntp(filename: Path, metadata: dict) -> dict:
    try:
        data = pd.read_csv(
            filename,
            delim_whitespace=True,
            header=0,
            names=["node", "x", "y", "semicolon"],
        )
    except pd.errors.ParserError:
        data = pd.read_csv(filename, delim_whitespace=True, header=0, names=["node", "x", "y"])
    data = data.loc[:, ["x", "y"]]

    node_coords = {}
    for row in data.iterrows():
        node_coords[row[0]] = {"x": FLOAT(row[1].x), "y": FLOAT(row[1].y)}

    update_node_coordinates(node_coords, metadata)
    return node_coords


def read_graph_sndlib_xml(filename: Path) -> nx.Graph:
    with open(filename, "r") as file:
        graph_dct = xmltodict.parse(file.read())["network"]["networkStructure"]

    graph = nx.DiGraph()

    for node in graph_dct["nodes"]["node"]:
        graph.add_node(node["@id"], x=FLOAT(node["coordinates"]["x"]), y=FLOAT(node["coordinates"]["y"]))

    for edge in graph_dct["links"]["link"]:
        cost = FLOAT(edge.get("routingCost", 1.0))
        if "preInstalledModule" in edge:
            bandwidth = FLOAT(edge["preInstalledModule"]["capacity"])
        elif "additionalModules" in edge and "addModule" in edge["additionalModules"]:
            module = edge["additionalModules"]["addModule"]
            if isinstance(module, list):
                module = module[0]
            bandwidth = FLOAT(module["capacity"])
        else:
            bandwidth = FLOAT(1.0)
        graph.add_edge(edge["source"], edge["target"], free_flow_times=cost, capacities=bandwidth, rho=1.0, mu=-0.5)
        graph.add_edge(edge["target"], edge["source"], free_flow_times=cost, capacities=bandwidth, rho=1.0, mu=-0.5)

    return graph


def read_traffic_mat_sndlib_xml(filename, delim = None) -> np.ndarray:
    with open(filename, "r") as file:
        xml_dct = xmltodict.parse(file.read())["network"]

    node_label_to_num = {node["@id"]: i for i, node in enumerate(xml_dct["networkStructure"]["nodes"]["node"])}
    traffic_mat = np.zeros((len(node_label_to_num), len(node_label_to_num)), dtype=FLOAT)
    for demand in xml_dct["demands"]["demand"]:
        source = node_label_to_num[demand["source"]]
        target = node_label_to_num[demand["target"]]
        traffic_mat[source, target] = demand["demandValue"]

    if not delim is None:
        traffic_mat /= delim * traffic_mat.max()       

    return Correspondences(
        traffic_mat=traffic_mat.copy(),
        node_traffic_mat=traffic_mat.copy(),
        sources=np.arange(len(node_label_to_num)),
        targets=np.arange(len(node_label_to_num)),
    )

def scale_graph_bandwidth_and_cost(graph: nx.Graph) -> nx.Graph:
    scaled_graph = graph.copy()
    max_bandwidth = max(nx.get_edge_attributes(graph, "capacities").values())
    max_cost = max(nx.get_edge_attributes(graph, "free_flow_times").values())
    for edge in graph.edges:
        scaled_graph.edges[edge]["capacities"] /= max_bandwidth
        scaled_graph.edges[edge]["free_flow_times"] /= max_cost
    return scaled_graph