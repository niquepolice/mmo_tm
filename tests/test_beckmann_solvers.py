from pathlib import Path

import networkx as nx

from src.models import BeckmannModel
from src.algs import frank_wolfe

from src.load_data import (
    read_graph_transport_networks_tntp,
    read_traffic_mat_transport_networks_tntp,
)

def test_cvxpy_and_fw_sioux_falls_mu_1():
    """CVXPY fails for mu != 1, therefore set mu = 1 here"""
    networks_path = Path("./TransportationNetworks")

    folder = "SiouxFalls"
    net_name = "SiouxFalls_net"
    traffic_mat_name = "SiouxFalls_trips"

    net_file = networks_path / folder / f"{net_name}.tntp"
    traffic_mat_file = networks_path / folder / f"{traffic_mat_name}.tntp"
    graph, metadata = read_graph_transport_networks_tntp(net_file)

    nx.set_edge_attributes(graph, name="mu", values=1)  # overwrite mu

    correspondences = read_traffic_mat_transport_networks_tntp(traffic_mat_file, metadata)

    beckmann_model = BeckmannModel(graph, correspondences)

    flows_e = beckmann_model.solve_cvxpy(verbose=True, solver="SCS",  eps_infeas=1e-12)
    # flows_e = beckmann_model.solve_cvxpy(verbose=True, solver="ECOS", feastol=1e-12)

    EPS_ABS = 1000
    times_e_fw, flows_e_fw, logs, optimal = frank_wolfe(
        beckmann_model, eps_abs=EPS_ABS, max_iter=2000, stop_by_crit=True, use_tqdm=False
    )

    assert optimal == True
    assert abs(beckmann_model.primal(flows_e_fw) - beckmann_model.primal(flows_e)) <= 2 * EPS_ABS
