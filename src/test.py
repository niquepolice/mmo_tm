
import networkx as nx
import numpy as np
import cvxpy as cp
from tqdm import tqdm

from src.load_data import (
    read_metadata_networks_tntp,
    read_graph_transport_networks_tntp,
    read_traffic_mat_transport_networks_tntp,
)
import src.test as test
from src.models import SDModel, BeckmannModel, TwostageModel
from src.algs import subgd, ustm, frank_wolfe, cyclic
from src.cvxpy_solvers import get_max_traffic_mat_mul
from src.commons import Correspondences
import matplotlib.pyplot as plt
    
import pandas as pd
import time
from matplotlib import rc
from platform import python_version
import graph_tool
import pickle
import datetime


# по названию города ==> модель и таблицу графа
def init_city(networks_path ,folder ,net_name ,traffic_mat_name ) :
    ##LOAD DATA
    net_file = networks_path / folder / f"{net_name}.tntp"
    traffic_mat_file = networks_path / folder / f"{traffic_mat_name}.tntp"
    graph, metadata = read_graph_transport_networks_tntp(net_file)
    correspondences = read_traffic_mat_transport_networks_tntp(traffic_mat_file, metadata)
    n = graph.number_of_nodes()
    print(f"{graph.number_of_edges()=}, {graph.number_of_nodes()=}")


    ## BECKMANN MODEL
    beckmann_model = BeckmannModel(graph, correspondences)
    eps = 1e-6
    mean_bw = beckmann_model.graph.ep.capacities.a.mean()
    mean_cost = beckmann_model.graph.ep.free_flow_times.a.mean()
    # cost suboptimality <= eps * (average link cost * avg bandwidth * |E| \approx total cost when beta=1)
    eps_abs = eps * mean_cost * mean_bw * graph.number_of_edges()
    eps_cons_abs = eps * mean_bw 
    # sum of capacity violation <= eps * average link capacity
    print(eps_abs, eps_cons_abs)

    result = {
        'eps_abs' : eps_abs ,                 
    }
    return beckmann_model , result


# Запуск моделей с выбором числа итераций и города и аргументов метода и кастомный нейминг
def run_method(method , name , solver_kwargs , model  ,city_name = '' , max_iter = 100) :
    times, flows, logs, optimal = method(model, **solver_kwargs)
    dgap, time_log, other = logs
    print(np.sum(flows))
    experiment_attrs = ({'duality_gap' : np.abs(dgap) , 'time_log' : time_log , 'primal' : other['primal']}  ,  name , max_iter , city_name )
    return experiment_attrs ,other 
def run_experiment(list_methods , model, city_name , max_iter) :
    experiments = []
    for method , name , solver_kwargs in list_methods :
        result = run_method(method , name , solver_kwargs , model ,city_name = city_name , max_iter = max_iter)
        experiments.append(result[0])
    return experiments

# список графиков ==>> вывод и сохранение
def plot( experiments , name_output_values , time_iters = False, loglog = False , save = False  ) :
    color_generator = plt.cm.get_cmap('tab20', len(experiments))
    colors = [color_generator(i) for i in np.linspace(0, 1, len(experiments))]
    time = datetime.datetime.now().time().strftime("%H:%M")
    date = datetime.datetime.now().date()
    experiments_folder = './experiments_results/'
    plt.figure(figsize = (12, 6))
    for col_id , experiment in enumerate(experiments) :
        result , name , max_iter , city_name = experiment
        for e,name_values in enumerate(name_output_values) :
            values = result[name_values]
            if time_iters :
                iters = result['time_log']
            else :
                iters = np.arange(max_iter)
            plt.subplot(1, len(name_output_values), e+1)
            
            if name_values == 'primal' :
                iters = iters[-int(len(iters)/2):]
                values = values[-int(len(values)/2):]
            if loglog == False :
                plt.plot(iters, values , color =colors[col_id] ,label = name)
                plt.yscale('log')
            else :
                plt.loglog(iters, values , color =colors[col_id] ,label = name)
            plt.ylabel(name_values, fontsize = 12)
            plt.xlabel('seconds' if time_iters else 'iterations' , fontsize = 12)
            plt.title('Сходимость ' + name_values +' на городе ' + city_name)
            plt.legend()
            
            experiment_path = experiments_folder +'iterations/'+ name_values+'_'+ name + '_' + city_name + '_' + str(max_iter) + 'iters_datetime_' + str(date)+'_' + str(time)+ '.csv'
            if save :
                df_values = pd.DataFrame([values,iters])
                df_values.to_csv( experiment_path  , index=False)
    
    if save :
        plt.savefig(experiments_folder +'pictures/'+ 'Experiment_' + ' '.join(name_output_values)+ '_date_'+ str(date) +'_time_'+str(time)+ '_city_' + city_name + '_' + str(max_iter) + 'iters' + '.png')
    plt.tight_layout()
    plt.show()
    

# Кастомно прочекать графики по значениям целевой функции в file.csv
def display(filename , last_iters = 0 ) :
    values = pd.read_csv(filename ).values
    
    plt.figure(figsize = (10, 5))

    iters = np.arange(len(values) if last_iters == 0 else last_iters)
    n = len(iters)
    values = values[-n:] 
    plt.plot(iters, values  ,label = filename)
    plt.ylabel('values', fontsize = 12)
    plt.xlabel('iterations', fontsize = 12)
    plt.title('Сходимость values ')
    plt.legend()
    plt.yscale('log')
    plt.show()


# display('./experiments_results/Frank Wolfe_SiouxFalls_100iters.csv' , 10)

