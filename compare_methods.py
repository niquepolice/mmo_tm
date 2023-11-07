import src.test as test
from pathlib import Path
from src.algs import subgd, ustm, frank_wolfe, cyclic
from src.my_algs import conjugate_frank_wolfe , Bi_conjugate_frank_wolfe , N_conjugate_frank_wolfe ,fukushima_frank_wolfe


networks_path = Path("./TransportationNetworks")


# folder = "SiouxFalls"
# net_name = "SiouxFalls_net"
# traffic_mat_name = "SiouxFalls_trips"

folder = "Anaheim"
net_name = "Anaheim_net"
traffic_mat_name = "Anaheim_trips"
    
# Не работает (mu != inf , но rho = 0) (sigma * = ... / rho ...) 
# folder = "Philadelphia"
# net_name = "Philadelphia_net"
# traffic_mat_name = "Philadelphia_trips"

# rho = 0
# folder = "Berlin-Tiergarten"
# net_name = 'berlin-tiergarten_net'
# traffic_mat_name = 'berlin-tiergarten_trips'

# folder = "Terrassa-Asymmetric"
# net_name = 'Terrassa-Asym_net'
# traffic_mat_name = 'Terrassa-Asym_trips'

# folder = "Eastern-Massachusetts"
# net_name = 'EMA_net'
# traffic_mat_name = 'EMA_trips'

# rho = 0 and fft = 0
# folder = "Chicago-Sketch"
# net_name = 'ChicagoSketch_net'
# traffic_mat_name = 'ChicagoSketch_trips'


# rho = 0
# folder = "Berlin-Mitte-Center"
# net_name = 'berlin-mitte-center_net'
# traffic_mat_name = 'berlin-mitte-center_trips'

# Не работает (ибо пока архитектура не учитывает дуги )
# folder = "Berlin-Center"
# net_name = 'berlin-center_net'
# traffic_mat_name = "berlin-center_trips"

# folder = "Berlin-Friedrichshain"
# net_name = "friedrichshain-center_net"
# traffic_mat_name = "friedrichshain-center_trips"

# key error  в sum_flows_from_tree  
# folder = "Winnipeg-Asymmetric"
# net_name = 'Winnipeg-Asym_net'
# traffic_mat_name = "Winnipeg-Asym_trips"

# folder = "Winnipeg"
# net_name = 'Winnipeg_net'
# traffic_mat_name = "Winnipeg_trips"

# folder = "Austin"
# net_name = 'Austin_net'
# traffic_mat_name = "Austin_trips"

# rho = 0
# folder = "Barcelona"
# net_name = 'Barcelona_net'
# traffic_mat_name = "Barcelona_trips"

# folder = "Berlin-Mitte-Prenzlauerberg-Friedrichshain-Center"
# net_name = 'berlin-mitte-prenzlauerberg-friedrichshain-center_net'
# traffic_mat_name = "berlin-mitte-prenzlauerberg-friedrichshain-center_trips"

# folder = "Hessen-Asymmetric"
# net_name = 'Hessen-Asym_net'
# traffic_mat_name = "Hessen-Asym_trips"

# rho = 0 
# folder = "GoldCoast"
# net_name = 'Goldcoast_network_2016_01'
# traffic_mat_name = "Goldcoast_trips_2016_01"



## LOAD CITY 
beckmann_model , city_info = test.init_city(networks_path=networks_path ,folder=folder ,net_name=net_name,traffic_mat_name=traffic_mat_name)
eps_abs = city_info['eps_abs']

# from collections import Counter

# for k ,v in Counter(beckmann_model.graph.ep.mu.a).items() :
#     print(k , v)

# print('asdad')

# for k ,v in Counter(beckmann_model.graph.ep.rho).items() :
#     print(k , v)


# ##EXPERIMENTS RUN
max_iter = 100
list_methods = []

## FUKUSHIMA FW

weight_param = [0.15]
for weight in weight_param :
    list_methods.append((fukushima_frank_wolfe ,'fukushima_frank_wolfe linesearch weighted =' + str(weight) , 
        {'eps_abs' : eps_abs , 'max_iter':max_iter , 'stop_by_crit': False , 'linesearch': True  , 'weight_parameter' : weight  } ))
# cnts = [5]
# for cnt in cnts :
#     list_methods.append((fukushima_frank_wolfe ,'fukushima_frank_wolfe linesearch N =' + str(cnt) , 
#         {'eps_abs' : eps_abs , 'max_iter':max_iter , 'stop_by_crit': False , 'linesearch': True  , 'cnt_directional' : cnt  } ))

##NFW
cnts = [3]
for cnt in cnts :
    list_methods.append((N_conjugate_frank_wolfe ,'N_conjugate_frank_wolfe linesearch N =' + str(cnt) , 
        {'eps_abs' : eps_abs , 'max_iter':max_iter , 'stop_by_crit': False , 'linesearch': True  , 'cnt_conjugates' : cnt  } ))

# ##BCFW linesearch
list_methods.append((Bi_conjugate_frank_wolfe ,'Bi_conjugate_frank_wolfe linesearch' , 
    {'eps_abs' : eps_abs , 'max_iter':max_iter , 'stop_by_crit': False , 'linesearch': True } ))
# # ##CFWM linesearch
list_methods.append((conjugate_frank_wolfe ,'conjugate_frank_wolfe linesearch' , 
    {'eps_abs' : eps_abs , 'max_iter':max_iter , 'stop_by_crit': False , 'linesearch': True  ,'alpha_default' : 0.6} ))
# # ##CFWM 
# list_methods.append((conjugate_frank_wolfe ,'conjugate_frank_wolfe' , 
#     {'eps_abs' : eps_abs , 'max_iter':max_iter , 'stop_by_crit': False} ))
# # ## FWM
# list_methods.append((frank_wolfe ,'frank_wolfe' , 
#     {'eps_abs' : eps_abs , 'max_iter':max_iter , 'stop_by_crit': False} ))
# # # ## FWM linesearch
list_methods.append((frank_wolfe ,'frank_wolfe linesearch' , 
    {'eps_abs' : eps_abs , 'max_iter':max_iter , 'stop_by_crit': False ,'linesearch':True}  ))
# # method , name , solver_kwargs = list_methods[0]
# # result = test.run_method(method , name , solver_kwargs , beckmann_model ,city_name = folder , max_iter = max_iter)

# # dgaps =result[0][0]['duality_gap'] 
# # steps = result[1]


# # test.plt.figure(figsize = (20, 20))
# # test.plt.plot(dgaps)
# # test.plt.scatter(steps , dgaps[steps] , color = 'red')
# # test.plt.yscale('log')
# # test.plt.show()
# # plt.plot(result[''])

experiments = test.run_experiment(list_methods , model=beckmann_model, city_name=folder , max_iter=max_iter)

# #DISPLAY RESULTS
test.plot( experiments , name_output_values=['relative_gap'] , save=True  ,time_iters=True)

