import src.test as test
from pathlib import Path
from src.algs import subgd, ustm, frank_wolfe, cyclic

networks_path = Path("./TransportationNetworks")

folder = "SiouxFalls"
net_name = "SiouxFalls_net"
traffic_mat_name = "SiouxFalls_trips"

# folder = "Anaheim"
# net_name = "Anaheim_net"
# traffic_mat_name = "Anaheim_trips"
    
# folder = "Barcelona"
# net_name = "Barcelona_net"
# traffic_mat_name = "Barcelona_trips"

## LOAD CITY 
beckmann_model , city_info = test.init_city(networks_path=networks_path ,folder=folder ,net_name=net_name,traffic_mat_name=traffic_mat_name)
eps_abs = city_info['eps_abs']

##EXPERIMENTS RUN
max_iter = 100
list_methods = []


## FWM
list_methods.append((frank_wolfe ,'frank_wolfe' , 
    {'eps_abs' : eps_abs , 'max_iter':max_iter , 'stop_by_crit': False} ))



experiments = test.run_experiment(list_methods , model=beckmann_model, city_name=folder , max_iter=max_iter)

#DISPLAY RESULTS
test.plot( experiments , name_output_values='duality_gap')





#NFWM
# N = 1
# list_methods.append(( 'cfwm', 'Nconjugate Frank Wolfe linesearch , N =' +str(N) , 
#     {'max_iter' : max_iter, 'stop_crit': 'max_iter','verbose' : True, 'verbose_step': 2000, 'save_history' : True , "linesearch" : True , 'NFW': N} ))
# N = 8
# list_methods.append(( 'cfwm', 'Nconjugate Frank Wolfe linesearch , N =' +str(N) , 
#     {'max_iter' : max_iter, 'stop_crit': 'max_iter','verbose' : True, 'verbose_step': 2000, 'save_history' : True , "linesearch" : True , 'NFW': N} ))
# N = 6
# list_methods.append(( 'cfwm', 'Nconjugate Frank Wolfe linesearch , N =' +str(N) , 
#     {'max_iter' : max_iter, 'stop_crit': 'max_iter','verbose' : True, 'verbose_step': 2000, 'save_history' : True , "linesearch" : True , 'NFW': N} ))
# N = 5
# list_methods.append(( 'cfwm', 'Nconjugate Frank Wolfe linesearch , N =' +str(N) , 
#     {'max_iter' : max_iter, 'stop_crit': 'max_iter','verbose' : True, 'verbose_step': 2000, 'save_history' : True , "linesearch" : True , 'NFW': N} ))
# N = 4
# list_methods.append(( 'cfwm', 'Nconjugate Frank Wolfe linesearch , N =' +str(N) , 
#     {'max_iter' : max_iter, 'stop_crit': 'max_iter','verbose' : True, 'verbose_step': 2000, 'save_history' : True , "linesearch" : True , 'NFW': N} ))
# N = 3
# list_methods.append(( 'cfwm', 'Nconjugate Frank Wolfe linesearch , N =' +str(N) , 
#     {'max_iter' : max_iter, 'stop_crit': 'max_iter','verbose' : True, 'verbose_step': 2000, 'save_history' : True , "linesearch" : True , 'NFW': N} ))
# N = 2
# list_methods.append(( 'cfwm', 'Nconjugate Frank Wolfe linesearch , N =' +str(N) , 
#     {'max_iter' : max_iter, 'stop_crit': 'max_iter','verbose' : True, 'verbose_step': 2000, 'save_history' : True , "linesearch" : True , 'NFW': N} ))


### FWM
# list_methods.append(( 'fwm', 'Frank Wolfe' , 
#     {'max_iter' : max_iter, 'stop_crit': 'max_iter','verbose' : True, 'verbose_step': 1000, 'save_history' : True} ))
### FWM linesearch
# list_methods.append(( 'fwm', 'Frank Wolfe linesearch' , 
#     {'max_iter' : max_iter, 'stop_crit': 'max_iter','verbose' : True, 'verbose_step': 2000, 'save_history' : True , "linesearch" : True} ))
### FWM lambda
# list_methods.append(( 'fwm', 'Frank Wolfe lambda_k ='+str(1.5) , 
#     {'max_iter' : max_iter, 'stop_crit': 'max_iter',
#                  'verbose' : True, 'verbose_step': 2000, 'save_history' : True , 'lambda_k': 1.5} ))


### CFWM
# list_methods.append(( 'cfwm', 'Conjugate Frank Wolfe' , 
#     {'max_iter' : max_iter, 'stop_crit': 'max_iter','verbose' : True, 'verbose_step': 2000, 'save_history' : True , "alpha_default" : 0.6 } ))
### CFWM linesearch
# list_methods.append(( 'cfwm', 'Conjugate Frank Wolfe linesearch' , 
#     {'max_iter' : max_iter, 'stop_crit': 'max_iter','verbose' : True, 'verbose_step': verbose_step, 'save_history' : True , "alpha_default" : 0.6 , "linesearch" :True} ))

### FWF weighted linesearch
# weights = [0.15]
# for w in weights :
#     list_methods.append(( 'fwf' , 'Fukushima Frank-Wolfe weighted(linesearch) =' + str(w) ,
#         {'max_iter' : max_iter, 'stop_crit': 'max_iter','verbose' : True, 'verbose_step': verbose_step, 'save_history' : True , 'weight_parameter' : w ,  'linesearch': True }))

### FWF weighted
# weights = [ 0.15 , 0.2 ]
# for w in weights :
#     list_methods.append(( 'fwf' , 'Fukushima Frank-Wolfe weighted =' + str(w) ,
#         {'max_iter' : max_iter, 'stop_crit': 'max_iter','verbose' : True, 'verbose_step': verbose_step, 'save_history' : True , 'weight_parameter' : w }))

### FWF lambda weighted linesearch
# weights = [ 0.2,0.25,0.3 , 0.35 , 0.4,0.5]
# for w in weights :
#     list_methods.append(( 'fwf' , 'Fukushima Frank-Wolfe lambda_k =' +str(1.5)+' weighted =' + str(w) ,
#         {'max_iter' : max_iter, 'stop_crit': 'max_iter','verbose' : True, 'verbose_step': verbose_step, 'save_history' : True , 'weight_parameter' : w ,  'linesearch': True  ,
#         'lambda_k':1.5}))



### FWF l_param with lambda mod
# l_parameters = [2,5]
# lambda_k = 1.5
# for l in l_parameters:
#     list_methods.append(( 'fwf' , 'Fukushima Frank-Wolfe lambda='+str(lambda_k)+' with l_param =' + str(l) ,
#         {'max_iter' : max_iter, 'stop_crit': 'max_iter','verbose' : True, 'verbose_step': 2000, 'save_history' : True ,
#                       'l_parameter' : l , 'linesearch' : True , 'lambda_k': lambda_k}))
    
### Biconjugate Frank Wolfe 
# list_methods.append(( 'cfwm', 'Biconjugate Frank Wolfe' , 
#     {'max_iter' : max_iter, 'stop_crit': 'max_iter',
#                  'verbose' : True, 'verbose_step': verbose_step, 'save_history' : True , 'biconjugate' : True } ))
# ### Biconjugate Frank Wolfe linesearch
# list_methods.append(( 'cfwm', 'Biconjugate Frank Wolfe(linesearch)' , 
#     {'max_iter' : max_iter, 'stop_crit': 'max_iter',
#                  'verbose' : True, 'verbose_step': 2000, 'save_history' : True , 'biconjugate' : True , 'linesearch':True } ))
### USTM
# eps_abs = 31
# list_methods.append(( 'ustm', 'USTM with eps_abs ='+ str(eps_abs) , 
#     {'eps_abs': eps_abs,'max_iter': max_iter, 'stop_crit': 'dual_gap', 'verbose' : True, 'verbose_step': 2000, 'save_history' : True} ))
### UGD 
# eps_abs = 31
# list_methods.append(( 'ugd', 'UGD with eps_abs ='+ str(eps_abs) , 
#     {'eps_abs': eps_abs,'max_iter': max_iter, 'stop_crit': 'dual_gap','verbose' : True, 'verbose_step': 4000, 'save_history' : True} ))
### WDA 
# list_methods.append(( 'wda', 'WDA' , 
#     {'max_iter' : max_iter, 'stop_crit': 'max_iter','verbose': True, 'verbose_step': 4000, 'save_history': True} ))
### WDA-noncomposite
# list_methods.append(( 'wda', 'WDA noncomposote' , 
#     {'max_iter' : max_iter, 'stop_crit': 'max_iter','verbose': True, 'verbose_step': 4000, 'save_history' : True} ))


