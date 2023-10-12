from typing import Optional

import numpy as np
import time
from tqdm import tqdm
from scipy.optimize import minimize_scalar
from src.models import TrafficModel, BeckmannModel, TwostageModel, Model



def conjugate_frank_wolfe(
    model: BeckmannModel,
    eps_abs: float,
    max_iter: int = 100,  # 0 for no limit (some big number)
    times_start: Optional[np.ndarray] = None,
    stop_by_crit: bool = True,
    use_tqdm: bool = True,
    alpha_default: float = 0.6 ,
    linesearch : bool = False ,
) -> tuple:
    """One iteration == 1 shortest paths call"""

    optimal = False

    # init flows, not used in averaging
    if times_start is None:
        times_start = model.graph.ep.free_flow_times.a.copy()
    flows = model.flows_on_shortest(times_start)

    max_dual_func_val = -np.inf
    dgap_log = []
    time_log = []

    rng = (
        range(1,1_000_000)
        if max_iter == 0
        else tqdm(range(1,max_iter+1), disable=not use_tqdm)
    )

    gamma = 1.0
    alpha = 1
    for k in rng:

        times = model.tau(flows)
        yk_FW = model.flows_on_shortest(times)

        if k > 1 :
            hessian = model.diff_tau(flows)
            denom = np.sum(( x_star - flows ) * hessian * ( yk_FW - flows_old ))
            if denom == 0 :
                alpha = 0
            else :
                alpha = np.sum(( x_star - flows ) * hessian * ( yk_FW- flows )) / np.sum(( x_star - flows ) * hessian * ( yk_FW- flows_old )) 
            
            if alpha < 0 :
                alpha = 0 
            if alpha > alpha_default :
                alpha = alpha_default

        if k == 1 :
            x_star = yk_FW
        else :
            x_star = x_star*alpha + (1-alpha)*yk_FW           

        if linesearch :
            res = minimize_scalar( lambda y : model.primal((1-y)*flows + y*x_star) , bounds = (0.0,1.0) , tol = 1e-12 )
            gamma = res.x
        else :
            gamma = 2.0/(k + 2)

        flows_old = flows
        flows = (1.0 - gamma) * flows + gamma * x_star

        dual_val = model.dual(times, yk_FW)
        max_dual_func_val = max(max_dual_func_val, dual_val)

        # equal to FW gap if dual_val == max_dual_func_val
        primal = model.primal(flows)
        dgap_log.append(primal - max_dual_func_val)
        time_log.append(time.time())

        if stop_by_crit and dgap_log[-1] <= eps_abs:
            optimal = True
            break

    return (
        times,
        flows,
        (dgap_log, np.array(time_log) - time_log[0]),
        optimal,
    )

## Реализация с того гитхаба 
def Bi_conjugate_frank_wolfe(
    model: BeckmannModel,
    eps_abs: float,
    max_iter: int = 100,  # 0 for no limit (some big number)
    times_start: Optional[np.ndarray] = None,
    stop_by_crit: bool = True,
    use_tqdm: bool = True,
    delta : float = 0.0001 ,
    linesearch : bool = False ,
) -> tuple:
    """One iteration == 1 shortest paths call"""

    optimal = False

    # init flows, not used in averaging
    if times_start is None:
        times_start = model.graph.ep.free_flow_times.a.copy()
    flows = model.flows_on_shortest(times_start)

    max_dual_func_val = -np.inf
    dgap_log = []
    time_log = []

    rng = (
        range(1,1_000_000)
        if max_iter == 0
        else tqdm(range(1,max_iter+1), disable=not use_tqdm)
    )

    gamma = 1.0
    for k in rng:
        if gamma > 0.99999 :
            is_1st_iter = True
            is_2st_iter = True
        if k == 1 or is_1st_iter : #FW
            is_1st_iter = False
            t = model.tau(flows)
            sk_FW = model.flows_on_shortest(t)    
            
            dk = sk_FW - flows
            sk_BFW_old = sk_FW
            # print('-----------------ITER:',k,'--- alpha =' , 1)
        elif k == 2 or is_2st_iter: #CFW
            is_2st_iter = False
            t = model.tau(flows)
            sk_FW = model.flows_on_shortest(t)    
            dk_FW = sk_FW - flows
            
            hessian = model.diff_tau(flows)
            dk_bar = sk_BFW_old - flows # sk_BFW_old from the previous iteration 1
            Nk = np.sum( dk_bar *hessian * dk_FW )
            Dk = np.sum( dk_bar *hessian * (dk_FW - dk_bar) )
            delta = 0.0001 # What value should I use?
            if Dk !=0 and 0 <= Nk/Dk <= 1-delta :
                alphak = Nk/Dk
            elif Dk !=0 and Nk/Dk > 1-delta :
                alphak = 1-delta
            else :
                alphak = 0
            # Generating new sk_BFW and dk_BFW
            sk_BFW = alphak * sk_BFW_old + (1-alphak) * sk_FW
            dk_BFW = sk_BFW - flows
            dk = dk_BFW
            # print('-----------------ITER:',k,'--- directions =', [ np.sum(sk_FW) , np.sum(sk_BFW_old)  ] )
            # print('-----------------ITER:',k,'--- alpha =' , [1-alphak , alphak ])
        else : #BFW
            t = model.tau(flows)
            sk_FW = model.flows_on_shortest(t)
            hessian = model.diff_tau(flows)
            dk_FW = sk_FW - flows

            dk_bar  = sk_BFW - flows
            dk_bbar = gamma * sk_BFW - flows + (1-gamma) * sk_BFW_old
            
            denom_mu_k = np.sum( dk_bbar* hessian * (sk_BFW_old - sk_BFW) )
            if denom_mu_k != 0 :
                mu_k = - np.sum( dk_bbar* hessian * dk_FW ) / denom_mu_k
            else :
                mu_k = 0
            denom_nu_k = np.sum( dk_bar* hessian * dk_bar)
            mu_k = max(0, mu_k)
            if denom_nu_k != 0 :
                nu_k = - np.sum( dk_bar* hessian * dk_FW ) / denom_nu_k + mu_k*gamma/(1-gamma)
            else :
                nu_k = 0
            nu_k = max(0, nu_k)
            
            betta_0 = 1 / ( 1 + mu_k + nu_k )
            betta_1 = nu_k * betta_0
            betta_2 = mu_k * betta_0

            


            # print('-----------------ITER:',k,'--- directions =' ,[ np.sum(sk_FW) , np.sum(sk_BFW) ,np.sum(sk_BFW_old)  ] )
            # print('-----------------ITER:',k,'--- alpha =' , [betta_0 , betta_1 , betta_2 ])
            sk_BFW_new = betta_0*sk_FW + betta_1*sk_BFW + betta_2*sk_BFW_old
            sk_BFW_old = sk_BFW
            sk_BFW = sk_BFW_new

            dk_BFW =  sk_BFW - flows
            dk = dk_BFW
            
            # print(np.sum(np.array(d_list)*hessian*dk , axis=1 ))

        if linesearch :
            res = minimize_scalar( lambda y : model.primal(flows + y*dk ) , bounds = (0.0,1.0) , tol = 1e-12 )
            gamma = res.x
        else :
            gamma = 2.0/(k + 1)

        # if len(d_list) > 1 :
        #     d_list.pop(0)
        # d_list.append(dk)
        

        dual_val = model.dual(t, sk_FW)
        max_dual_func_val = max(max_dual_func_val, dual_val)
        
        
        flows = flows + gamma*dk


        # equal to FW gap if dual_val == max_dual_func_val
        primal = model.primal(flows)
        dgap_log.append(primal - max_dual_func_val)
        time_log.append(time.time())

        if stop_by_crit and dgap_log[-1] <= eps_abs:
            optimal = True
            break

    return (
        t,
        flows,
        (dgap_log, np.array(time_log) - time_log[0]),
        optimal,
    )



## Моя реализция NFW 
def N_conjugate_frank_wolfe(
    model: BeckmannModel,
    eps_abs: float,
    max_iter: int = 100,  # 0 for no limit (some big number)
    times_start: Optional[np.ndarray] = None,
    stop_by_crit: bool = True,
    use_tqdm: bool = True,
    linesearch : bool = False ,
    cnt_conjugates : int = 3
) -> tuple:
    """One iteration == 1 shortest paths call"""

    optimal = False

    # init flows, not used in averaging
    if times_start is None:
        times_start = model.graph.ep.free_flow_times.a.copy()
    flows = model.flows_on_shortest(times_start)

    max_dual_func_val = -np.inf
    dgap_log = []
    time_log = []

    rng = (
        range(1,1_000_000)
        if max_iter == 0
        else tqdm(range(1,max_iter+1), disable=not use_tqdm)
    )

    gamma = 1.0
    d_list = []
    S_list = []
    gamma_list = []
    gamma = 1
    epoch = 0
    for k in rng:
        
        if gamma > 0.99999 :
            epoch = 0
            S_list = []
            d_list = []
        if k == 1  or epoch == 0:
            epoch  = epoch + 1
            t = model.tau(flows)
            sk_FW = model.flows_on_shortest(t)    
            dk = sk_FW - flows
            S_list.append(sk_FW)
            d_list.append(dk)
        else :
            t = model.tau(flows)
            sk_FW = model.flows_on_shortest(t)
            dk_FW = sk_FW - flows
            hessian = model.diff_tau(flows)
            
            B = np.sum(d_list*hessian*d_list    , axis=1)
            A = np.sum(d_list*hessian*dk_FW     , axis=1)    
            N = len(B)
            betta = [-1]*(N+1)
            betta_sum = 0
            delta = 0.0001
            for m in range(N,0,-1) :
                betta[m] = -A[-m]/(B[-m]*(1- gamma_list[-m])) + betta_sum*gamma_list[-m]/(1-gamma_list[-m]) 
                if betta[m] < 0 :
                    betta[m] = 0
                # elif betta[m] > 1- delta :
                #     betta[m] = 1 - delta 
                #     betta_sum = betta_sum + 1 - delta
                else :
                    betta_sum = betta_sum + betta[m]
            alpha_0 = 1/(1+betta_sum)
            alpha = np.array(betta)[1:] * alpha_0

            # if max(np.max(alpha) , alpha_0) > 0.99 :
            #     alpha_0 = 0.2
            #     alpha = 0.8 * np.ones(len(alpha)) / len(alpha) 
            
            
            alpha = alpha[::-1]
            
            # print('-----------------ITER:' , k , '--- directions =' , [np.sum(sk_FW)]+list(np.sum(S_list,axis=1)) )
            # print('-----------------ITER:',k,'--- alpha =' , [alpha_0 ]+ list(alpha[::-1]) )
            
            sk = alpha_0*sk_FW + np.sum(alpha*np.array(S_list).T , axis=1)
            dk = sk - flows

            # print('CHECK CONJUGATE :' , len(d_list)  , 'alpha:' , alpha  , 'alpha_0: ' , alpha_0 , 'list_conjugates: ' , np.sum(dk*hessian*d_list , axis=1))
            d_list.append(dk)
            S_list.append(sk)


            epoch = epoch + 1
            if epoch > cnt_conjugates  :
                d_list.pop(0)
                S_list.pop(0)
                gamma_list.pop(0)


        if linesearch :
            res = minimize_scalar( lambda y : model.primal(flows + y*dk ) , bounds = (0.0,1.0) , tol = 1e-12 )
            gamma = res.x
        else :
            gamma = 2.0/(k + 2)
        
        gamma_list.append(gamma)


        dual_val = model.dual(t, sk_FW)
        max_dual_func_val = max(max_dual_func_val, dual_val)
        
        
        flows = flows + gamma*dk


        # equal to FW gap if dual_val == max_dual_func_val
        primal = model.primal(flows)
        dgap_log.append(primal - max_dual_func_val)
        time_log.append(time.time())

        if stop_by_crit and dgap_log[-1] <= eps_abs:
            optimal = True
            break

    return (
        t,
        flows,
        (dgap_log, np.array(time_log) - time_log[0]),
        optimal,
    )