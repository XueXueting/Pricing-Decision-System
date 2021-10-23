import pickle
import os
from pulp import *
from math import sqrt, log, exp
from math import *
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import xlrd
import random #########to generate random number
import time
import matplotlib.pyplot as plt
# from PricingCode_MDM_actual import UserConstraints

Solver_setup = PULP_CBC_CMD()

# import sys
#UserConstraints = [([900,100,0.3,1,500,0.1,0.3,1,0.9,0.1,0.3,1,0.9,0.1,0.3,1,0.9,0.1,0.3,1],100)]
#UserConstraints = sys.argv[2]


def fun_PriceError_max_mainMDM(p,x,vc,cost,c):
    ##########read data 
    VMaxIter, n, MaxSimIter = vc.shape
    NIter=1
    MaxIter, n, MaxSimIter = p.shape
    optimalprice_iter_price=np.zeros((n,NIter,MaxSimIter))
    profit_MDM_iter=np.zeros((NIter,MaxSimIter))
    profit_MDM_price=np.zeros(MaxSimIter)
    opt_obj_iter=np.zeros((NIter,MaxSimIter))

    point=np.zeros((MaxIter,n+1,MaxSimIter))
    demand=np.zeros((MaxIter,n+1,MaxSimIter))
    #index=np.ones((MaxIter,n+1,MaxSimIter))
    index=np.ones((MaxIter,n+1,MaxSimIter))
    indexlist = [[[0 for t in range(MaxSimIter)] for j in range(n+1)] for i in range(MaxIter)]
    index=np.array(indexlist)
    index.shape



    duration_MDM_estimation=np.zeros(MaxSimIter)
    duration_MDM_opt=np.zeros(MaxSimIter)


    for t in range(MaxSimIter):
        #########preprocess data to reroder the data
        P,sharematrix,indexoforder=data_preprocess(p[:,:,t],x[:,:,t])
    #     P=price
        demand[:,:,t] = sharematrix
        index[:,:,t] = indexoforder
        ##########estimation
        start_mdm_est = time.time()
        #point[:,:,t],error = estimation_Twotypes_l1norm_pulp(index[:,:,t],demand[:,:,t],P)
        point[:,:,t],error = estimation_Monotone_max_l1norm_pulp(index[:,:,t],demand[:,:,t],P)
        end_mdm_est = time.time()
        duration_MDM_estimation[t]=end_mdm_est-start_mdm_est
        ##########optimization, get optimal solution from MDM approach, no price constraint so far
        start_mdm_opt = time.time()
        optx, optdelta, optimalprice_iter_price[:,0,t], opt_obj = optimization_MIP_pulp(cost,demand[:,:,t],point[:,:,t],index[:,:,t],c)
        end_mdm_opt = time.time()
        duration_MDM_opt[t]=end_mdm_opt-start_mdm_opt
        ########evaluate the true profit
        index_a,freq_prod,share_MDM=Share_Calculation(vc[:,:,t],optimalprice_iter_price[:,0,t])
        #profit_MDM=0
        for j in range(n):
            profit_MDM_price[t]= profit_MDM_price[t] + (optimalprice_iter_price[j,0,t]-cost[j])*share_MDM[j]



    duration_mdm_price=duration_MDM_opt+duration_MDM_estimation
    
    return optimalprice_iter_price[:,0,:], profit_MDM_price, duration_mdm_price, point, demand, index

def estimation_Monotone_max_l1norm_pulp(index,demand,P):
    MaxIter, n = P.shape
    Sample=range(MaxIter) #######return a vector from 0 to MaxIter -1, MaxIter is sample size
    Market=range(n+1) ######return a vector from 0 to n, include outside option
    Product=range(n) ######return a vector from 0 to n-1, n is the number of products
    lowerBound = np.zeros((MaxIter,n+1))
    # Preapring an Optimization Model
    model_l1norm = LpProblem("L1 Norm Empirical Minimization", LpMinimize)
    
    ############## Defining decision variables
    point = [  [LpVariable("point{0},{1}".format(i+1,j)) for j in Market] for i in Sample ]
    t = [ [LpVariable("t{0},{1}".format(i+1,j+1)) for j in Product ] for i in Sample]
    u = LpVariable('u', 0.0)
    
    ################## Setting the objective
    #julia code @objective(estimation_model,Min,sum(t[i,j] for i=1:MaxIter,j=1:n))
    obj_var = u
    #sum(t[i][j] for i in Sample for j in Product)
    model_l1norm += obj_var
    ################# Adding constraints   
    for i in Sample:
        model_l1norm += (u >= sum(t[i][j] for j in Product))
    for i in Sample:
        for j in Product:
            model_l1norm += (1-demand[i,n])*demand[i,j]*t[i][j] >= (1-demand[i,n])*point[i][j]- demand[i,j]*point[i][n]- (1-demand[i,n])*demand[i,j]*P[i,j]
            model_l1norm += (1-demand[i,n])*demand[i,j]*t[i][j] >=  -(1-demand[i,n])*point[i][j] + demand[i,j]*point[i][n] + (1-demand[i,n])*demand[i,j]*P[i,j]
    ################## monotonicity constraint
    #########for product 1 to n, index starts from 0 until n-1
    for j in range(n):
        pre=0  ########index starts from 0 until n-1
        i=1
        while i < MaxIter - 1:
            if demand[index[i,j],j]==demand[index[pre,j],j]:
                i=i+1
            else:     
                model_l1norm += point[index[i,j]][j]*demand[index[pre,j],j]<=demand[index[i,j],j]*point[index[pre,j]][j]               
                pre=i
                i=pre+1
    ### monotonicity constraint for outside option, index is n, as the starting index is 0
    pre=0
    i=1
    while i < MaxIter - 1:
        if demand[index[i,n],n]==demand[index[pre,n],n]:
            i=i+1
        else:      
            model_l1norm += (1-demand[index[pre,n]][n])*point[index[i,n]][n]<=(1-demand[index[i,n],n])*point[index[pre,n]][n]
            pre=i
            i=pre+1
    # Solving the optimization problem 
    model_l1norm.solve(Solver_setup)
    
    # Printing the optimal solutions obtained
    pointEst = np.zeros((MaxIter,n+1))
    #print("Optimal Solutions:")
    for i in range(MaxIter):
        for j in range(n+1):     
            pointEst[i,j] = value(point[i][j])
            
            #        print("pointwise estimation of entry \n %s %s is: %g" % (i, j , pointEst[i,j]))
    # Get objective value 
    opt_obj = value(obj_var)
    
    return pointEst,opt_obj;

def fun_valuation_pool_generation_mixed_lognormalUniform(VMaxIter,maxn,MaxSimIter):
    vc_full=np.zeros((VMaxIter,maxn,MaxSimIter))
    V_full=np.zeros((maxn,MaxSimIter))
    ########increase the valuation by index from 1 to 10
    for j in range(maxn):
        for t in range(MaxSimIter):
            V_full[j,t]=(j+1)+random.random()
    for t in range(MaxSimIter):
        #LB[:,t],UB[:,t],vc[:,:,t]=Valuation_Generation(V[:,t],VMaxIter,n)
        #vc[:,:,t]=Valuation_Generation_mixed_NormalUniform(V[:,t],VMaxIter,n)
        vc_full[:,:,t] = Valuation_Generation_mixed_lognormalUniform(V_full[:,t],VMaxIter,maxn)#####first 5 uniform, last 5 lognormal
    return V_full, vc_full

def Valuation_Generation_mixed_lognormalUniform(Vmean,VMaxIter,n):
    Sample=range(VMaxIter) 
    Market=range(n+1) 
    Product=range(n) 
    vc = np.zeros((VMaxIter,n))

    a=0.5
    u_mean=Vmean
    u_var=(1/12)*np.multiply(Vmean,Vmean)
    u_2ndm=np.multiply(Vmean,Vmean)+u_var 
    ul_var=(1/12)*np.multiply(Vmean,Vmean)
    ul_2ndm=np.multiply(Vmean,Vmean)+ul_var
    log_mean = 2*np.log(Vmean)-0.5*np.log(ul_2ndm) ######## \mu =2log(mean)-0.5*log(mean^2+var)
    log_var = np.log(ul_2ndm)-2*np.log(Vmean) #####\var=log(mean^2+var) - log(mean^2)
    log_std=np.zeros(n)
    #######first produdct sample from lognormal, the second from uniform
    nn=1#####indicate the second product
    for i in Sample:
        #first half uniform distributed, second half lognormal
        for j in range(int(n/2)):  
            vc[i,j]=Vmean[j]+Vmean[j]*(-a+random.random()) ######a=0.5mu, b=1.5mu, mean =mu, Variance 1/12(b^2-a^2)=1/12mu^2
        for j in range(int(n/2),n):
            log_std[j] = sqrt(log_var[j])
            vc[i,j]=np.random.lognormal(log_mean[j], log_std[j])            
            
    return vc



def fun_PricePoolGeneration(MaxIter,vc):
    VMaxIter, n, MaxSimIter = vc.shape
    ########generate share
    VG=np.zeros((n,MaxSimIter))
    for t in range(MaxSimIter):
        for j in range(n):
            VG[j,t]=np.mean(vc[:,j,t])
    x=np.zeros((MaxIter,n+1,MaxSimIter))
    p=np.zeros((MaxIter,n,MaxSimIter))
    #while (np.min(x)<0.000001):
    #p=uniform_price_generation_GM(VG,MaxIter,n)   
    for t in range(MaxSimIter):
        for i in range(MaxIter):
            while (np.min(x[i,:,t])<0.00001):
                p[i,:,t]=uniform_price_generation_GM(VG[:,t],1,n)   
                index_a,freq_prod,x[i,:,t]=Share_Calculation(vc[:,:,t],p[i,:,t])

    return p,x


def uniform_price_generation_GM(V,MaxIter,n):    
    Sample, Market, Product =range(MaxIter),range(n+1), range(n) 
    p = np.zeros((MaxIter,n))
    K=7 ##########for MNL,MMNL
    for j in Sample:
        for i in Product:
            #p[i,j]=0.2*k*V[i]*(1-0.1+0.2*random.random()) 
            p[j,i]=V[i]*(1-0.2+0.4*random.random()) 
    return p;

def Share_Calculation(vc,price):####uniform generate valuation
    MaxIter, n = vc.shape
    share=np.zeros(n+1)####including the outside
    maxsurplus=np.zeros(MaxIter)
    index_max=-1*np.ones(MaxIter)
    freq_prod=np.zeros(n)
    for i in range(MaxIter):
        #k=random.randint(3,K) ##### return a random integer number between 3 to K        
        for j in range(n):
            if (vc[i,j]-price[j]>maxsurplus[i]):#for MNL, nested logit
                maxsurplus[i] = vc[i,j]-price[j]
                index_max[i]=j
        if (index_max[i]!= -1) :
            freq_prod[int(index_max[i])]=freq_prod[int(index_max[i])]+1
    for j in range(n):
        share[j]=freq_prod[j]/MaxIter
    share[n]=1-sum(share[0:n])
    return index_max,freq_prod,share;

def fun_incre_price_experiment_index(MaxIterM):

    pind_temp = range(MaxIterM)
    shrink_size = [90, 80, 70, 60, 50, 40, 30, 20, 10, 5]
    output,final_output = [],[]
 
    for i in range(len(shrink_size)):
        pind_temp = list(random.sample(pind_temp, shrink_size[i]))
        output.append( pind_temp)
        
    for i in range(len(shrink_size)):
        final_output.append( output[len(shrink_size) - 1 - i ] )
    return final_output

def main_DataGeneration(VMaxIter,maxn,MaxSimIter,MaxIterM,vsampleN):
    V_full, vc_full = fun_valuation_pool_generation_mixed_lognormalUniform(VMaxIter,maxn,MaxSimIter)
    p_full,x_full=fun_PricePoolGeneration(MaxIterM,vc_full)
    ind = list(range(VMaxIter))
    sample_ind100=random.sample(ind,vsampleN)
    psample_ind=fun_incre_price_experiment_index(MaxIterM)
    return V_full, vc_full, p_full, np.array(sample_ind100), psample_ind

def fun_true_pricing_MIP(vc,sample_ind,cost):
    VMaxIter, n, MaxSimIter = vc.shape
    M=np.zeros(MaxSimIter)
    pOpt=np.zeros((n,MaxSimIter))
    profit_true=np.zeros(MaxSimIter)
    duration_true=np.zeros(MaxSimIter)
    Mcut=len(sample_ind)
    for t in range(MaxSimIter):
        M[t]=np.max(vc[:,:,t])
        start_true = time.time()
        pOpt[:,t],opt_obj=Pricing_Valuation_SAA_ms_pulp(vc[sample_ind.astype(int),:,t])
        end_true = time.time()
        duration_true[t]=end_true-start_true
        #index_a,freq_prod,share=Share_Calculation(vc,pOpt)
        index_a,freq_prod,share_true=Share_Calculation(vc[:,:,t],pOpt[:,t])    
        for j in range(n):
            profit_true[t]= profit_true[t] + (pOpt[j,t]-cost[j])*share_true[j]

    #price_true=pOpt
    return pOpt, profit_true, duration_true

def Pricing_Valuation_SAA_ms_pulp(vc):
    MaxIter, n = vc.shape
    Sample=range(MaxIter) #######return a vector from 0 to MaxIter -1, MaxIter is sample size
    Market=range(n+1) ######return a vector from 0 to n, include outside option
    Product=range(n) ######return a vector from 0 to n-1, n is the number of products
    EXPERIMENT = [(i, j) for i in Sample for j in Product]
    lowerBoundp = np.zeros(n)
    lowerBoundpay = np.zeros((MaxIter,n))
    lowerBoundsp = np.zeros((MaxIter,n))
    
    M=np.max(vc)
    
    prob_1 = LpProblem("Pricing_valuation_saa", LpMaximize)
    p = [LpVariable("pp{0}".format(i+1), 0) for i in Product]
    pay = [[LpVariable("ppay{0},{1}".format(i+1,j+1), 0 )  for j in Product] for i in Sample ]
    sp = [[LpVariable("ssp{0},{1}".format(i+1,j+1), 0 ) for j in Product] for i in Sample]
    y = [[LpVariable("yy{0},{1}".format(i+1,j+1), cat = "Binary") for j in Product] for i in Sample]

    obj_pricing_saa = sum(pay[i][j] for i in Sample for j in Product)/MaxIter
    prob_1 += obj_pricing_saa
    
    ################# Adding constraints
    for i in Sample:
        for j in Product: 
            con1 = pay[i][j] - p[j] + M*(1 - y[i][j])
            prob_1 += con1 >= 0
            prob_1 += pay[i][j] <= p[j]
        for j in Product:
            prob_1 += sp[i][j] == vc[i,j] * y[i][j] - pay[i][j]
        prob_1 += sum(y[i][j] for j in Product) <= 1
        #####################surplus maximize
        for j in Product:
            prob_1 += sum(sp[i][l] for l in Product) >= vc[i,j] - p[j]
    #####################redundant constraint
    for i in Sample:
        for k in Sample:
            prob_1 += sum(sp[i][j] for j in Product) >= sum(vc[i,l]*y[k][l]-pay[k][l] for l in Product)
 
    prob_1.solve(Solver_setup)                           
    # Printing the optimal solutions obtained
    pOpt = np.zeros((n))    
    for j in range(n):
        pOpt[j]=value(p[j])

    # Get objective value 
    opt_obj = value(obj_pricing_saa)
    return pOpt, opt_obj

def fun_shareGeneration(p_full,psample_ind_one,list_Prod_index_one,vc):
    VMaxIter, n, MaxSimIter = vc.shape
    MaxIter=len(psample_ind_one)
    ########generate share
    VG=np.zeros((n,MaxSimIter))
    for t in range(MaxSimIter):
        for j in range(n):
            VG[j,t]=np.mean(vc[:,j,t])
    x=np.zeros((MaxIter,n+1,MaxSimIter))
    p=np.zeros((MaxIter,n,MaxSimIter))
    #while (np.min(x)<0.000001):
    #p=uniform_price_generation_GM(VG,MaxIter,n)   
    for t in range(MaxSimIter):
        for i in range(MaxIter):
            #while (np.min(x[i,:,t])<0.00001):
                #p[i,:,t]=uniform_price_generation_GM(VG[:,t],1,n)  
            p[i,:,t]=p_full[psample_ind_one[i],list_Prod_index_one,t]
            index_a,freq_prod,x[i,:,t]=Share_Calculation(vc[:,:,t],p[i,:,t])

    return p,x

# def fun_PriceError_max_mainMDM(p,x,vc,cost):
#     ##########read data 
#     VMaxIter, n, MaxSimIter = vc.shape
#     NIter=1
#     MaxIter, n, MaxSimIter = p.shape
#     optimalprice_iter_price=np.zeros((n,NIter,MaxSimIter))
#     profit_MDM_iter=np.zeros((NIter,MaxSimIter))
#     profit_MDM_price=np.zeros(MaxSimIter)
#     opt_obj_iter=np.zeros((NIter,MaxSimIter))

#     point=np.zeros((MaxIter,n+1,MaxSimIter))
#     demand=np.zeros((MaxIter,n+1,MaxSimIter))
#     #index=np.ones((MaxIter,n+1,MaxSimIter))
#     index=np.ones((MaxIter,n+1,MaxSimIter))
#     indexlist = [[[0 for t in range(MaxSimIter)] for j in range(n+1)] for i in range(MaxIter)]
#     index=np.array(indexlist)
#     index.shape



#     duration_MDM_estimation=np.zeros(MaxSimIter)
#     duration_MDM_opt=np.zeros(MaxSimIter)


#     for t in range(MaxSimIter):
#         #########preprocess data to reroder the data
#         P,sharematrix,indexoforder=data_preprocess(p[:,:,t],x[:,:,t])
#     #     P=price
#         demand[:,:,t] = sharematrix
#         index[:,:,t] = indexoforder
#         ##########estimation
#         start_mdm_est = time.time()
#         #point[:,:,t],error = estimation_Twotypes_l1norm_pulp(index[:,:,t],demand[:,:,t],P)
#         point[:,:,t],error = estimation_Monotone_max_l1norm_pulp(index[:,:,t],demand[:,:,t],P)
#         end_mdm_est = time.time()
#         duration_MDM_estimation[t]=end_mdm_est-start_mdm_est
#         ##########optimization, get optimal solution from MDM approach, no price constraint so far
#         start_mdm_opt = time.time()
#         optx, optdelta, optimalprice_iter_price[:,0,t], opt_obj = optimization_MIP_pulp(cost,demand[:,:,t],point[:,:,t],index[:,:,t])
#         end_mdm_opt = time.time()
#         duration_MDM_opt[t]=end_mdm_opt-start_mdm_opt
#         ########evaluate the true profit
#         index_a,freq_prod,share_MDM=Share_Calculation(vc[:,:,t],optimalprice_iter_price[:,0,t])
#         #profit_MDM=0
#         for j in range(n):
#             profit_MDM_price[t]= profit_MDM_price[t] + (optimalprice_iter_price[j,0,t]-cost[j])*share_MDM[j]



#     duration_mdm_price=duration_MDM_opt+duration_MDM_estimation
    
#     return optimalprice_iter_price[:,0,:], profit_MDM_price, duration_mdm_price, point, demand, index

def fun_PriceError_mainMDM(p,x,vc,cost):
    ##########read data 
    VMaxIter, n, MaxSimIter = vc.shape
    NIter=1
    MaxIter, n, MaxSimIter = p.shape
    optimalprice_iter_price=np.zeros((n,NIter,MaxSimIter))
    profit_MDM_iter=np.zeros((NIter,MaxSimIter))
    profit_MDM_price=np.zeros(MaxSimIter)
    opt_obj_iter=np.zeros((NIter,MaxSimIter))

    point=np.zeros((MaxIter,n+1,MaxSimIter))
    demand=np.zeros((MaxIter,n+1,MaxSimIter))
    #index=np.ones((MaxIter,n+1,MaxSimIter))
    index=np.ones((MaxIter,n+1,MaxSimIter))
    indexlist = [[[0 for t in range(MaxSimIter)] for j in range(n+1)] for i in range(MaxIter)]
    index=np.array(indexlist)
    index.shape



    duration_MDM_estimation=np.zeros(MaxSimIter)
    duration_MDM_opt=np.zeros(MaxSimIter)


    for t in range(MaxSimIter):
        #########preprocess data to reroder the data
        P,sharematrix,indexoforder=data_preprocess(p[:,:,t],x[:,:,t])
    #     P=price
        demand[:,:,t] = sharematrix
        index[:,:,t] = indexoforder
        ##########estimation
        start_mdm_est = time.time()
        point[:,:,t],error = estimation_Twotypes_l1norm_pulp(index[:,:,t],demand[:,:,t],P)
        end_mdm_est = time.time()
        duration_MDM_estimation[t]=end_mdm_est-start_mdm_est
        ##########optimization, get optimal solution from MDM approach, no price constraint so far
        start_mdm_opt = time.time()
        optx, optdelta, optimalprice_iter_price[:,0,t], opt_obj = optimization_MIP_pulp(cost,demand[:,:,t],point[:,:,t],index[:,:,t],c)
        end_mdm_opt = time.time()
        duration_MDM_opt[t]=end_mdm_opt-start_mdm_opt
        ########evaluate the true profit
        index_a,freq_prod,share_MDM=Share_Calculation(vc[:,:,t],optimalprice_iter_price[:,0,t])
        #profit_MDM=0
        for j in range(n):
            profit_MDM_price[t]= profit_MDM_price[t] + (optimalprice_iter_price[j,0,t]-cost[j])*share_MDM[j]



    duration_mdm_price=duration_MDM_opt+duration_MDM_estimation
    
    return optimalprice_iter_price[:,0,:], profit_MDM_price, duration_mdm_price, point, demand, index

def data_preprocess(p,x):
    ###input paramter in data preprocess :
    price=p
    share=x
    #preprocess data
    MaxIter, n = price.shape
    ###get the index in the order of each column
    sharematrix=share
    indexoforder=np.zeros((MaxIter,n+1),"int")
    for i in range(n+1):
        origin=list(sharematrix[:,i])
        share_sort=sorted(origin)
        ss = []
        for j in range(len(share_sort)):
            ss.insert(j,origin.index(share_sort[j]))
            indexoforder[j,i]=ss[j] ##########get the index of the smallest entry to largest entry in the original array
            #indexoforder[:,i]=sortperm(sharematrix[:,i]) #julia code
            ###sharematrix[(indexoforder[:,0]),0]
    return price,sharematrix,indexoforder;

def estimation_Twotypes_l1norm_pulp(index,demand,P):
    MaxIter, n = P.shape
    Sample=range(MaxIter) #######return a vector from 0 to MaxIter -1, MaxIter is sample size
    Market=range(n+1) ######return a vector from 0 to n, include outside option
    Product=range(n) ######return a vector from 0 to n-1, n is the number of products
    lowerBound = np.zeros((MaxIter,n+1))
    # Preapring an Optimization Model
    model_l1norm = LpProblem("L1 Norm Empirical Minimization", LpMinimize)
    
    ############## Defining decision variables
    point = [  [LpVariable("point{0},{1}".format(i+1,j)) for j in Market] for i in Sample ]
    t = [ [LpVariable("t{0},{1}".format(i+1,j+1)) for j in Product ] for i in Sample]
    
    ################## Setting the objective
    #julia code @objective(estimation_model,Min,sum(t[i,j] for i=1:MaxIter,j=1:n))
    obj_var = sum(t[i][j] for i in Sample for j in Product)
    model_l1norm += obj_var
    ################# Adding constraints   
    for i in Sample:
        for j in Product:
            model_l1norm += (1-demand[i,n])*demand[i,j]*t[i][j] >= (1-demand[i,n])*point[i][j]- demand[i,j]*point[i][n]- (1-demand[i,n])*demand[i,j]*P[i,j]
            model_l1norm += (1-demand[i,n])*demand[i,j]*t[i][j] >=  -(1-demand[i,n])*point[i][j] + demand[i,j]*point[i][n] + (1-demand[i,n])*demand[i,j]*P[i,j]
    ##############add convexity constraint
    #########for product 1 to n, index starts from 0 until n-1
    for j in Product:
        #for i=2:(MaxIter-1)
        pre=0 ####index starts from 0
        i=1
        while i<MaxIter-1:
            pro=i+1
            if demand[index[i,j],j]==demand[index[pre,j],j]:   
                model_l1norm += point[index[i,j]][j]==point[index[pre,j]][j]             
                i=i+1
                pro=i+1
            else:
                while demand[index[i,j],j]==demand[index[pro,j],j]:                  
                    model_l1norm += point[index[pro,j]][j]==point[index[i,j]][j]
                    pro=pro+1
                    if pro>=MaxIter:
                        break
            if pro<= MaxIter-1:
                if demand[index[i,j],j]==demand[index[pro,j],j]:                   
                    model_l1norm += point[index[pro,j]][j]==point[index[i,j]][j]
                    
                else:
                    model_l1norm += (demand[index[i,j],j]-demand[index[pre,j],j])/(demand[index[pro,j],j]-demand[index[pre,j],j])*point[index[pro,j]][j]+(demand[index[pro,j],j]-demand[index[i,j],j])/(demand[index[pro,j],j]-demand[index[pre,j],j])*point[index[pre,j]][j]<=point[index[i,j]][j]
                    
            pre=i
            i=pro
                     
    ###convextiy constraint for outside option, index is n, as the starting index is 0
    pre=0 ####index starts from 0
    i=1 
    while i< MaxIter-1:
        pro=i+1
        if demand[index[i,n],n]==demand[index[pre,n],n]:
            model_l1norm += point[index[i,n]][n]==point[index[pre,n]][n]     
            i=i+1
            pro=i+1
        else:
            while demand[index[i,n],n]==demand[index[pro,n],n]:             
                model_l1norm += point[index[pro,n]][n]==point[index[i,n]][n]     
                pro=pro+1
                if pro>=MaxIter:
                        break
        if pro<= MaxIter-1:
            if demand[index[i,n],n]==demand[index[pro,n],n]:          
                model_l1norm += point[index[pro,n]][n]==point[index[i,n]][n]
                
            else:              
                model_l1norm += (demand[index[i,n],n]-demand[index[pre,n],n])/(demand[index[pro,n],n]-demand[index[pre,n],n])*point[index[pro,n]][n]+(demand[index[pro,n],n]-demand[index[i,n],n])/(demand[index[pro,n],n]-demand[index[pre,n],n])*point[index[pre,n]][n]>=point[index[i,n]][n]
                
        pre=i
        i=pro
    ################## monotonicity constraint
    #########for product 1 to n, index starts from 0 until n-1
    for j in range(n):
        pre=0  ########index starts from 0 until n-1
        i=1
        while i < MaxIter - 1:
            if demand[index[i,j],j]==demand[index[pre,j],j]:
                i=i+1
            else:     
                model_l1norm += point[index[i,j]][j]*demand[index[pre,j],j]<=demand[index[i,j],j]*point[index[pre,j]][j]               
                pre=i
                i=pre+1
    ### monotonicity constraint for outside option, index is n, as the starting index is 0
    pre=0
    i=1
    while i < MaxIter - 1:
        if demand[index[i,n],n]==demand[index[pre,n],n]:
            i=i+1
        else:      
            model_l1norm += (1-demand[index[pre,n]][n])*point[index[i,n]][n]<=(1-demand[index[i,n],n])*point[index[pre,n]][n]
            pre=i
            i=pre+1
    # Solving the optimization problem 
    model_l1norm.solve(Solver_setup)
    
    # Printing the optimal solutions obtained
    pointEst = np.zeros((MaxIter,n+1))
    #print("Optimal Solutions:")
    for i in range(MaxIter):
        for j in range(n+1):     
            pointEst[i,j] = value(point[i][j])
            
            #        print("pointwise estimation of entry \n %s %s is: %g" % (i, j , pointEst[i,j]))
    # Get objective value 
    opt_obj = value(obj_var)
    
    return pointEst,opt_obj;

# def estimation_Monotone_max_l1norm_pulp(index,demand,P):
#     MaxIter, n = P.shape
#     Sample=range(MaxIter) #######return a vector from 0 to MaxIter -1, MaxIter is sample size
#     Market=range(n+1) ######return a vector from 0 to n, include outside option
#     Product=range(n) ######return a vector from 0 to n-1, n is the number of products
#     lowerBound = np.zeros((MaxIter,n+1))
#     # Preapring an Optimization Model
#     model_l1norm = LpProblem("L1 Norm Empirical Minimization", LpMinimize)
    
#     ############## Defining decision variables
#     point = [  [LpVariable("point{0},{1}".format(i+1,j)) for j in Market] for i in Sample ]
#     t = [ [LpVariable("t{0},{1}".format(i+1,j+1)) for j in Product ] for i in Sample]
#     u = LpVariable('u', 0.0)
    
#     ################## Setting the objective
#     #julia code @objective(estimation_model,Min,sum(t[i,j] for i=1:MaxIter,j=1:n))
#     obj_var = u
#     #sum(t[i][j] for i in Sample for j in Product)
#     model_l1norm += obj_var
#     ################# Adding constraints   
#     for i in Sample:
#         model_l1norm += (u >= sum(t[i][j] for j in Product))
#     for i in Sample:
#         for j in Product:
#             model_l1norm += (1-demand[i,n])*demand[i,j]*t[i][j] >= (1-demand[i,n])*point[i][j]- demand[i,j]*point[i][n]- (1-demand[i,n])*demand[i,j]*P[i,j]
#             model_l1norm += (1-demand[i,n])*demand[i,j]*t[i][j] >=  -(1-demand[i,n])*point[i][j] + demand[i,j]*point[i][n] + (1-demand[i,n])*demand[i,j]*P[i,j]
#     ################## monotonicity constraint
#     #########for product 1 to n, index starts from 0 until n-1
#     for j in range(n):
#         pre=0  ########index starts from 0 until n-1
#         i=1
#         while i < MaxIter - 1:
#             if demand[index[i,j],j]==demand[index[pre,j],j]:
#                 i=i+1
#             else:     
#                 model_l1norm += point[index[i,j]][j]*demand[index[pre,j],j]<=demand[index[i,j],j]*point[index[pre,j]][j]               
#                 pre=i
#                 i=pre+1
#     ### monotonicity constraint for outside option, index is n, as the starting index is 0
#     pre=0
#     i=1
#     while i < MaxIter - 1:
#         if demand[index[i,n],n]==demand[index[pre,n],n]:
#             i=i+1
#         else:      
#             model_l1norm += (1-demand[index[pre,n]][n])*point[index[i,n]][n]<=(1-demand[index[i,n],n])*point[index[pre,n]][n]
#             pre=i
#             i=pre+1
#     # Solving the optimization problem 
#     model_l1norm.solve(Solver_setup)
    
#     # Printing the optimal solutions obtained
#     pointEst = np.zeros((MaxIter,n+1))
#     #print("Optimal Solutions:")
#     for i in range(MaxIter):
#         for j in range(n+1):     
#             pointEst[i,j] = value(point[i][j])
            
#             #        print("pointwise estimation of entry \n %s %s is: %g" % (i, j , pointEst[i,j]))
#     # Get objective value 
#     opt_obj = value(obj_var)
    
#     return pointEst,opt_obj;

def optimization_MIP_pulp(cost,demand,point,index,c):
    #point = pointEst
    MaxIter, n = point.shape
    n=n-1
    M=MaxIter
    Sample=range(M) #######return a vector from 0 to M -1, M is sample size
    Market=range(n+1) ######return a vector from 0 to n, include outside option
    Product=range(n) ######return a vector from 0 to n-1, n is the number of products
    ProductEXPERIMENT = [(i, j) for i in Product for j in Sample]
    MarketEXPERIMENT = [(i, j) for i in Market for j in Sample]
    MarketEXPERIMENTZ = [(i, j) for i in Market for j in range(M-1)]
    # Preapring an Optimization Model
    optimization_MIP_model = LpProblem("optimizationMIPMDM", LpMaximize)
    ############## Defining decision variables 
    #####demo: X = LpVariable.dicts('X',range(2), lowBound = 0, upBound = 1, cat = pulp.LpInteger)
    #x[1:(n+1)]>=0
    x = LpVariable.dicts('x', Market, lowBound = 0)  
    #lam[1:(n+1),1:M]>=0
    lam = LpVariable.dicts('lam', MarketEXPERIMENT, lowBound = 0) #####auxiliaray variable lambda
    #z[1:(n+1),1:(M-1)],Bin
    z = LpVariable.dicts('z', MarketEXPERIMENTZ, lowBound = 0, upBound = 1, cat = LpInteger)
    #FI[1:(n+1)]
    FI = LpVariable.dicts('FI', Market, None)
    #delta[1:(n+1)]
    delta = LpVariable.dicts('delta', Market, None)
    #optimalpricce[1:(n)]
    optimalprice=LpVariable.dicts('optimalprice', Product, None)
    ################## Setting the objective
    optimization_MIP_model += lpSum(delta[j] - cost[j]*x[j] for j in Product) - delta[n]
   # Adding constraints
    ###########adding constraints on lambda and z
    for i in range(n+1):
        optimization_MIP_model += lam[(i,1)]-z[(i,1)]<=0
        optimization_MIP_model += lam[(i,M-1)]-z[(i,M-2)]<=0
        for j in range(1,M-1):
            optimization_MIP_model += lam[(i,j)]-(z[(i,j)]+z[(i,j-1)])<=0
    #############second type of constraint, adding the relationship between demand and price
    for i in range(n):
        optimization_MIP_model += lpSum(lam[(i,j)] for j in range(M)) == 1
        optimization_MIP_model += lpSum(z[(i,j)] for j in range(M-1)) == 1                            
        optimization_MIP_model += lpSum(lam[(i,j)]*demand[index[(j,i)],i] for j in range(M)) - x[i] == 0  
        optimization_MIP_model += lpSum(lam[(i,j)]*(point[index[(j,i)],i]/demand[index[(j,i)],i]) for j in range(M)) - FI[i] == 0 
        optimization_MIP_model += lpSum(lam[(i,j)]*point[index[(j,i)],i] for j in range(M)) - delta[i] == 0  
    optimization_MIP_model += lpSum(lam[(n,j)] for j in range(M)) == 1
    optimization_MIP_model += lpSum(z[(n,j)] for j in range(M-1)) == 1
    optimization_MIP_model += lpSum(lam[(n,j)]*demand[index[(j,n)],n] for j in range(M)) - x[n] == 0
    optimization_MIP_model += lpSum(lam[(n,j)]*(point[index[(j,n)],n]/(1-demand[index[(j,n)],n])) for j in range(M)) - FI[n] == 0
    optimization_MIP_model += lpSum(lam[(n,j)]*point[index[(j,n)],n] for j in range(M)) - delta[n] == 0
    optimization_MIP_model += lpSum(x[j] for j in range(n+1)) == 1
    ################ constraints: bound x
    for j in range(n+1):
        optimization_MIP_model += x[j]<=demand[index[(MaxIter-1,j)],j]
        optimization_MIP_model += x[j]>=demand[index[(1,j)],j] 
    ################ constraints: get optimal price
    for j in range(n):
        optimization_MIP_model += optimalprice[j]==FI[j]-FI[n]

    ###############################################################
    
#     for j in range(n):
#         optimization_MIP_model += optimalprice[j]<=1.1*baseprice[j]
#         optimization_MIP_model += optimalprice[j]>=0.9*baseprice[j]
#
#   2-product with 2 constraints: UserConstraints: [([0.9,0.1],10), ([0.5,0.2],20)]

    UserConstraints = c
#     UserConstraints = [([900,100,0.3,1,500,0.1,0.3,1,0.9,0.1,0.3,1,0.9,0.1,0.3,1,0.9,0.1,0.3,1],100)]
    for (c,b) in UserConstraints:
        optimization_MIP_model += lpSum((c[i] * optimalprice[i]) for i in range(n)) <= b
        print(c[i] * optimalprice[i] for i in range(n))
#
    ################################################################

    # Solving the optimization problem
    optimization_MIP_model.solve(Solver_setup)
    # Printing the optimal solutions obtained
    optx = np.zeros((n+1))
    optdelta = np.zeros((n+1))
    optimalprice_vec = np.zeros((n))
    for j in range(n+1):
        optx[j]=x[j].varValue
        optdelta=delta[j].varValue
    for j in range(n):
        optimalprice_vec[j]=optimalprice[j].varValue
    # Get objective value 
    opt_obj = value(optimization_MIP_model.objective)
    #print("\nOptimal value: \t%g" % optimization_MIP_model.objVal)    
    return optx, optdelta, optimalprice_vec, opt_obj;

def fun_mainMNL(p,x,vc,cost):
    ##########read data 
    VMaxIter, n, MaxSimIter = vc.shape
    V_MNL=np.zeros((n,MaxSimIter))
    Price_MNL=np.zeros((n,MaxSimIter))
    profit_MNL=np.zeros(MaxSimIter)
    duration_MNL_estimation=np.zeros(MaxSimIter)
    duration_MNL_opt=np.zeros(MaxSimIter)
#     p=p_origin
#     x=x_origin
    for t in range(MaxSimIter):
        start_mnl_est = time.time()
        V_MNL[:,t],error_MNL=calibration_MNL_pulp(p[:,:,t],x[:,:,t])
        #V_MNL,error_MNL=calibration_MNL_pulp(p,x)
        #beta_MNL,error_MNL_beta=calibration_MNL_pulp_beta(p,x,attr)
        end_mnl_est = time.time()
        duration_MNL_estimation[t]=end_mnl_est-start_mnl_est
        start_mnl_opt = time.time()
        optx_MNL,Price_MNL[:,t],opt = PricingMNL(V_MNL[:,t],cost)
        end_mnl_opt = time.time()
        duration_MNL_opt[t]=end_mnl_opt-start_mnl_opt

        index_a,freq_prod,share_MNL=Share_Calculation(vc[:,:,t],Price_MNL[:,t])
        for j in range(n):
            profit_MNL[t]= profit_MNL[t] + (Price_MNL[j,t]-cost[j])*share_MNL[j]
        profit_MNL
        duration_MNL=duration_MNL_estimation+duration_MNL_opt
    return Price_MNL,profit_MNL,duration_MNL

def calibration_MNL_pulp(p,x): ############
    MaxIter, n = p.shape
    Sample=range(MaxIter) #######return a vector from 0 to MaxIter -1, MaxIter is sample size
    Market=range(n+1) ######return a vector from 0 to n, include outside option
    Product=range(n) ######return a vector from 0 to n-1, n is the number of products
    model_cali_MNL = LpProblem('MNL calibration',LpMinimize)
    ############## Defining decision variables
  
    V = [ LpVariable("V{0}".format(i+1)) for i in Product ]   
    t = [ [ LpVariable("t{0},{1}".format(i+1,j+1)) for j in Product ] for i in Sample ]
    
    ################## Setting the objective  
    obj_var = sum(t[i][j] for i in Sample for j in Product)
    model_cali_MNL += obj_var
    
    ################# Adding constraints
    for i in Sample:
        for j in Product:
            model_cali_MNL += t[i][j]>=V[j]+log(1-sum(x[i,k] for k in range(n)))-log(x[i,j])-p[i,j]
            model_cali_MNL += t[i][j]>=-(V[j]+log(1-sum(x[i,k] for k in range(n)))-log(x[i,j])-p[i,j])
    
    # Solving the optimization problem  
    model_cali_MNL.solve(Solver_setup)
    
    
    VEst = np.zeros((n))
    #print("Optimal Solutions:")
    for j in range(n):      
        VEst[j] = value(V[j])
    opt_obj=value(obj_var)
    return VEst,opt_obj;

def PricingMNL(V,cost): #############calcualte true optimal pricing with MNL choice model
    from cvxopt import solvers, blas, matrix, spmatrix, spdiag, log, exp, div
    #solvers.options['show_progress'] = False    
    n = len(V)
    n = n+1
    V0=np.zeros((n,1))
    for j in range(n-1):
        V0[j]=V[j]
    VM = matrix(np.asmatrix(V0)) ######define V matrix , which is the deterministic utility for n+1 products, with outside product 0 value
    cost0=np.zeros((n,1))
    for j in range(n-1):
        cost0[j]=cost[j]
    costM=matrix(np.asmatrix(cost0))
    I0=np.zeros((n,1))
    I0[n-1,0]=1
    IM=matrix(np.asmatrix(I0)) ########the coefficent of log(x), since we only have log(x_0) 
    
    G = matrix(np.asmatrix(-np.identity(n)))
    h = matrix(np.asmatrix(np.zeros((n,1))))
    A, b = matrix(1.0, (1,n)), matrix(1.0)
    # maximize    VM'*x-x'*log x+IM*log(x)
    #####equivalent to minimize -VM'*x+x'*log x-IM'*log(x)
    # subject to  G*x <= h
    #             A*x = b
    #
    # variable x (n).
    def F(x=None, z=None):
        if x is None: return 0, matrix(1.0, (n,1))   
        if min(x) <= 0: return None
        f =  costM.T*x - VM.T*x +x.T*log(x) - IM.T * log(x)
        grad = costM -VM + 1.0 + log(x) - spdiag(IM) * x**-1
        if z is None: return f, grad.T       
        H =  spdiag(z[0]* (x**-1 + spdiag(IM) * x**-2)) 
        return f, grad.T, H
    sol = solvers.cp(F, G, h, A=A, b=b)
    x = sol['x']
    price=np.zeros(n-1)
    for j in range(n-1):
        price[j]=V[j]+log(x[n-1])-log(x[j])
    profit=0
    for j in range(n-1):
        profit=profit+(price[j]-cost[j])*x[j]
    #print(x)
    print('true x = %s' % str(x))
    print('true optprice = %s' % str(price))
    print('true profit = %s' % str(profit))

    return x, price, profit

    
def fun_avg(p_origin,x_origin,vc,cost):
    VMaxIter, n, MaxSimIter = vc.shape
    profit_avg=np.zeros(MaxSimIter)
    AvgP=np.zeros((n,MaxSimIter))
    for t in range(MaxSimIter):
        for j in range(n):
            AvgP[j,t]=np.mean(vc[:,j,t])    
   
        #index_a,freq_prod,share=Share_Calculation(vc,pOpt)
        index_a,freq_prod,share_avg=Share_Calculation(vc[:,:,t],AvgP[:,t])    
        for j in range(n):
            profit_avg[t]= profit_avg[t] + (AvgP[j,t]-cost[j])*share_avg[j]

    #price_true=pOpt
    return profit_avg

def fun_evaluate_optimality_gap(profit_true,profit_MDM_price,profit_MNL):
    MaxSimIter = profit_true.shape[0]
    ################evaluate the optimality gap
    gap_MDM_price=np.zeros(MaxSimIter)
    gap_MNL=np.zeros(MaxSimIter)
    #gap_avg=np.zeros(MaxSimIter)
    for t in range(MaxSimIter):
        gap_MDM_price[t]=profit_MDM_price[t]/profit_true[t]
        gap_MNL[t]=profit_MNL[t]/profit_true[t]
        #gap_avg[t]=profit_avg[t]/profit_true[t]
    return gap_MDM_price, gap_MNL

import cvxpy as cp

def Pricing_MNL_ECOS(val, cc):
    n = len(val)
    x = cp.Variable(n)
    alpha = cp.Variable(n)
    beta, gamma = cp.Variable(1), cp.Variable(1)
    
    exp_constraints_alpha = [ cp.constraints.exponential.ExpCone(alpha[j], x[j], 1) for j in range(n) ]
    constr_beta =  cp.constraints.exponential.ExpCone(beta, 1, 1 - np.ones(n).T @ x)
    constr_gamma = cp.constraints.exponential.ExpCone(gamma, 1 - np.ones(n).T @ x, 1 )
    constraints_aggr = exp_constraints_alpha + [constr_beta] + [constr_gamma]+ [np.ones(n).T @ x <= 1 ] + [0 <= x]
    obj_var = np.array(val).T @ x - np.array(cc).T @ x
    prob = cp.Problem(cp.Maximize( obj_var + np.ones(n).T @ alpha + beta+ gamma ),
                      constraints_aggr)
    
    prob.solve(solver = 'ECOS')
    
    return x.value,  obj_var.value

def Share_calculation_MNL(v,p,alpha = 1):
    n = len(v)
    output = np.zeros(n+1)
    down_part = 1 + sum(  np.exp(v[j] - alpha * p[j] ) for j in range(n)  )
    for j in range(n):
        output[j] = np.exp(v[j] - alpha * p[j]) / down_part
    output[n] = 1 - sum( output[j] for j in range(n))
    return output


def MNL_share_geneartion(val, price, alpha=1):
    m,n = price.shape
    share = np.zeros((m,n+1))
    for k in range(m):
        share[k,:] = Share_calculation_MNL(val ,price[k,:], alpha = alpha)
    return share

def pricePurturb_geneartion_uniform(V,MaxIter,eta):    
    n=len(V)
    Sample, Market, Product =range(MaxIter),range(n+1), range(n) 
    p = np.zeros((MaxIter,n))
    for j in Sample:
        for i in Product:            
            p[j,i]=V[i]*(1-eta+2*eta*random.random()) 
    return p;


# def MNL_price_to_share(val, price):
#     n = len(val)
#     down_part = 1 + sum( exp(val[j] - price[j] ) for j in range(n))
#     output = np.zeros(n+1)
#     for j in range(n):
#         output[j] = exp(val[j] - price[j])/ down_part
#     output[n] = 1 - sum(output[j] for j in range(n)) 
#     return output


def Get_Elasticity(p,x):
    MaxIter, n = p.shape
    ElasticityM = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            ######change of price i incur the change of share j
            ElasticityM[i,j]=((x[j*4+2,i]-x[0,i])/x[0,i])/((p[j*4+2,j]-p[0,j])/p[0,j]) ######second price change is always +5%
    return ElasticityM



############Profit Evaluation using Elasticity matrix
def share_evaluation_Elasticity(price,ElasticityM,baseprice,baseshare):
    n=len(baseprice)
    share=np.zeros(n+1)
    sums=0
    for i in range(n):
        share[i]=baseshare[i]
        for j in range(n):
            share[i]=share[i] + ElasticityM[i,j]*(price[j]-baseprice[j])*baseshare[i]/baseprice[j]
        sums=sums+share[i]
    share[n]=1-sums
    return share
    
def QP_Elasticity(ElasticityM,baseprice, baseshare,cost): #########QP model to sovle for optimal prices
    from cvxopt import solvers, blas, matrix, spmatrix, spdiag, log, exp, div
    #solvers.options['show_progress'] = False    
    n=len(baseprice)
    Q=np.zeros((n,n))
    q=np.zeros(n)    
    for i in range(n):
        for j in range(n):
            Q[i,j]=baseshare[i]*ElasticityM[i,j]/baseprice[j]
    for j in range(n):
        q[j]=baseshare[j]+sum(baseshare[i]*(baseprice[i]-cost[i])*ElasticityM[i,j]/baseprice[j] for i in range(n))
    qv=matrix(np.asarray(q))
    QM=matrix(np.asmatrix(Q))    
    G0=np.concatenate((np.identity(n), -1*np.identity(n)), axis=0)
    h0=np.concatenate((baseprice*0.1, 0.1*baseprice), axis=0)
    
    G = matrix(np.asmatrix(G0))
    h = matrix(np.array(h0))
    
    #A, b = matrix(1.0, (1,n)), matrix(1.0)
    # maximize    x'*Q*x+q'*x
    #########equivalanet to minimize -x'*Q*x-q'*x
    #####equivalent to minimize -VM'*x+x'*log x-IM'*log(x)
    # subject to  G*x <= h
    #             A*x = b
    #
    # variable x (n).
    def F(x=None, z=None): #x denote deltap, dp
        if x is None: return 0, matrix(0.0, (n,1))   
        #if min(dp) <=-0.1*baseprice : return None
        f =  - qv.T*x - x.T*QM*x 
        grad = - qv - (QM + QM.T)*x #+ log(x) - spdiag(IM) * x**-1
        if z is None: return f, grad.T       
        H =  - QM - QM.T 
        return f, grad.T, H
    sol = solvers.cp(F, G, h)
    dp = sol['x']
    price=np.zeros(n)
    for j in range(n):
        price[j]=baseprice[j]+dp[j]
    
    #print(x)
    print('true optprice = %s' % str(price))
    return price


def calibration_NL_pulp(p,x,nk): ############
    MaxIter, n = p.shape
    K = len(nk)
    Sample=range(MaxIter) #######return a vector from 0 to MaxIter -1, MaxIter is sample size
    Market=range(n+1) ######return a vector from 0 to n, include outside option
    Product=range(n) ######return a vector from 0 to n-1, n is the number of products
    ################get Q
    Q=np.zeros((MaxIter,K))
    for i in Sample:
        for k in range(K):
            Q[i,k]=sum(x[i,sum(nk[0:k]):sum(nk[0:k+1])])
    EXPERIMENT = [(i, j) for i in Sample for j in Product]
    calibration_NL_model = LpProblem("calibrationNL", LpMinimize)
    ############## Defining decision variables    
    #V[1:n]
    V = LpVariable.dicts('V',Product, None)
    invtau = LpVariable.dicts('invtau',range(K), lowBound = 1)
    #t[1:MaxIter,1:n]
    t = LpVariable.dicts('t',EXPERIMENT, None)    #####auxiliaray variable
    ################## Setting the objective
    calibration_NL_model += lpSum(t[(i,j)] for i in Sample for j in Product)
    ################# Adding constraints
    for i in Sample:
        for k in range(K):
            for j in range(nk[k]):
                calibration_NL_model += t[(i,sum(nk[0:k])+j)] >= (V[sum(nk[0:k])+j]+(invtau[k]-1)*(log(1-sum(x[(i,l)] for l in range(n)))-log(Q[i,k]))+log(1-sum(x[(i,l)] for l in range(n)))-log(x[i,sum(nk[0:k])+j])-p[i,sum(nk[0:k])+j])
                calibration_NL_model += t[(i,sum(nk[0:k])+j)] >= -(V[sum(nk[0:k])+j]+(invtau[k]-1)*(log(1-sum(x[(i,l)] for l in range(n)))-log(Q[i,k]))+log(1-sum(x[(i,l)] for l in range(n)))-log(x[i,sum(nk[0:k])+j])-p[i,sum(nk[0:k])+j])
    # Solving the optimization problem
    calibration_NL_model.solve()
    VEst = np.zeros((n))
    InvTauEst = np.zeros((K))
    #print("Optimal Solutions:")
    for j in range(n):      
        VEst[j]=V[j].varValue
    for k in range(K):      
        InvTauEst[k]=invtau[k].varValue
    opt_obj=calibration_NL_model.objective
    return VEst,InvTauEst, opt_obj;

def PricingNL(V,nk,v_tau,cost): #############calcualte true optimal pricing with NL choice model
    from cvxopt import solvers, blas, matrix, spmatrix, spdiag, log, exp, div
    #solvers.options['show_progress'] = False
    ################each entry in nk provide the number of products in each nest
    n=sum(nk) ######number of product   
    K=len(nk) ####number of nests
    V0=np.zeros((n+1,1))
    for j in range(n):
        V0[j]=V[j]
    VM = matrix(np.asmatrix(V0)) ######define V matrix , which is the deterministic utility for n products, with outside product 0 value
    cost0=np.zeros((n+1,1))
    for j in range(n):
        cost0[j]=cost[j]
    costM=matrix(np.asmatrix(cost0))
    I0=np.zeros((n+1,1))
    I0[n,0]=1
    IM=matrix(np.asmatrix(I0)) ########the coefficent of functions relate to (x_0) ,IM.T * x=x_0
    
    one_row=np.c_[np.ones((1,nk[0])),np.zeros((1,n+1-nk[0]))]
    IQ=one_row
    for k in range(K-1):
        one_row=np.c_[np.zeros((1,sum(nk[0:(k+1)]))),np.ones((1,nk[k+1])),np.zeros((1,n+1-sum(nk[0:(k+2)])))]
        IQ=np.r_[IQ,one_row]
    IQM=matrix(np.asmatrix(IQ))
    TauM_v=np.zeros((K,1))
    for k in range(K):
        TauM_v[k]=(1-v_tau[k])/v_tau[k]
    TauM=matrix(np.asmatrix(TauM_v)) #####a column vector
    ITau_v=np.zeros((K,1))
    for k in range(K):
        ITau_v[k]=1/v_tau[k]
    ITauM=matrix(np.asmatrix(ITau_v)) ####a column vector
    
    RI=np.identity(n+1)
    RI[n,n]=0
    RIM=matrix(np.asmatrix(RI))#########exclude outside option
    
    OneK_v=np.ones((K,1))
    OneK=matrix(np.asmatrix(OneK_v)) ####a column vector
    
    Onen_v=np.ones((n+1,1))
    Onen_v[n]=0    
    Onen=matrix(np.asmatrix(Onen_v)) ####a column vector
    
    ################define the coefficient matrix will be used in hessian matrix
    ##########define the coefficient of 1/(1-sum Q)
    one_sub=(1/v_tau[0]+1/v_tau[0])*np.ones((nk[0],nk[0]))
    for j in range(K-1):
        one_sub_col=(1/v_tau[0]+1/v_tau[j+1])*np.ones((nk[0],nk[j+1]))
        one_sub=np.c_[one_sub,one_sub_col]
    one_sub=np.c_[one_sub,np.zeros((nk[0],1))]
    Hk=one_sub
    for k in range(K-1):        
        one_sub=(1/v_tau[k+1]+1/v_tau[0])*np.ones((nk[k+1],nk[0]))
        for j in range(K-1):
            one_sub_col=(1/v_tau[k+1]+1/v_tau[j+1])*np.ones((nk[k+1],nk[j+1]))
            one_sub=np.c_[one_sub,one_sub_col]
        one_sub=np.c_[one_sub,np.zeros((nk[k+1],1))]
        Hk=np.r_[Hk,one_sub]
    Hk=np.r_[Hk,np.zeros((1,n+1))]
    HkM=matrix(np.asmatrix(Hk))
    ##########define the coefficient of sum ItauM'*Q/(1-sum Q)^2
    Hk2=np.c_[np.ones((n,n)),np.zeros((n,1))]
    Hk2=np.r_[Hk2,np.zeros((1,n+1))]
    Hk2M=matrix(np.asmatrix(Hk2))
   
    ##########define the coefficient of 1/Qk
    HD=((1-v_tau[0])/v_tau[0])*np.c_[np.ones((nk[0],nk[0])),np.zeros((nk[0],n+1-nk[0]))]    
    for k in range(K-1):
        OneSub=((1-v_tau[k+1])/v_tau[k+1])*np.c_[np.zeros((nk[k+1],sum(nk[0:k+1]))),np.ones((nk[k+1],nk[k+1])),np.zeros((nk[k+1],n+1-sum(nk[0:(k+2)])))]  
        HD=np.r_[HD,OneSub]      
    HD=np.r_[HD,np.zeros((1,n+1))]
    HDM=matrix(np.asmatrix(HD))
    
    ##############the last colum and last row in hessian
    LastCol=np.matmul(np.transpose(IQ),ITau_v)
    LastCol_sub=LastCol[0:n]
    HL=np.c_[np.zeros((n,n)),LastCol_sub]
    HL=np.r_[HL,np.transpose(LastCol)]
    HLM=matrix(np.asmatrix(HL))
    
    G = matrix(np.asmatrix(-np.identity(n+1))) ##########x_i >= 0, for i = 0,..,n
    h = matrix(np.asmatrix(np.zeros((n+1,1)))) 
    A, b = matrix(1.0, (1,n+1)), matrix(1.0)   ##########sum_{i=0}^n x_i =1, for i = 0,..,n
    # maximize    VM'*x-x'*log x+IM*log(x)
    #####equivalent to minimize -VM'*x+x'*log x-IM'*log(x)
    # subject to  G*x <= h
    #             A*x = b
    #
    # variable x (n).
    def F(x=None, z=None):
        if x is None: return 0, matrix(1.0, (n+1,1))   
        if min(x) <= 0: return None        
        Q= IQM*x
        f =  costM.T*x-VM.T*x +(RIM*x).T*log(x) + (spdiag(TauM) *Q).T*log(Q) - (ITauM.T*Q)*log(IM.T*x)
        grad = costM -VM + Onen + RIM*log(x)+ IQM.T*(spdiag(TauM)*(1+log(Q))) - log(IM.T*x)*IQM.T*ITauM - IM*(ITauM.T*Q)*(IM.T*x)**-1
        if z is None: return f, grad.T 
        H =  spdiag(RIM*(x**-1)) + HDM*spdiag(IQM.T*(Q**-1)) + spdiag(IM)*(ITauM.T*Q)*(IM.T*x)**-2 - HLM *(IM.T*x)**-1
  
        ####################only with x_jk no x_0
        #f =  -VM.T*x +(RIM*x).T*log(x) + (spdiag(TauM) *Q).T*log(Q) - (ITauM.T*Q)*log(1-OneK.T*Q)
        #grad = -VM + Onen + RIM*log(x)+ IQM.T*(spdiag(TauM)*(1+log(Q))) - log(1-OneK.T*Q)*IQM.T*ITauM + Onen*(ITauM.T*Q)*(1-OneK.T*Q)**-1 
        #if z is None: return f, grad.T       
        #H =  spdiag(RIM*(x**-1)) + HDM*spdiag(IQM.T*(Q**-1)) + ((1-OneK.T*Q)**-1)*HkM + (ITauM.T*Q)*((1-OneK.T*Q)**-2)*Hk2M
        return f, grad.T, z[0]* H
    sol = solvers.cp(F, G, h, A=A, b=b)
    solx = sol['x']
    Qv=np.matmul(IQ,solx)
    
    TauM_vD=np.zeros((K,K))
    for k in range(K):
        TauM_vD[k,k]=TauM_v[k]
    p_logQ=np.matmul(TauM_vD,np.log(Qv)) #####return a column vector with K dimention
    p_logQ_n1=np.matmul(np.transpose(IQ),p_logQ) #####return a column vector with n+1 dimention
    p_logQ_n=p_logQ_n1[0:n]
    
    p_logx0_n1=np.matmul(np.transpose(IQ),ITau_v)
    p_logx0_n=p_logx0_n1[0:n]
    #########calcualte price
    price=np.zeros(n)
    for j in range(n):
        price[j]=V[j]+np.log(solx[n])*p_logx0_n[j] - p_logQ_n[j] - np.log(solx[j])
    profit=0
    for j in range(n):
        profit=profit+(price[j]-cost[j])*solx[j]
    #print(x)
    
    print('true optprice = %s' % str(price))
    print('true x = %s' % str(solx))
    print('true profit = %s' % str(profit))

    return solx, price, profit
    


def optimization_MIP_pulp_GM(cost,demand,point,index,baseprice,c):
    #point = pointEst
    MaxIter, n = point.shape
    n=n-1
    M=MaxIter
    Sample=range(M) #######return a vector from 0 to M -1, M is sample size
    Market=range(n+1) ######return a vector from 0 to n, include outside option
    Product=range(n) ######return a vector from 0 to n-1, n is the number of products
    ProductEXPERIMENT = [(i, j) for i in Product for j in Sample]
    MarketEXPERIMENT = [(i, j) for i in Market for j in Sample]
    MarketEXPERIMENTZ = [(i, j) for i in Market for j in range(M-1)]
    # Preapring an Optimization Model
    optimization_MIP_model = LpProblem("optimizationMIPMDM", LpMaximize)
    ############## Defining decision variables 
    #####demo: X = LpVariable.dicts('X',range(2), lowBound = 0, upBound = 1, cat = pulp.LpInteger)
    #x[1:(n+1)]>=0
    x = LpVariable.dicts('x', Market, lowBound = 0)  
    #lam[1:(n+1),1:M]>=0
    lam = LpVariable.dicts('lam', MarketEXPERIMENT, lowBound = 0) #####auxiliaray variable lambda
    #z[1:(n+1),1:(M-1)],Bin
    z = LpVariable.dicts('z', MarketEXPERIMENTZ, lowBound = 0, upBound = 1, cat = LpInteger)
    #FI[1:(n+1)]
    FI = LpVariable.dicts('FI', Market, None)
    #delta[1:(n+1)]
    delta = LpVariable.dicts('delta', Market, None)
    #optimalpricce[1:(n)]
    optimalprice=LpVariable.dicts('optimalprice', Product, None)
    ################## Setting the objective
    optimization_MIP_model += lpSum(delta[j] - cost[j]*x[j] for j in Product) - delta[n]
   # Adding constraints
    ###########adding constraints on lambda and z
    for i in range(n+1):
        optimization_MIP_model += lam[(i,1)]-z[(i,1)]<=0
        optimization_MIP_model += lam[(i,M-1)]-z[(i,M-2)]<=0
        for j in range(1,M-1):
            optimization_MIP_model += lam[(i,j)]-(z[(i,j)]+z[(i,j-1)])<=0
    #############second type of constraint, adding the relationship between demand and price
    for i in range(n):
        optimization_MIP_model += lpSum(lam[(i,j)] for j in range(M)) == 1
        optimization_MIP_model += lpSum(z[(i,j)] for j in range(M-1)) == 1                            
        optimization_MIP_model += lpSum(lam[(i,j)]*demand[index[(j,i)],i] for j in range(M)) - x[i] == 0  
        optimization_MIP_model += lpSum(lam[(i,j)]*(point[index[(j,i)],i]/demand[index[(j,i)],i]) for j in range(M)) - FI[i] == 0 
        optimization_MIP_model += lpSum(lam[(i,j)]*point[index[(j,i)],i] for j in range(M)) - delta[i] == 0  
    optimization_MIP_model += lpSum(lam[(n,j)] for j in range(M)) == 1
    optimization_MIP_model += lpSum(z[(n,j)] for j in range(M-1)) == 1
    optimization_MIP_model += lpSum(lam[(n,j)]*demand[index[(j,n)],n] for j in range(M)) - x[n] == 0
    optimization_MIP_model += lpSum(lam[(n,j)]*(point[index[(j,n)],n]/(1-demand[index[(j,n)],n])) for j in range(M)) - FI[n] == 0
    optimization_MIP_model += lpSum(lam[(n,j)]*point[index[(j,n)],n] for j in range(M)) - delta[n] == 0
    optimization_MIP_model += lpSum(x[j] for j in range(n+1)) == 1
    ################ constraints: bound x
    for j in range(n+1):
        optimization_MIP_model += x[j]<=demand[index[(MaxIter-1,j)],j]
        optimization_MIP_model += x[j]>=demand[index[(1,j)],j] 
    ################ constraints: get optimal price
    for j in range(n):
        optimization_MIP_model += optimalprice[j]==FI[j]-FI[n] 
#     for j in range(n):
#         optimization_MIP_model += optimalprice[j]<=1.1*baseprice[j]
#         optimization_MIP_model += optimalprice[j]>=0.9*baseprice[j]
        
   #   2-product with 2 constraints: UserConstraints: [([0.9,0.1],10), ([0.5,0.2],20)]

    UserConstraints = c
    for (c,b) in UserConstraints:
        optimization_MIP_model += lpSum((c[i] * optimalprice[i]) for i in range(n)) <= b
        print(c[i] * optimalprice[i] for i in range(n))
        
        
    # Solving the optimization problem
    optimization_MIP_model.solve()
    # Printing the optimal solutions obtained
    optx = np.zeros((n+1))
    optdelta = np.zeros((n+1))
    optimalprice_vec = np.zeros((n))
    for j in range(n+1):
        optx[j]=x[j].varValue
        optdelta=delta[j].varValue
    for j in range(n):
        optimalprice_vec[j]=optimalprice[j].varValue
    # Get objective value 
    opt_obj = value(optimization_MIP_model.objective)
    #print("\nOptimal value: \t%g" % optimization_MIP_model.objVal)    
    return optx, optdelta, optimalprice_vec, opt_obj;

def prediction_MDM_pulp(demand,point,index,p_out):### return the predicted market share under p_out
    MaxIter, n = point.shape
    n=n-1
    M=MaxIter
    Sample=range(M) #######return a vector from 0 to M -1, M is sample size
    Market=range(n+1) ######return a vector from 0 to n, include outside option
    Product=range(n) ######return a vector from 0 to n-1, n is the number of products
    ProductEXPERIMENT = [(i, j) for i in Product for j in Sample]
    MarketEXPERIMENT = [(i, j) for i in Market for j in Sample]
    MarketEXPERIMENTZ = [(i, j) for i in Market for j in range(M-1)]
    # Preapring an Optimization Model
    validation_MDM_model = LpProblem("validationMDM", LpMinimize)
    ############## Defining decision variables 
    #####demo: X = LpVariable.dicts('X',range(2), lowBound = 0, upBound = 1, cat = pulp.LpInteger)
    #x[1:(n+1)]>=0
    x = LpVariable.dicts('x', Market, lowBound = 0)  
    #lam[1:(n+1),1:M]>=0
    lam = LpVariable.dicts('lam', MarketEXPERIMENT, lowBound = 0) #####auxiliaray variable lambda
    #z[1:(n+1),1:(M-1)],Bin
    z = LpVariable.dicts('z', MarketEXPERIMENTZ, lowBound = 0, upBound = 1, cat = LpInteger)
    #FI[1:(n+1)]
    FI = LpVariable.dicts('FI', Market, None)
    #dv[1:n]
    dv = LpVariable.dicts('dv', Product, None)
    ################## Setting the objective
    validation_MDM_model += lpSum(dv[j] for j in Product)
    ################# Adding constraints
    ###########adding constraints on lambda and z
    for i in range(n+1):
        validation_MDM_model += lam[(i,1)]-z[(i,1)]<=0
        validation_MDM_model += lam[(i,M-1)]-z[(i,M-2)]<=0
        for j in range(1,M-1):
            validation_MDM_model += lam[(i,j)]-(z[(i,j)]+z[(i,j-1)])<=0
    #############second type of constraint, adding the relationship between demand and price
    for i in range(n):
        validation_MDM_model += lpSum(lam[(i,j)] for j in range(M)) == 1
        validation_MDM_model += lpSum(z[(i,j)] for j in range(M-1)) == 1                            
        validation_MDM_model += lpSum(lam[(i,j)]*demand[index[(j,i)],i] for j in range(M)) - x[i] == 0  
        validation_MDM_model += lpSum(lam[(i,j)]*(point[index[(j,i)],i]/demand[index[(j,i)],i]) for j in range(M)) - FI[i] == 0 
        validation_MDM_model += dv[i] >= FI[i]-FI[n]-p_out[i]  
        validation_MDM_model += dv[i] >= -(FI[i]-FI[n]-p_out[i])  
    validation_MDM_model += lpSum(lam[(n,j)] for j in range(M)) == 1
    validation_MDM_model += lpSum(z[(n,j)] for j in range(M-1)) == 1
    validation_MDM_model += lpSum(lam[(n,j)]*demand[index[(j,n)],n] for j in range(M)) - x[n] == 0
    validation_MDM_model += lpSum(lam[(n,j)]*(point[index[(j,n)],n]/(1-demand[index[(j,n)],n])) for j in range(M)) - FI[n] == 0
    validation_MDM_model += lpSum(x[j] for j in range(n+1)) == 1
    # Solving the optimization problem
    validation_MDM_model.solve()
    # Printing the optimal solutions obtained
    x_out = np.zeros((n+1))
    for j in range(n+1):
        x_out[j]=x[j].varValue
    # Get objective value 
    prediction_error = validation_MDM_model.objective
    #print("\nOptimal value: \t%g" % optimization_MIP_model.objVal)    
    return x_out;

def prediction_MNL_pulp(V,p_out):
    n=len(V)
    x_out=np.zeros((n+1))
    for i in range(n):
        x_out[i]=exp(V[i]-p_out[i])/(1+sum(exp(V[j]-p_out[j]) for j in range(n)))
    x_out[n]=1-sum(x_out[j] for j in range(n))
    return x_out;

def prediction_NL_pulp(V,v_Invtau,nk,p_out):#######input inverse of tau
    n=len(V)
    K=len(nk)
    x_out=np.zeros((n+1))        
    EQ=np.zeros((K,1)) #########store sum_{i in same k} exp(V_i-price_i) 
    for k in range(K):
        EQ[k]=0
        for i in range(nk[k]):
            EQ[k]=EQ[k]+np.exp(V[sum(nk[0:k])+i]-p_out[sum(nk[0:k])+i])
    x_one_sum=1  
    for k in range(K):
        x_one_sum=x_one_sum+np.power(EQ[k],1/v_Invtau[k])
    for k in range(K):
        for i in range(nk[k]):
            x_out[sum(nk[0:k])+i]=np.exp(V[sum(nk[0:k])+i]-p_out[sum(nk[0:k])+i])*np.power(EQ[k],1/v_Invtau[k]-1)/x_one_sum
    x_out[n]=1/x_one_sum  

    return x_out;

def cross_validation_MDM_pulp(p_in, x_in, p_out, x_out, cost): ############MaxSimIter(=50) denotes the number of cross validation experiment     
    #m_in=80   
    #m_out=1
    #MaxSimIter=50 #############change the outside index
    m_in, n,  MaxSimIter = p_in.shape
    m_out, n,  MaxSimIter = p_out.shape    
    #########declare variable
    profit_out_true=np.zeros((m_out,MaxSimIter))    
    x_out_fit_exp=np.zeros((m_out,n+1,MaxSimIter))
    profit_out_fit_exp=np.zeros((m_out,MaxSimIter))    
    error_predict_exp=np.zeros((m_out,MaxSimIter)) ###############record the prediction error
    error_predict_MDM_MSE=np.zeros((m_out,MaxSimIter)) ###############record the prediction error
    for r in range(MaxSimIter):        
        #################
        price,sharematrix,indexoforder=data_preprocess(p_in[:,:,r],x_in[:,:,r])
        P=price
        demand = sharematrix
        index = indexoforder
        ##########estimation
        point,error = estimation_Twotypes_l1norm_pulp(index,demand,P)
        for i in range(m_out):            
            x_out_fit_exp[i,:,r] = prediction_MDM_pulp(demand,point,index,p_out[i,:,r])
            profit_out_fit_exp[i,r]=sum((p_out[i,j,r]-cost[j])*x_out_fit_exp[i,j,r] for j in range(n));
            ############calculate prediction error
            #####calculate true profi
            profit_out_true[i,r]= sum((p_out[i,j,r]-cost[j])*x_out[i,j,r] for j in range(n))
            error_predict_exp[i,r]=fabs(profit_out_fit_exp[i,r]-profit_out_true[i,r])/profit_out_true[i,r]
            for j in range(n+1):
                error_predict_MDM_MSE[i,r]=error_predict_MDM_MSE[i,r]+np.square(x_out_fit_exp[i,j,r]-x_out[i,j,r])
            error_predict_MDM_MSE[i,r]=error_predict_MDM_MSE[i,r]/(n+1)
    return profit_out_fit_exp, x_out_fit_exp, error_predict_exp,error_predict_MDM_MSE;

def cross_validation_MNL_pulp(p_in, x_in, p_out, x_out, cost): ############MaxSimIter(=50) denotes the number of cross validation experiment     
    #m_in=80   
    #m_out=1
    #MaxSimIter=50 #############change the outside index
    m_in, n,  MaxSimIter = p_in.shape
    m_out, n,  MaxSimIter = p_out.shape    
    #########declare variable
    profit_out_true=np.zeros((m_out,MaxSimIter))    
    x_out_fit_MNL=np.zeros((m_out,n+1,MaxSimIter))    
    profit_out_fit_MNL=np.zeros((m_out,MaxSimIter))        
    error_predict_MNL=np.zeros((m_out,MaxSimIter)) ###############record the prediction error
    error_predict_MNL_MSE=np.zeros((m_out,MaxSimIter)) ###############record the prediction error
    for r in range(MaxSimIter):        
        #################use in sample to do estimation
        V_est,error = calibration_MNL_pulp(p_in[:,:,r],x_in[:,:,r])        
        for i in range(m_out):            
            x_out_fit_MNL[i,:,r] = prediction_MNL_pulp(V_est,p_out[i,:,r])
            profit_out_fit_MNL[i,r]=sum((p_out[i,j,r]-cost[j])*x_out_fit_MNL[i,j,r] for j in range(n));
            ############calculate prediction error
            #####calculate true profi
            profit_out_true[i,r]= sum((p_out[i,j,r]-cost[j])*x_out[i,j,r] for j in range(n))
            error_predict_MNL[i,r]=fabs(profit_out_fit_MNL[i,r]-profit_out_true[i,r])/profit_out_true[i,r]
            for j in range(n+1):
                error_predict_MNL_MSE[i,r]=error_predict_MNL_MSE[i,r]+np.square(x_out_fit_MNL[i,j,r]-x_out[i,j,r])
            error_predict_MNL_MSE[i,r]=error_predict_MNL_MSE[i,r]/(n+1)
    return profit_out_fit_MNL, x_out_fit_MNL, error_predict_MNL,error_predict_MNL_MSE;

def cross_validation_NL_pulp(p_in, x_in, p_out, x_out, cost,nk): ############MaxSimIter(=50) denotes the number of cross validation experiment     
    #m_in=80   
    #m_out=1
    #MaxSimIter=50 #############change the outside index
    m_in, n,  MaxSimIter = p_in.shape
    m_out, n,  MaxSimIter = p_out.shape    
    #########declare variable
    profit_out_true=np.zeros((m_out,MaxSimIter))    
    x_out_fit_NL=np.zeros((m_out,n+1,MaxSimIter))    
    profit_out_fit_NL=np.zeros((m_out,MaxSimIter))        
    error_predict_NL=np.zeros((m_out,MaxSimIter)) ###############record the prediction error
    error_predict_NL_MSE=np.zeros((m_out,MaxSimIter)) ###############record the MSE of share
    for r in range(MaxSimIter):        
        #################use in sample to do estimation
        #nk=[7,7,6]
        V_NL,InvTau_NL,error_NL=calibration_NL_pulp(p_in[:,:,r],x_in[:,:,r],nk)
        #V_est,error = calibration_MNL_pulp(p_in[:,:,r],x_in[:,:,r])        
        for i in range(m_out):            
            x_out_fit_NL[i,:,r] = prediction_NL_pulp(V_NL,InvTau_NL,nk,p_out[i,:,r])
            profit_out_fit_NL[i,r]=sum((p_out[i,j,r]-cost[j])*x_out_fit_NL[i,j,r] for j in range(n));
            ############calculate prediction error
            #####calculate true profi
            profit_out_true[i,r]= sum((p_out[i,j,r]-cost[j])*x_out[i,j,r] for j in range(n))
            error_predict_NL[i,r]=fabs(profit_out_fit_NL[i,r]-profit_out_true[i,r])/profit_out_true[i,r]
            for j in range(n+1):
                error_predict_NL_MSE[i,r]=error_predict_NL_MSE[i,r]+np.square(x_out_fit_NL[i,j,r]-x_out[i,j,r])
            error_predict_NL_MSE[i,r]=error_predict_NL_MSE[i,r]/(n+1)
    return profit_out_fit_NL, x_out_fit_NL, error_predict_NL,error_predict_NL_MSE;

def outsample_validation_elasticity(ElasticityM ,p, x, p_out, x_out, cost,nk): ############MaxSimIter(=50) denotes the number of cross validation experiment     
    #m_in=80   
    #m_out=1
    #MaxSimIter=50 #############change the outside index
    m_out, n,  MaxSimIter = p_out.shape    
    #########declare variable
    profit_out_true=np.zeros((m_out,MaxSimIter))    
    x_out_fit_Elasticity=np.zeros((m_out,n+1,MaxSimIter))    
    profit_out_fit_Elasticity=np.zeros((m_out,MaxSimIter))        
    error_predict_Elasticity=np.zeros((m_out,MaxSimIter)) ###############record the prediction error
    error_predict_Elasticity_MSE=np.zeros((m_out,MaxSimIter)) ###############record the MSE of share
    baseshare=x[0,:]
    baseprice=p[0,:]
    for r in range(MaxSimIter):        
        #################use in sample to do estimation
        #nk=[7,7,6]
        #V_NL,InvTau_NL,error_NL=calibration_NL_pulp(p_in[:,:,r],x_in[:,:,r],nk)
        #V_est,error = calibration_MNL_pulp(p_in[:,:,r],x_in[:,:,r])    
        for i in range(m_out):            
            x_out_fit_Elasticity[i,:,r]=share_evaluation_Elasticity(p_out[i,:,r],ElasticityM,p[0,:],baseshare)
            profit_out_fit_Elasticity[i,r]=sum((p_out[i,j,r]-cost[j])*x_out_fit_Elasticity[i,j,r] for j in range(n));
            ############calculate prediction error
            #####calculate true profi
            profit_out_true[i,r]= sum((p_out[i,j,r]-cost[j])*x_out[i,j,r] for j in range(n))
            error_predict_Elasticity[i,r]=fabs(profit_out_fit_Elasticity[i,r]-profit_out_true[i,r])/profit_out_true[i,r]
            for j in range(n+1):
                error_predict_Elasticity_MSE[i,r]=error_predict_Elasticity_MSE[i,r]+np.square(x_out_fit_Elasticity[i,j,r]-x_out[i,j,r])
            error_predict_Elasticity_MSE[i,r]=error_predict_Elasticity_MSE[i,r]/(n+1)            
    return profit_out_fit_Elasticity, x_out_fit_Elasticity, error_predict_Elasticity,error_predict_Elasticity_MSE;


def generate_outsample_onepoint (x, max_value, min_value): ########generate one out-sample point in the interial of data
    ########demo: random.randint(1,101)#### generate a random integer between 1 and 100## first import random
    MaxIter, nc = x.shape
    n=nc-1
    #######generate one out sample id, whose evry entry is in the interior
    outID=random.randint(0,MaxIter-1) ##### randomly generate out sample
    tag=0 ######to indicate whether the generated out sample is suitable, which means it is in the interior, 0 means it is suitable
    for j in range(n+1):
        if (x[outID,j]==min_value[j]) | (x[outID,j]==max_value[j]):
            tag=1
        while tag==1:            
            outID=random.randint(0,MaxIter-1) ##### randomly generate out sample
            tag=0 ######to indicate whether the generated out sample is suitable, which means it is in the interior, 0 means it is suitable
            for j in range(n+1):
                if (x[outID,j]==min_value[j]) | (x[outID,j]==max_value[j]):
                    tag=1
    return outID; 

def get_bound_sharedata (x):
    ########demo: random.randint(1,101)#### generate a random integer between 1 and 100## first import random
    MaxIter, nc = x.shape
    n=nc-1
    ###########get the minimal share and maximal share for each product
    ################get the maximum value
    index_max= np.zeros((n+1),"int")
    max_value= np.zeros((n+1))
    for i in range(n+1):
        maxv=0
        maxi=0
        for j in range(MaxIter):
            if x[j,i]>maxv:
                maxv=x[j,i]
                maxi=j
        index_max[i]=maxi
        max_value[i]=maxv
    ################get the minimum value
    index_min= np.zeros((n+1),"int")
    min_value= np.zeros((n+1))
    for i in range(n+1):
        minv=1
        mini=0
        for j in range(MaxIter):
            if x[j,i]<minv:
                minv=x[j,i]
                mini=j
        index_min[i]=mini
        min_value[i]=minv
    return max_value, min_value;

def cross_validation_SeperateSample(m_in,m_out,MaxSimIter,p,x): ############MaxSimIter(=50) denotes the number of cross validation experiment     
    #m_in=80   
    #m_out=1
    #MaxSimIter=50 #############change the outside index
    MaxIter, n = p.shape
    ###########get the minimal share and maximal share for each product
    max_value, min_value = get_bound_sharedata (x)    
    ############choose out-sample
    p_in=np.zeros((m_in,n,MaxSimIter))
    x_in=np.zeros((m_in,n+1,MaxSimIter))
    p_out=np.zeros((m_out,n,MaxSimIter))
    x_out=np.zeros((m_out,n+1,MaxSimIter))
    record_out_index=np.zeros((m_out,MaxSimIter),"int") #####record the out sample index    
    for r in range(MaxSimIter):
        ########demo: random.randint(1,101)#### generate a random integer between 1 and 100## first import random
        #######generate one out sample id, whose evry entry is in the interior
        #outID = generate_outsample_onepoint (x, max_value, min_value)
        #record_out_index[0,r]=outID
        for t in range(m_out):
            outID = generate_outsample_onepoint (x, max_value, min_value)           
            while (outID in record_out_index[:,r]):
                outID = generate_outsample_onepoint (x, max_value, min_value)
            record_out_index[t,r]=outID
        for t in range(m_out):
            p_out[:,:,r]=p[record_out_index[:,r],:]
            x_out[:,:,r]=x[record_out_index[:,r],:]
        tag_in=0;
        for i in range(MaxIter):
            if (i not in record_out_index[:,r]):
                p_in[tag_in,:,r]=p[i,:]
                x_in[tag_in,:,r]=x[i,:]
                tag_in = tag_in + 1 
    return p_in, x_in, p_out, x_out;

def fun_plot_OutSample_Share(x_out,x_out_fit_exp,x_out_fit_MNL,x_out_fit_NL,x_out_fit_Elasticity):
    fig = plt.figure(figsize=(14,12))
    #for j in range(n):
    ax1 = fig.add_subplot(451)
    j=0
    ax1.scatter(x_out[0,j,:], x_out_fit_exp[0,j,:],c='r',marker='o',label="MDM")
    ax1.scatter(x_out[0,j,:], x_out_fit_MNL[0,j,:],c='b',marker='^',label="MNL")
    ax1.scatter(x_out[0,j,:], x_out_fit_NL[0,j,:],c='g',marker='x',label="NL")
    ax1.scatter(x_out[0,j,:], x_out_fit_Elasticity[0,j,:],c='k',marker='s',label="Elasticity")
    lb=np.min([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ub=np.max([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ax1.plot([lb,ub], [lb,ub],ls="--")
    ax1.set_xlim([lb,ub])
    ax1.set_ylim([lb,ub])
    plt.legend(loc='upper left')

    ax1 = fig.add_subplot(452)
    j=1
    ax1.scatter(x_out[0,j,:], x_out_fit_exp[0,j,:],c='r',marker='o',label="MDM")
    ax1.scatter(x_out[0,j,:], x_out_fit_MNL[0,j,:],c='b',marker='^',label="MNL")
    ax1.scatter(x_out[0,j,:], x_out_fit_NL[0,j,:],c='g',marker='x',label="NL")
    ax1.scatter(x_out[0,j,:], x_out_fit_Elasticity[0,j,:],c='k',marker='s',label="Elasticity")
    lb=np.min([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ub=np.max([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ax1.plot([lb,ub], [lb,ub],ls="--")
    ax1.set_xlim([lb,ub])
    ax1.set_ylim([lb,ub])

    ax1 = fig.add_subplot(453)
    j=2
    ax1.scatter(x_out[0,j,:], x_out_fit_exp[0,j,:],c='r',marker='o',label="MDM")
    ax1.scatter(x_out[0,j,:], x_out_fit_MNL[0,j,:],c='b',marker='^',label="MNL")
    ax1.scatter(x_out[0,j,:], x_out_fit_NL[0,j,:],c='g',marker='x',label="NL")
    ax1.scatter(x_out[0,j,:], x_out_fit_Elasticity[0,j,:],c='k',marker='s',label="Elasticity")
    lb=np.min([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ub=np.max([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ax1.plot([lb,ub], [lb,ub],ls="--")
    ax1.set_xlim([lb,ub])
    ax1.set_ylim([lb,ub])

    ax1 = fig.add_subplot(454)
    j=3
    ax1.scatter(x_out[0,j,:], x_out_fit_exp[0,j,:],c='r',marker='o',label="MDM")
    ax1.scatter(x_out[0,j,:], x_out_fit_MNL[0,j,:],c='b',marker='^',label="MNL")
    ax1.scatter(x_out[0,j,:], x_out_fit_NL[0,j,:],c='g',marker='x',label="NL")
    ax1.scatter(x_out[0,j,:], x_out_fit_Elasticity[0,j,:],c='k',marker='s',label="Elasticity")
    lb=np.min([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ub=np.max([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ax1.plot([lb,ub], [lb,ub],ls="--")
    ax1.set_xlim([lb,ub])
    ax1.set_ylim([lb,ub])

    ax1 = fig.add_subplot(455)
    j=4
    ax1.scatter(x_out[0,j,:], x_out_fit_exp[0,j,:],c='r',marker='o',label="MDM")
    ax1.scatter(x_out[0,j,:], x_out_fit_MNL[0,j,:],c='b',marker='^',label="MNL")
    ax1.scatter(x_out[0,j,:], x_out_fit_NL[0,j,:],c='g',marker='x',label="NL")
    ax1.scatter(x_out[0,j,:], x_out_fit_Elasticity[0,j,:],c='k',marker='s',label="Elasticity")
    lb=np.min([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ub=np.max([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ax1.plot([lb,ub], [lb,ub],ls="--")
    ax1.set_xlim([lb,ub])
    ax1.set_ylim([lb,ub])


    ax1 = fig.add_subplot(456)
    j=5
    ax1.scatter(x_out[0,j,:], x_out_fit_exp[0,j,:],c='r',marker='o',label="MDM")
    ax1.scatter(x_out[0,j,:], x_out_fit_MNL[0,j,:],c='b',marker='^',label="MNL")
    ax1.scatter(x_out[0,j,:], x_out_fit_NL[0,j,:],c='g',marker='x',label="NL")
    ax1.scatter(x_out[0,j,:], x_out_fit_Elasticity[0,j,:],c='k',marker='s',label="Elasticity")
    lb=np.min([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ub=np.max([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ax1.plot([lb,ub], [lb,ub],ls="--")
    ax1.set_xlim([lb,ub])
    ax1.set_ylim([lb,ub])


    ax1 = fig.add_subplot(457)
    j=6
    ax1.scatter(x_out[0,j,:], x_out_fit_exp[0,j,:],c='r',marker='o',label="MDM")
    ax1.scatter(x_out[0,j,:], x_out_fit_MNL[0,j,:],c='b',marker='^',label="MNL")
    ax1.scatter(x_out[0,j,:], x_out_fit_NL[0,j,:],c='g',marker='x',label="NL")
    ax1.scatter(x_out[0,j,:], x_out_fit_Elasticity[0,j,:],c='k',marker='s',label="Elasticity")
    lb=np.min([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ub=np.max([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ax1.plot([lb,ub], [lb,ub],ls="--")
    ax1.set_xlim([lb,ub])
    ax1.set_ylim([lb,ub])


    ax1 = fig.add_subplot(458)
    j=7
    ax1.scatter(x_out[0,j,:], x_out_fit_exp[0,j,:],c='r',marker='o',label="MDM")
    ax1.scatter(x_out[0,j,:], x_out_fit_MNL[0,j,:],c='b',marker='^',label="MNL")
    ax1.scatter(x_out[0,j,:], x_out_fit_NL[0,j,:],c='g',marker='x',label="NL")
    ax1.scatter(x_out[0,j,:], x_out_fit_Elasticity[0,j,:],c='k',marker='s',label="Elasticity")
    lb=np.min([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ub=np.max([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ax1.plot([lb,ub], [lb,ub],ls="--")
    ax1.set_xlim([lb,ub])
    ax1.set_ylim([lb,ub])


    ax1 = fig.add_subplot(459)
    j=8
    ax1.scatter(x_out[0,j,:], x_out_fit_exp[0,j,:],c='r',marker='o',label="MDM")
    ax1.scatter(x_out[0,j,:], x_out_fit_MNL[0,j,:],c='b',marker='^',label="MNL")
    ax1.scatter(x_out[0,j,:], x_out_fit_NL[0,j,:],c='g',marker='x',label="NL")
    ax1.scatter(x_out[0,j,:], x_out_fit_Elasticity[0,j,:],c='k',marker='s',label="Elasticity")
    lb=np.min([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ub=np.max([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ax1.plot([lb,ub], [lb,ub],ls="--")
    ax1.set_xlim([lb,ub])
    ax1.set_ylim([lb,ub])


    ax1 = fig.add_subplot(4,5,10)
    j=9
    ax1.scatter(x_out[0,j,:], x_out_fit_exp[0,j,:],c='r',marker='o',label="MDM")
    ax1.scatter(x_out[0,j,:], x_out_fit_MNL[0,j,:],c='b',marker='^',label="MNL")
    ax1.scatter(x_out[0,j,:], x_out_fit_NL[0,j,:],c='g',marker='x',label="NL")
    ax1.scatter(x_out[0,j,:], x_out_fit_Elasticity[0,j,:],c='k',marker='s',label="Elasticity")
    lb=np.min([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ub=np.max([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ax1.plot([lb,ub], [lb,ub],ls="--")
    ax1.set_xlim([lb,ub])
    ax1.set_ylim([lb,ub])

    ax1 = fig.add_subplot(4,5,11)
    j=10
    ax1.scatter(x_out[0,j,:], x_out_fit_exp[0,j,:],c='r',marker='o',label="MDM")
    ax1.scatter(x_out[0,j,:], x_out_fit_MNL[0,j,:],c='b',marker='^',label="MNL")
    ax1.scatter(x_out[0,j,:], x_out_fit_NL[0,j,:],c='g',marker='x',label="NL")
    ax1.scatter(x_out[0,j,:], x_out_fit_Elasticity[0,j,:],c='k',marker='s',label="Elasticity")
    lb=np.min([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ub=np.max([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ax1.plot([lb,ub], [lb,ub],ls="--")
    ax1.set_xlim([lb,ub])
    ax1.set_ylim([lb,ub])

    ax1 = fig.add_subplot(4,5,12)
    j=11
    ax1.scatter(x_out[0,j,:], x_out_fit_exp[0,j,:],c='r',marker='o',label="MDM")
    ax1.scatter(x_out[0,j,:], x_out_fit_MNL[0,j,:],c='b',marker='^',label="MNL")
    ax1.scatter(x_out[0,j,:], x_out_fit_NL[0,j,:],c='g',marker='x',label="NL")
    ax1.scatter(x_out[0,j,:], x_out_fit_Elasticity[0,j,:],c='k',marker='s',label="Elasticity")
    lb=np.min([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ub=np.max([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ax1.plot([lb,ub], [lb,ub],ls="--")
    ax1.set_xlim([lb,ub])
    ax1.set_ylim([lb,ub])

    ax1 = fig.add_subplot(4,5,13)
    j=12
    ax1.scatter(x_out[0,j,:], x_out_fit_exp[0,j,:],c='r',marker='o',label="MDM")
    ax1.scatter(x_out[0,j,:], x_out_fit_MNL[0,j,:],c='b',marker='^',label="MNL")
    ax1.scatter(x_out[0,j,:], x_out_fit_NL[0,j,:],c='g',marker='x',label="NL")
    ax1.scatter(x_out[0,j,:], x_out_fit_Elasticity[0,j,:],c='k',marker='s',label="Elasticity")
    lb=np.min([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ub=np.max([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ax1.plot([lb,ub], [lb,ub],ls="--")
    ax1.set_xlim([lb,ub])
    ax1.set_ylim([lb,ub])

    ax1 = fig.add_subplot(4,5,14)
    j=13
    ax1.scatter(x_out[0,j,:], x_out_fit_exp[0,j,:],c='r',marker='o',label="MDM")
    ax1.scatter(x_out[0,j,:], x_out_fit_MNL[0,j,:],c='b',marker='^',label="MNL")
    ax1.scatter(x_out[0,j,:], x_out_fit_NL[0,j,:],c='g',marker='x',label="NL")
    ax1.scatter(x_out[0,j,:], x_out_fit_Elasticity[0,j,:],c='k',marker='s',label="Elasticity")
    lb=np.min([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ub=np.max([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ax1.plot([lb,ub], [lb,ub],ls="--")
    ax1.set_xlim([lb,ub])
    ax1.set_ylim([lb,ub])

    ax1 = fig.add_subplot(4,5,15)
    j=14
    ax1.scatter(x_out[0,j,:], x_out_fit_exp[0,j,:],c='r',marker='o',label="MDM")
    ax1.scatter(x_out[0,j,:], x_out_fit_MNL[0,j,:],c='b',marker='^',label="MNL")
    ax1.scatter(x_out[0,j,:], x_out_fit_NL[0,j,:],c='g',marker='x',label="NL")
    ax1.scatter(x_out[0,j,:], x_out_fit_Elasticity[0,j,:],c='k',marker='s',label="Elasticity")
    lb=np.min([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ub=np.max([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ax1.plot([lb,ub], [lb,ub],ls="--")
    ax1.set_xlim([lb,ub])
    ax1.set_ylim([lb,ub])

    ax1 = fig.add_subplot(4,5,16)
    j=15
    ax1.scatter(x_out[0,j,:], x_out_fit_exp[0,j,:],c='r',marker='o',label="MDM")
    ax1.scatter(x_out[0,j,:], x_out_fit_MNL[0,j,:],c='b',marker='^',label="MNL")
    ax1.scatter(x_out[0,j,:], x_out_fit_NL[0,j,:],c='g',marker='x',label="NL")
    ax1.scatter(x_out[0,j,:], x_out_fit_Elasticity[0,j,:],c='k',marker='s',label="Elasticity")
    lb=np.min([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ub=np.max([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ax1.plot([lb,ub], [lb,ub],ls="--")
    ax1.set_xlim([lb,ub])
    ax1.set_ylim([lb,ub])

    ax1 = fig.add_subplot(4,5,17)
    j=16
    ax1.scatter(x_out[0,j,:], x_out_fit_exp[0,j,:],c='r',marker='o',label="MDM")
    ax1.scatter(x_out[0,j,:], x_out_fit_MNL[0,j,:],c='b',marker='^',label="MNL")
    ax1.scatter(x_out[0,j,:], x_out_fit_NL[0,j,:],c='g',marker='x',label="NL")
    ax1.scatter(x_out[0,j,:], x_out_fit_Elasticity[0,j,:],c='k',marker='s',label="Elasticity")
    lb=np.min([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ub=np.max([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ax1.plot([lb,ub], [lb,ub],ls="--")
    ax1.set_xlim([lb,ub])
    ax1.set_ylim([lb,ub])

    ax1 = fig.add_subplot(4,5,18)
    j=17
    ax1.scatter(x_out[0,j,:], x_out_fit_exp[0,j,:],c='r',marker='o',label="MDM")
    ax1.scatter(x_out[0,j,:], x_out_fit_MNL[0,j,:],c='b',marker='^',label="MNL")
    ax1.scatter(x_out[0,j,:], x_out_fit_NL[0,j,:],c='g',marker='x',label="NL")
    ax1.scatter(x_out[0,j,:], x_out_fit_Elasticity[0,j,:],c='k',marker='s',label="Elasticity")
    lb=np.min([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ub=np.max([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ax1.plot([lb,ub], [lb,ub],ls="--")
    ax1.set_xlim([lb,ub])
    ax1.set_ylim([lb,ub])

    ax1 = fig.add_subplot(4,5,19)
    j=18
    ax1.scatter(x_out[0,j,:], x_out_fit_exp[0,j,:],c='r',marker='o',label="MDM")
    ax1.scatter(x_out[0,j,:], x_out_fit_MNL[0,j,:],c='b',marker='^',label="MNL")
    ax1.scatter(x_out[0,j,:], x_out_fit_NL[0,j,:],c='g',marker='x',label="NL")
    ax1.scatter(x_out[0,j,:], x_out_fit_Elasticity[0,j,:],c='k',marker='s',label="Elasticity")
    lb=np.min([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ub=np.max([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ax1.plot([lb,ub], [lb,ub],ls="--")
    ax1.set_xlim([lb,ub])
    ax1.set_ylim([lb,ub])

    ax1 = fig.add_subplot(4,5,20)
    j=19
    ax1.scatter(x_out[0,j,:], x_out_fit_exp[0,j,:],c='r',marker='o',label="MDM")
    ax1.scatter(x_out[0,j,:], x_out_fit_MNL[0,j,:],c='b',marker='^',label="MNL")
    ax1.scatter(x_out[0,j,:], x_out_fit_NL[0,j,:],c='g',marker='x',label="NL")
    ax1.scatter(x_out[0,j,:], x_out_fit_Elasticity[0,j,:],c='k',marker='s',label="Elasticity")
    lb=np.min([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ub=np.max([x_out[0,j,:],x_out_fit_exp[0,j,:],x_out_fit_MNL[0,j,:],x_out_fit_NL[0,j,:]])
    ax1.plot([lb,ub], [lb,ub],ls="--")
    ax1.set_xlim([lb,ub])
    ax1.set_ylim([lb,ub])
    # plt.show()
    plt.savefig('public/img/GM_x_out.png')

    return plt





