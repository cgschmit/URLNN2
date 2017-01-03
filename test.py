import sys

import pylab as plb
import numpy as np
import mountaincar

#INITIALIZATION

#Just to import the inner value of the class MountainCar such as x, x_d etc.
mountain_car = mountaincar.MountainCar()

# For the call of Simulate Time Step
dt=0.01; # Time of each step
n_steps=100; # Number of timesteps

#Number of time the trial is repeated
n_trials=10;

#Number of Neurons for the grid (nNeurons X nNeurons)
nNeurons=20;

#Variable for Neuron's Grid Initialization
x_min=-150; #position
x_max=30; #position
xPoint_min=-15; #velocity
xPoint_max=15; #velocity
pos=np.linspace(x_min,x_max,nNeurons); #repartition of neuron through the grid (position)
vel=np.linspace(xPoint_min,xPoint_max,nNeurons); #repartition of neuron through the grid (velocity)
pos_c=np.ones((nNeurons))*pos; #Initialize centers with reparted value (position)
vel_c=np.ones((nNeurons))*vel; #Initialize centers with reparted value (velocity)

#Number of Actions possible at each step: Left, Right or Nothing
Actions=3; 

#Reward Factor (gamma)
Gamma=0.95;

#Eligibility Decay rate (lambda)
Lambda=0.5; #Can be comprised between 0 and 1

#Learning Rate (eta)
Eta=0.1 #LR<<1

#Neuron Activity Matrix
r=np.zeros((nNeurons,nNeurons));

#Width of Gaussian which is equal to the space between each center
sigma_x=abs(abs(x_max)+abs(x_min))/nNeurons; #space between each center on the position axis
sigma_xPoint=abs(abs(xPoint_max)+abs(xPoint_min))/nNeurons; #space between each center on the velocity axis

# Initializing weights and eligibility to zero
w=np.zeros((nNeurons,nNeurons,Actions));    #DIM 20x20x3

#Initilizing Q(s,a)
Q=np.zeros((nNeurons,nNeurons,Actions));    #DIM 20x20x3   

#Parameters Tau
Tau=1;

#MountainCarViewer
mv = mountaincar.MountainCarViewer(mountain_car)


Q_a=np.zeros((Actions));
SUM_Q=0;
Q_a_prime=np.zeros((Actions));
P=np.zeros((Actions));
Reward=0;

# Recalculate weight 
# AT EACH TIME STEP!
cond=0;

mountain_car.reset();
mv.create_figure(n_steps, n_steps);

for i in range(0,n_trials):
    # Initializiing e, s and a to 0
    state=np.zeros((nNeurons,nNeurons)); # Dim of STATE: Pos X Vel
    actions=np.zeros((1,Actions)); # Actions = 3 --> "left", "right", "none".
    e=np.zeros((nNeurons,nNeurons,Actions));
    # For each TIME STEP: (of one experiment)
    
    mountain_car.simulate_timesteps(n_steps, dt) #BOUGE PUIS CALCUL OU CALCUL INITIAL PUIS BOUGE ?
    
    while(cond<=0):
        
        # CALCULATE Q(s,a) 
        ################################################################################
        for X in range(0,nNeurons): # To go through position
            for XD in range(0,nNeurons): # To go through velocities
                r[X,XD]=np.exp(- pow(((pos_c[X]-mountain_car.x)/sigma_x),2) - pow(((vel_c[XD]-mountain_car.x_d)/sigma_xPoint),2));
        
        for X in range(0,nNeurons): # To go trhough position
            for XD in range(0,nNeurons): # To go through velocities
                for A in range(0,Actions): # To go through actions
                    #Calculte Q!!!
                    Q[X,XD,A]=w[X,XD,A]*r[X,XD]; 
        
        for A in range(0,Actions):
            Q_a[A]=np.sum(Q[:,:,0],(0,1));  #IMPORTANT: A=0 --> direction=0;
                                            # A=1 --> direction >0 (right)
                                            # A=2 --> direction <0 (left)
            
        SUM_Q=np.sum(Q);
        
        p_test=0;
        
        for A in range(0,Actions):
            P[A]=np.exp(Q_a[A]/Tau)/(SUM_Q/Tau);
            if(P[A]>p_test):
                p_test=P[A];
                index=A;
                if A==0:
                    direction=0;
                elif A==1:
                    direction=1;
                elif A==2:
                    direction=-1;
                    
        mountain_car.apply_force(direction);
                    
        #DO one more step to calculate Q(s',a')
        ################################################################################
        mountain_car._simulate_single_timestep(dt);
        ################################################################################
        
        
        #Plot new position of the car
        ################################################################################
        #mv.update_figure();
        mv._plot_positions();
        ################################################################################
        
        
        #CALCULATE Q(s',a')
        ################################################################################
        for X in range(0,nNeurons): # To go trhough position
            for XD in range(0,nNeurons): # To go through velocities
                r[X,XD]=np.exp(- pow(((pos_c[X]-mountain_car.x)/sigma_x),2) - pow(((vel_c[XD]-mountain_car.x_d)/sigma_xPoint),2));
        
        for X in range(0,nNeurons): # To go trhough position
            for XD in range(0,nNeurons): # To go through velocities
                for A in range(0,Actions): # To go through actions
                    #Calculte Q!!!
                    Q[X,XD,A]=w[X,XD,A]*r[X,XD]; 
        
        for A in range(0,Actions):
            Q_a_prime[A]=np.sum(Q[:,:,0],(0,1)); 
        ################################################################################
              
        Reward=mountain_car._get_reward();
        
        #Calcul Delta_t
            #Delta_t=r_(t+1) - [Q(s,a) - Gamma*Q(s',a')]
        ################################################################################
        Delta_t=Reward-((np.max(Q_a)-Gamma*np.max(Q_a_prime)))
        ################################################################################
        
        #Calcul Eligibility
        ################################################################################
        e=Gamma*Lambda*e+Reward;
        ################################################################################  
        
        #Update Weights
            #NewWeights=OldWeights+LearningRate*Delta_t*Eligibility
        ################################################################################
        w=w+Gamma*Delta_t*e;
        ################################################################################
        

        #WHILE EXIT CONDITION
        if(mountain_car.x>0):
            cond=1;
