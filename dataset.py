import numpy as np
import jax
import jax.numpy as jnp
from scipy import linalg
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split




def pulse(t,T):
    
    sigma = 50.0
    
    w = 0.97
    
    return 0.5*( jnp.tanh(sigma*(t+w-T)) + jnp.tanh(sigma*(w-t+T)) )


def pulse_shape(amp_array,t_array):
    
    T = t_array[-1]
    
    return jnp.einsum('i,j...->ji...',pulse(t_array,T),amp_array)








def random_theta(N_modes,N_rings,N_traj,amp_theta,T,N_tpoints):
    
    # creates a random configuration for theta
    
    
    Theta_in =  np.random.randn(2*N_modes,N_rings) + 1j*0.0 
    
    

    Theta_in = np.concatenate([np.zeros((N_modes,N_rings)),Theta_in],axis=0)
        
    
    
    Theta_exp = np.zeros((3*N_modes,N_traj,N_rings),dtype=complex)
    
    for ring in range(0,N_rings):
    
        Theta_exp[...,ring] = np.einsum('i,j->ji',np.ones(N_traj),Theta_in[...,ring])
        
        
    t_array = np.linspace(0,1,N_tpoints)
    
    return amp_theta*pulse_shape(Theta_exp,t_array)





def expanded_data(Psi,target,amp_in,amp_out):
    
    
    Psi_exp = np.einsum('i,...j->...ji',amp_in,Psi)
    
    
    target_exp = np.einsum('i,...j->...ji',amp_out,target)
 
        
    return [Psi_exp,target_exp]





    


def mnist8_dataset(N_modes,amp_psi,amp_target,
               N_digits,test_size,T,N_tpoints,interval_labels,N_traj):
    
    digits = load_digits()
    
    # flatten the images
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(   \
               data, digits.target, test_size=test_size,shuffle=False)
    
    
    
    trainset_01 = np.where(y_train<N_digits)[0]
    
    testset_01 = np.where(y_test<N_digits)[0]
    
    X_train, X_test, y_train, y_test =   \
    [X_train[trainset_01,:], X_test[testset_01,:], \
                 y_train[trainset_01], y_test[testset_01]]
    

    trainset_size = np.shape(X_train)[0]
    
    dim_input = np.shape(X_train)[1]
    
    Psi_train =  np.zeros([N_modes,trainset_size],dtype=complex)
    
    Psi_train[:dim_input,:] = np.transpose(X_train)/16
    

    
    testset_size = np.shape(X_test)[0]
    
    Psi_test =  np.zeros([N_modes,testset_size],dtype=complex)
   
    Psi_test[:dim_input,:] = np.transpose(X_test)/16  
    
    
    target_train =  np.zeros([N_modes,trainset_size],dtype=complex)
        
    target_test =  np.zeros([N_modes,testset_size],dtype=complex)
    
    
   
    
    for j in range(0,N_digits):
    
        target_train[j*interval_labels,:] = (2*np.equal(y_train,j)-1)
    
        target_test[j*interval_labels,:] = (2*np.equal(y_test,j)-1)

    
    [Psi_train,target_train] = expanded_data(Psi_train,   
                                   target_train,
                                   amp_psi*np.ones(N_traj),
                                   amp_target*np.ones(N_traj))
    
    
    [Psi_test,target_test] = expanded_data(Psi_test,   \
                                   target_test,
                                   amp_psi*np.ones(N_traj),
                                   amp_target*np.ones(N_traj))
    
    
    t_array = np.linspace(0,1,N_tpoints)
    
    Psi_train = pulse_shape(Psi_train,t_array)
    
    target_train = pulse_shape(target_train,t_array)
    
    Psi_test = pulse_shape(Psi_test,t_array)
    
    target_test = pulse_shape(target_test,t_array)
    
    
    ans = [Psi_train,target_train,Psi_test,target_test]
    
    return [jax.device_put(x) for x in ans]