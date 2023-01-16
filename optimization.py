import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from tqdm import tqdm
from functools import partial


import dataset as ds
import cost_function as cf
import sl_device as sld




@jit
def TR(Psi,Theta,T):
    
    # important: because of our definition of the coupling of the waveguide with the cavity, the waveguide modes must acquire a minus sign during time-reversal
    
    N_tpoints = jnp.shape(Theta)[1]

    t_array = jnp.linspace(0,T,N_tpoints)
    
    Psi_TR = - jnp.conj( ds.pulse_shape(Psi[:,-1,...],t_array) )
    
    Theta_TR = - jnp.flip( jnp.conj(ds.pulse_shape(Theta[:,-1,...],t_array)),\
                          axis = -1 )
    
    return [Psi_TR, Theta_TR]



@partial(jax.jit, static_argnames=['interval_labels'])
def field_to_label(Psi,interval_labels):
    
    N_modes = jnp.shape(Psi)[0]
    
    return jnp.argmax(jnp.real(Psi[::interval_labels,-1,:]),axis=0)
    
    

@partial(jax.jit, static_argnames=['interval_labels'])
def output_label(Psi_in,Theta_in,kappa,k_abs,chi,interval_labels,T):
    
    [Psi_out,Theta_out] = sld.net_output_jax(Psi_in,Theta_in,   \
                                         kappa,k_abs,chi,T)
    
    return field_to_label(Psi_out,interval_labels)
    
    

@partial(jax.jit, static_argnames=['interval_labels','N_samples'])    
def test_acc(target,Psi_in_all,Theta_in,kappa,k_abs,chi,   \
             interval_labels,T,N_samples=300):
    
    Psi_in = Psi_in_all[...,:N_samples]
    
    target_label = field_to_label(target[...,:N_samples],interval_labels)
    
    Theta_in_tiled = jnp.einsum('i,jkl->jkil',   \
                                jnp.ones(np.shape(Psi_in)[-1]),Theta_in)
    
    pred_label = output_label(Psi_in,Theta_in_tiled,   \
                      kappa,k_abs,chi,interval_labels,T)
    
    truth = jnp.less( jnp.abs(target_label-pred_label), 0.1 ).astype(int)
    
    return [jnp.sum(truth)/jnp.shape(target_label)[0],  \
            pred_label,target_label]


 


@partial(jax.jit, static_argnames=['interval_labels'])
def echo_step_jax(Psi_in,Theta_in,kappa,k_abs,chi,
                  target,eta,interval_labels,T):
    
    # implements the Hamiltonian part of HEB: forward pass + backward pass
 
   
    
    # forward step 

    [Psi_out,Theta_out] =   \
    sld.net_output_jax(Psi_in,Theta_in,kappa,k_abs,chi,T)
    
    cost = cf.C(Psi_out,target,interval_labels)
    
    acc = jnp.less( jnp.abs(field_to_label(Psi_out,interval_labels)  \
                    - field_to_label(target,interval_labels) ), \
                   0.1 ).astype(int)
    
    # perturbation of the output and phase conjugation operation
    
    [Psi_bw_in,Theta_bw_in] = TR( Psi_out -    \
          1j*eta*cf.dC(Psi_out,target,interval_labels) , Theta_out, T ) 

    # backward step 
    
    [Psi_bw_out,Theta_bw_out] =     \
    sld.net_output_jax(Psi_bw_in,Theta_bw_in,kappa,k_abs,chi,T)
     
    [Psi_f,Theta_f] = TR(Psi_bw_out,Theta_bw_out,T)
    
    return [Psi_f,Theta_f,cost,acc]



def decay_step(Theta_in,eta,T):
    
    # implements the decay step
    
    
    return jnp.real(Theta_in) + eta*jnp.imag(Theta_in)   + 1j*0
    
  





  



@partial(jax.jit, static_argnames=['interval_labels'])
def HEB_step(Psi_in,Theta_in,kappa,k_abs,chi,
             target,eps,eta,interval_labels,T):
    
    # implements a full HEB step: echo step + decay step
    
    
    
    [Psi_out,Theta_out,cost,acc] =    \
          echo_step_jax(Psi_in,Theta_in,kappa,k_abs,chi,   
               target,eps,interval_labels,T)
    
    Theta_out = decay_step(Theta_out,eta/2,T)
    
    [Psi_out,Theta_out,cost,acc_1] =    \
          echo_step_jax(Psi_in,Theta_out,kappa,k_abs,chi,  
               target,-eps,interval_labels,T)
    
    Theta_out = decay_step(Theta_out,-eta/2,T)
    

    
    return [Theta_out,cost,acc]












def optimization(target,Psi_in,Theta_in,kappa,k_abs,chi,  
    N_train_steps,eps,eta,interval_labels,T,params_corr=None):

    
    [N_modes,N_tsteps,N_samples] = jnp.shape(Psi_in)[:3]
    
    
    for step in tqdm(range(N_train_steps)):
        
        sample_index = np.random.randint(N_samples)
        
        Psi_in_step = (Psi_in[:,:,sample_index,...])
        
        target_step = (target[:,:,sample_index,...])
        
        [Theta_in,cost_n,acc_n] =    \
         HEB_step(Psi_in_step,Theta_in,kappa,k_abs,chi,   \
         target_step,eps,eta,interval_labels,T)
        
         
        if(step==0):
            
            cost = jnp.copy(cost_n)[jnp.newaxis,:]
            
            acc = jnp.copy(acc_n)[jnp.newaxis,:]
            
        else:
            
            cost = jnp.concatenate( [cost,cost_n[jnp.newaxis,:]],  \
                                   axis=0 )
        
            acc = jnp.concatenate( [acc,acc_n[jnp.newaxis,:]],   \
                                  axis=0 )
        
        
    return [Theta_in,cost,acc]




