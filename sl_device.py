import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
from functools import partial


import eqs_motion as eqsm
import Hamiltonian as ham

       


@jit
def output_Stat_jax(PhiWVG,kappa,k_abs,chi,T):
    
    # This function computes the output of a single microring.
    # The input is a superposition of gaussian pulses, with amplitudes given by Psi_in,Theta_in.
    # The output is also a superposition of pulses. The output state at each time is provided.
    # Input/Output Dimensions: Psi_out = [modes,time,samples], Theta_out = [modes,time,samples].
    # The output is computed by the method based on iteration in time.
    # At each time step, it uses the previous time-step as the initial ansatz
    
    
    
    
    Phi = 0.0*PhiWVG[:,0,...]
    
    # Rescaling the inputs (see method to simulate lossy cavities)
    
    PhiWVG_resc = jnp.sqrt( kappa/(kappa+2*k_abs) )*PhiWVG
        
    
    # Newton's method. Iterate in time, using each time step as the ansatz for next time step.
    
    Phi = eqsm.RK4_evolve_jax(Phi,PhiWVG_resc,[kappa,k_abs,chi],T)  
        
    return PhiWVG + jnp.sqrt(kappa)*Phi



    

@jit
def net_output_jax_0(Psi_in,Theta_in,kappa,k_abs,chi,T):
    
    # This function computes the output for a network with several concatenated rings. 
    # The output is computed by the method based on iteration in time.
    # Input/Output Dimensions: Psi_in = [modes,time,...], Theta_in = [modes,time,...,rings]. 
    # (...) denote extra indices for samples, hyperparameters, etc.
    # Output Dimensions: Psi_out = [modes,...], Theta_out = [modes,time,...,rings].
    
    
    N_modes = np.shape(Theta_in)[0]

    N_rings = np.shape(Theta_in)[-1]
    

    Theta_out = 0.0*Theta_in  # vector that stores output Theta
    
    # We iterate over the rings. Psi_in is the input for the firs ring. 
    # For the next rings, the input is the output of the previous ring.
    
    PhiWVG = ham.encode(Psi_in,Theta_in[...,0]) 

        
    Phi_out = output_Stat_jax(PhiWVG,kappa,k_abs,chi,T)


            
    [Psi_out,Theta_out] = ham.decode(Phi_out)
         
    
    Theta_out = Theta_out[...,jnp.newaxis]
    
    
    for ring in range(1,N_rings):

        PhiWVG = ham.encode(Psi_out,Theta_in[...,ring])

        Phi_out = output_Stat_jax(PhiWVG,kappa,k_abs,chi,T)

        [Psi_out,Theta_out_ring] = ham.decode(Phi_out)
        
        Theta_out = jnp.concatenate([Theta_out,   \
                                 Theta_out_ring[...,jnp.newaxis]],axis=-1)
        
        
    return [Psi_out,Theta_out]



@jit
def net_output_jax(Psi_in,Theta_in,kappa,k_abs,chi,T): 
    
    z = jnp.zeros(jnp.shape(Psi_in)[2])
    
    return jit( jax.vmap( net_output_jax_0,   \
         in_axes=(2,2,0,0,0,None), out_axes=2 ) 
              )(Psi_in,Theta_in,kappa+z,k_abs+z,chi+z,T)





@jit
def output_Stat_diffrax(PhiWVG,kappa,k_abs,chi,T):
    
    # This function computes the output of a single microring.
    # The input is a superposition of gaussian pulses, with amplitudes given by Psi_in,Theta_in.
    # The output is also a superposition of pulses. The output state at each time is provided.
    # Input/Output Dimensions: Psi_out = [modes,time,samples], Theta_out = [modes,time,samples].
    # The output is computed by the method based on iteration in time.
    # At each time step, it uses the previous time-step as the initial ansatz
    
    
    
    
    Phi = 0.0*PhiWVG[:,0,...]
    
    # Rescaling the inputs (see method to simulate lossy cavities)
    
    PhiWVG_resc = jnp.sqrt( kappa/(kappa+2*k_abs) )*PhiWVG
        
    
    # Newton's method. Iterate in time, using each time step as the ansatz for next time step.
    
    Phi = eqsm.RK4_evolve_diffrax(Phi,PhiWVG_resc,[kappa,k_abs,chi],T)  
        
    return PhiWVG + jnp.sqrt(kappa)*Phi



@jit
def ring_output(carry,ring):
    
    [Psi_in,Theta_in,kappa,k_abs,chi,T] = carry
    
    Phi_in = ham.encode(Psi_in,Theta_in[...,ring])

    Phi_out = output_Stat_diffrax(Phi_in,kappa,k_abs,chi,T)

    [Psi_out,Theta_out_ring] = ham.decode(Phi_out)
    
    return [ [Psi_out,Theta_in,kappa,k_abs,chi,T], Theta_out_ring ]
    

@jit
def net_output_diffrax_0(Psi_in,Theta_in,kappa,k_abs,chi,T):
    
    # This function computes the output for a network with several concatenated rings. 
    # The output is computed by the method based on iteration in time.
    # Input/Output Dimensions: Psi_in = [modes,time,...], Theta_in = [modes,time,...,rings]. 
    # (...) denote extra indices for samples, hyperparameters, etc.
    # Output Dimensions: Psi_out = [modes,...], Theta_out = [modes,time,...,rings].
    
    
    N_modes = np.shape(Theta_in)[0]

    N_rings = np.shape(Theta_in)[-1]
    

    Theta_out = 0.0*Theta_in  # vector that stores output Theta
    
    # We iterate over the rings. Psi_in is the input for the firs ring. 
    # For the next rings, the input is the output of the previous ring.
    
    
    [ [Psi_out,Theta_in,kappa,k_abs,chi,T], Theta_out ] =  \
                           jax.lax.scan(ring_output,
                                [Psi_in,Theta_in,kappa,k_abs,chi,T],
                                jnp.arange(0,N_rings))  
    
    
    Theta_out = jnp.transpose(Theta_out,axes=(1,2,0))
           
    return [Psi_out,Theta_out]



@jit
def net_output_diffrax(Psi_in,Theta_in,kappa,k_abs,chi,T): 
    
    z = jnp.zeros(jnp.shape(Psi_in)[2])
    
    return jit( jax.vmap( net_output_diffrax_0,   \
         in_axes=(2,2,0,0,0,None), out_axes=2 ) 
              )(Psi_in,Theta_in,kappa+z,k_abs+z,chi+z,T)



