import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from jax.experimental.ode import odeint
from tqdm import tqdm
from jax.numpy.fft import fft,ifft
import scipy
from functools import partial
from diffrax import diffeqsolve, Dopri5, Tsit5, ODETerm, SaveAt, PIDController


import Hamiltonian as ham




@jit 
def interpolation_0(Phi,t,T):
    
    t_array = jnp.linspace(0,T,jnp.shape(Phi)[0])
    
    return jnp.interp(t,t_array,Phi)
    
    
interpolation = jax.jit( jax.vmap(interpolation_0,   \
                                  in_axes=(0,None,None),out_axes=0) )



@jit    
def dt_Phi_jax(Phi,t,args):
    
    # Output: the time derivative of Phi 
    
    
    [PhiWVG,params,T] = args
    
    kappa = params[0]
    
    
    # First we compute the gradient of H
    
    dHdPhi =  jax.grad(ham.Hamiltonian,argnums=0)(Phi,params)
    
    
    # Now we obtain the time derivatives accoridng to Hamilton's equations
    
    [dHdq,dHdp] = jnp.split(dHdPhi,2)       
    
    [dphidt,dpidt] = [dHdp,-dHdq]
    
    ans = jnp.concatenate([dphidt,dpidt],axis=0) 
    
    PhiWVG_t = interpolation(PhiWVG,t,T)
    
    
    return ans - kappa/2*Phi - jnp.sqrt(kappa)*PhiWVG_t



@jit
def RK4_evolve_jax(Phi_0,Phi_in,params,T):

    # Output: Phi at time t=1 obtained using 4th order RK method
    #         The output is a batch of several trajectories calculated in parallel
    # Input: Phi_0 = initial state of Phi at time t=0
    #        PhiWVG = waveguide input
    #        params = parameters of the Hamiltonian, dissipation and cavity-waveguide coupling
 
    t_array = jnp.linspace(0,T,jnp.shape(Phi_in)[1])
    
    ans = odeint(dt_Phi_jax,Phi_0,t_array,[Phi_in,params,T],atol=1.0e-5)
    
    return jnp.transpose(ans)
  
    

    
@jit
def RK4_evolve_diffrax(Phi_0,Phi_in,params,T):

    # Output: Phi at time t=1 obtained using 4th order RK method
    #         The output is a batch of several trajectories calculated in parallel
    # Input: Phi_0 = initial state of Phi at time t=0
    #        PhiWVG = waveguide input
    #        params = parameters of the Hamiltonian, dissipation and cavity-waveguide coupling
 
    t_array = jnp.linspace(0,T,jnp.shape(Phi_in)[1])
    
    def vector_field(t, Phi, args):
        return dt_Phi_jax(Phi,t,[Phi_in,params,T])
    
    term = ODETerm(vector_field)
    solver = Tsit5()
    saveat = SaveAt(ts=t_array)
    stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

    sol = diffeqsolve(term, solver, t0=0, t1=T, dt0=0.05, y0=Phi_0, saveat=saveat,stepsize_controller=stepsize_controller)
    
    return jnp.transpose(sol.ys)
  
    
