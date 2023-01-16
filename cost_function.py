import jax
import jax.numpy as jnp
from jax import jit
from tqdm import tqdm
from functools import partial





@partial(jax.jit, static_argnames=['interval_labels'])
def C(Psi_out,target,interval_labels):
    
    # computes the MSE cost function

    N_modes = jnp.shape(Psi_out)[0]
    
    return jnp.sum( jnp.abs( Psi_out[::interval_labels,-1,...]  \
                       - target[::interval_labels,-1,...] )**2, axis=0 )
  

    
@partial(jax.jit, static_argnames=['interval_labels'])
def C_realargs(psi,pi_psi,target,interval_labels):
    
    Psi = psi + 1j*pi_psi
    
    return C(Psi,target,interval_labels)
    
    
@partial(jax.jit, static_argnames=['interval_labels'])
def dC(Psi,target,interval_labels):
    
    # computes the complex derivative of the MSE cost function with respect to the output 
    # (i.e. the perturbation in the output)

    [psi,pi_psi] = [jnp.real(Psi),jnp.imag(Psi)]
    
    dCdpsi = jax.vmap( jax.jacfwd(C_realargs,argnums=0),   \
                      in_axes=(2,2,2,None), out_axes=2 ) \
             (psi,pi_psi,target,interval_labels)
    
    dCdpi = jax.vmap( jax.jacfwd(C_realargs,argnums=1),    \
                     in_axes=(2,2,2,None), out_axes=2 )  \
             (psi,pi_psi,target,interval_labels)

    return 0.5*( dCdpsi + 1j*dCdpi )


 
    
    
 