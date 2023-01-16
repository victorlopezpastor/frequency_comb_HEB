import jax
import jax.numpy as jnp
from jax.numpy.fft import fft,ifft
from jax import jit
import numpy as np
from scipy import linalg
from tqdm import tqdm
from time import time





@jit
def encode(Psi,Theta):
    
    # Output: the fields in a single real vector Phi 
    # Phi is formed by concatenating (psi, momentum of psi, theta, momentum of theta)
    
    return jnp.concatenate([jnp.real(Psi),jnp.real(Theta),jnp.imag(Psi),jnp.imag(Theta)],axis=0)



@jit
def decode(Phi):
    
    # Output: the complex fields Psi,Theta

    [psi,theta_b,theta_w1,theta_w2,   \
     pi_psi,pi_theta_b,pi_theta_w1,pi_theta_w2] = jnp.split(Phi,8)
    
    Psi = psi+1j*pi_psi
    
    Theta_b = theta_b+1j*pi_theta_b
    
    Theta_w1 = theta_w1 + 1j*pi_theta_w1
    
    Theta_w2 = theta_w2 + 1j*pi_theta_w2

    Theta = jnp.concatenate([Theta_b,Theta_w1,Theta_w2],axis=0)
    
    return [Psi,Theta]
    
    

@jit
def toep_mat_mul_r(theta,psi):
    
    # Output: vector with components f_k = sum_{j=0}^{N_k-1} theta_{j} psi_{j+k}
    
    # The algorithm is based on an efficient method for multiplying a Toeplitz matrix 
    # by a vector (see: https://alinush.github.io/2020/03/19/multiplying-a-vector-by-a-toeplitz-matrix.html)
    
    
    v_1 = jnp.copy(theta)[0,jnp.newaxis,...]
    
    v_2 = 0*theta[1:,...]
    
    v_3 = jnp.copy(theta[0,jnp.newaxis,...])
    
    v_4 = jnp.flip( jnp.copy(theta[1:,...]), axis=0 )
    
    v = jnp.concatenate((v_1,v_2,v_3,v_4),axis=0)
    
    
    x = jnp.copy(psi)
    
    anc = 0*psi
    
    x_pad = jnp.concatenate((x,anc),axis=0)
    
    v_ft = fft( v, axis=0 )
    
    [ans, extra] = jnp.split( ifft( v_ft*fft(x_pad,axis=0), axis=0 ), 2) 
    
    return ans[1:,...]





@jit
def Hamiltonian(Phi,params):
    
    # Output: the Hamiltonian 
    # The input vector Phi is a real vector that combines Psi and Theta as following:
    #      [real(Psi),imag(Psi),real(Theta),imag(Theta)]
    
    
    [kappa,k_abs,chi] = params
    
    
    # We get the complex fields Psi = q + i*p
    
    [psi_bias,theta_W,pi_psi_bias,pi_theta_W] = jnp.split(Phi,4)
    
    Psi_bias = psi_bias + 1j*pi_psi_bias
    
    Theta_W = theta_W + 1j*pi_theta_W
    
    # Psi-Theta interaction
    
    g_psi = toep_mat_mul_r(jnp.conj(Psi_bias),Psi_bias)
    
    g_theta = toep_mat_mul_r(jnp.conj(Theta_W),Theta_W)
    
    H = 2*chi*jnp.real( jnp.sum( g_psi*jnp.conj(g_theta) ,axis=0) )
    
    H = H + chi*jnp.real( jnp.sum( g_psi*jnp.conj(g_psi) ,axis=0) )
    
    H = H + chi*jnp.real( jnp.sum( g_theta*jnp.conj(g_theta) ,axis=0) )
   
    return H


    

