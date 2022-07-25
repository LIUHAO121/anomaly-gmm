import tensorflow as tf



def energy(gamma, z):
    """
    calculate energy for every sample in z
    gamma: (batch,timesteps,k)
    z: (batch,timesteps,l)
    
    i   : index of samples
    k   : index of components
    t   : index of time
    l,m : index of features
    
    return n_samples
    """
    gamma_sum = tf.reduce_sum(gamma, axis=1) # (i,k)
    mu = tf.einsum('itk,itl->ikl',gamma,z) / gamma_sum[:,:,None]  # (i,k,l) 每个sample之间的mu和sigma都是独立的
    z_centered = tf.sqrt(gamma[:,:,:,None]) * (z[:,:,None,:] - mu[:,None, :, :]) # (i,t,k,l)
    sigma = tf.einsum('itkl,itkm->iklm',z_centered,z_centered) / gamma_sum[:,:,None,None] # (i,k,l,m) 
    
    z_centered = z[:,:,None,:] - mu[:,None, :, :] # (i,t,k,l) -> (i,k,1,l) or (i,k,l,1)
    z_centered_last = z_centered[:,-1,:,:]
    z_c_left = z_centered_last[:,:,None,:]
    z_c_right = z_centered_last[:,:,:,None]
    inverse_sigma = tf.linalg.inv(sigma)  # (i,k,l,m)
    matrix_matmul = tf.squeeze(tf.matmul(tf.matmul(z_c_left,inverse_sigma),z_c_right)) # (i,k)
    
    e_i_k = tf.math.exp(-0.5 * matrix_matmul) # (i,k)
    det_i_k = tf.sqrt(tf.math.abs((tf.linalg.det(2 * 3.1415 * sigma)))) # (i,k)
    
    e = tf.reduce_sum((e_i_k / det_i_k) * gamma[:,-1,:],axis=-1)
    
    return -tf.math.log(e)
    
gamma = tf.random.normal([32,100,4])
z = tf.random.normal([32,100,25])

gamma = tf.math.softmax(gamma)


e = energy(gamma=gamma, z=z)
print(e)