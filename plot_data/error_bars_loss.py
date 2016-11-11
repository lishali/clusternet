import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
import networkx as nx
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import pandas as pd

n_cores = 4

@ops.RegisterGradient("grassman_with_2d")
def _test1(op, grad_e, grad_v):
    """Gradient for SelfAdjointEigV2 derived with Joan with no adjustment for subspace"""
    e = op.outputs[0]
    v = op.outputs[1]
    #dim = v.get_shape()
    with ops.control_dependencies([grad_e.op, grad_v.op]):
        if grad_v is not None:  
            E = array_ops.diag(e)
            v_proj = array_ops.slice(v, [0,0], [20,2])
            grad_grassman = grad_v - math_ops.batch_matmul(math_ops.batch_matmul(v_proj, array_ops.transpose(v_proj)), grad_v)
            grad_a = math_ops.batch_matmul(grad_grassman, math_ops.batch_matmul(E, array_ops.transpose(grad_v)))+math_ops.batch_matmul(grad_v, math_ops.batch_matmul(E, array_ops.transpose(grad_grassman)))
    return grad_a
def balanced_stochastic_blockmodel(communities=2, groupsize=3, p_in=0.5, p_out=0.1, seed=None):
    #gives dense adjacency matrix representaiton of randomly generated SBM with balanced community size

    G = nx.planted_partition_graph(l=communities, k=groupsize, p_in=p_in, p_out =p_out, seed=seed)
    A = nx.adjacency_matrix(G).todense()
    
    return A
def target_subspace(adj, groupsize, communities, diag, dim_proj):
    normalizer = tf.cast(2.0*groupsize*communities, dtype=tf.float64)
    total_degree = tf.cast(tf.reduce_sum(adj), dtype=tf.float64)
    r = tf.sqrt(total_degree/normalizer)
    BH_op = (tf.square(r)-1)*tf.diag(tf.ones(shape=[communities*groupsize], dtype=tf.float64))-r*adj+diag 
    val, vec = tf.self_adjoint_eig(BH_op) #this is already normalized so no need to normalize
    subspace = tf.slice(vec, [0,0], [communities*groupsize, dim_proj])
    return r, subspace

def proj_magnitude(space, vector):
    projection_op = tf.matmul(space, tf.transpose(space))
    projection = tf.matmul(projection_op, vector)
    return tf.sqrt(tf.reduce_sum(tf.square(projection))) #tf.reduce_sum(tf.abs(projection))#


def rnd_vec_normed(communities, groupsize, seed=None):
    rnd_vec1 = tf.Variable(tf.random_normal(shape=[communities*groupsize,1], mean=0.0,stddev=1.0,
                                                    dtype=tf.float64,
                                                    seed=seed))
    return normalize_vec(rnd_vec1)


def parallel_function(f):
    def easy_parallize(f, sequence):
        """ assumes f takes sequence as input, easy w/ Python's scope """
        from multiprocessing import Pool
        pool = Pool(processes=n_cores) # depends on available cores
        result = pool.map(f, sequence) # for i in sequence: result[i] = f(i)
        cleaned = [x for x in result if not x is None] # getting results
        cleaned = np.asarray(cleaned)
        pool.close() # not optimal! but easy
        pool.join()
        return cleaned
    from functools import partial
    return partial(easy_parallize, f)



def r_loss(communities = 2, group_size = 10, seed=None, p=0.4, q=0.05, r=1.0, projection_dim=2):
    """testing to see if the loss will decrease backproping through very simple function"""
    B = np.asarray(balanced_stochastic_blockmodel(communities, group_size, p, q, seed)).astype(np.double)
    B = tf.cast(B, tf.float64)
    Diag = tf.diag(tf.reduce_sum(B,0))
    Diag = tf.cast(Diag, tf.float64)

    #r_grid = tf.linspace(r_min, r_max, grid_size)
    r = tf.cast(r, tf.float64)
    
    BH = (tf.square(r)-1)*tf.diag(tf.ones(shape=[communities*group_size], dtype=tf.float64))-tf.mul(r, B)+Diag 
    
    with tf.Session() as sess:
        eigenval, eigenvec = tf.self_adjoint_eig(BH)
        eigenvec_proj = tf.slice(eigenvec, [0,0], [communities*group_size, projection_dim])
                
        true_assignment_a = tf.concat(0, [-1*tf.ones([group_size], dtype=tf.float64),
                                      tf.ones([group_size], dtype=tf.float64)])
        true_assignment_b = -1*true_assignment_a
        true_assignment_a = tf.expand_dims(true_assignment_a, 1)
        true_assignment_b = tf.expand_dims(true_assignment_b, 1)

            
        projected_a = tf.matmul(tf.matmul(eigenvec_proj, tf.transpose(eigenvec_proj)), true_assignment_a)#tf.transpose(true_assignment_a))
        projected_b = tf.matmul(tf.matmul(eigenvec_proj, tf.transpose(eigenvec_proj)), true_assignment_b)#tf.transpose(true_assignment_b))
            
        loss = tf.minimum(tf.reduce_sum(tf.square(tf.sub(projected_a, true_assignment_a))),
                              tf.reduce_sum(tf.square(tf.sub(projected_b, true_assignment_b))))
            

        d = sess.run(loss)
    return d
  

def data_create(r_min=-10, r_max=10, number=100, p_q_pairs=[(0.25, 0.05),
                                                                      (0.25, 0.1), (0.25, 0.15), 
                                                                      (0.4, 0.05), (0.4,0.2), (0.4, 0.3),
                                                                      (0.9, 0.05), (0.9, 0.45), (0.9, 0.7)], seed_r=None):
    for j in p_q_pairs:
        tmp = {}
        for r_value in np.linspace(r_min, r_max, num=number):
            p_value = j[0]
            q_value = j[1]
            tmp["{}".format(r_value)] = r_loss(seed=seed_r, p=p_value, q=q_value, r=r_value)
        tmp = pd.DataFrame(tmp, index=[0])
        tmp.to_csv("/Users/xiangli/Desktop/clusternet/plot_data/r_optimization/p{}q{}r_min{}seed{}num{}.csv".format(p_value,q_value, r_min, seed_r, number))

def data_create_specific(i):
	return data_create(r_min=-2, r_max=3, number=1, p_q_pairs=[(0.4, 0.3)], seed_r=i)


data_create_specific.Parallel = parallel_function(data_create_specific)
parallel_result = data_create_specific.parallel(range(20))


















    


