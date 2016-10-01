import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
import networkx as nx
import numpy as np
import pandas as pd

@ops.RegisterGradient("gradient_no_unitary_adjustment")
def _test1(op, grad_e, grad_v):
    """Gradient for SelfAdjointEigV2 derived with Joan with no adjustment for subspace"""
    e = op.outputs[0]
    v = op.outputs[1]
    #dim = v.get_shape()
    with ops.control_dependencies([grad_e.op, grad_v.op]):
        if grad_v is not None:  
            E = array_ops.diag(e)
            #v_proj = arrary.ops.slice(v, [0,0], [])
            grad_grassman = grad_v# - math_ops.batch_matmul(math_ops.batch_matmul(v, array_ops.transpose(grad_v)), v)
            grad_a = math_ops.batch_matmul(grad_grassman, math_ops.batch_matmul(E, array_ops.transpose(grad_v)))+math_ops.batch_matmul(grad_v, math_ops.batch_matmul(E, array_ops.transpose(grad_grassman)))
    return grad_a

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


def test_svm_cluster(communities = 2, group_size = 10, seed=1, seed_r=1, p=0.8, q=0.05, name='test1', projection_dim=2, iterations=100, 
                     print_ratio=10, l_rate=0.1, mean=2.0, sd=0.4):
    """testing to see if the loss will decrease backproping through very simple function"""
    B = np.asarray(balanced_stochastic_blockmodel(communities, group_size, p, q, seed)).astype(np.double)
    B = tf.cast(B, dtype = tf.float64)
    
    Diag = tf.diag(tf.reduce_sum(B,0))
    Diag = tf.cast(Diag, tf.float64)

    r =  tf.Variable(tf.random_normal(shape=[1], mean=mean,
                                 stddev=sd, dtype=tf.float64,
                                 seed=seed_r, name=None))

    
    BH = (tf.square(r)-1)*tf.diag(tf.ones(shape=[communities*group_size], dtype=tf.float64))-tf.mul(r, B)+Diag 
    

    with tf.Session() as sess:
        g = tf.get_default_graph()
        
        with g.gradient_override_map({'SelfAdjointEigV2': name}):
            eigenval, eigenvec = tf.self_adjoint_eig(BH)
            #we try to do svm in this subspace 
            #or we can project it down to 1 dimensions, do the clustering there via some threshold and check if it makes sense 
            #by computing the loss, if it is too big, we change the angle we project down to...
            
            
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
            
            optimizer = tf.train.GradientDescentOptimizer(l_rate)
            
            train = optimizer.minimize(loss, var_list=[r])

            eigenvec_grad = tf.gradients(eigenvec, r)
            loss_grad = tf.gradients(loss, r)
            
            
            
            r_op, target = target_subspace(adj=B, groupsize=group_size, communities=communities, diag=Diag, dim_proj=projection_dim)  
            
            r_op_projection_a = tf.matmul(tf.matmul(target, tf.transpose(target)), true_assignment_a)
            r_op_projection_b = tf.matmul(tf.matmul(target, tf.transpose(target)), true_assignment_b)
            r_op_loss = tf.minimum(tf.reduce_sum(tf.square(tf.sub(r_op_projection_a, true_assignment_a))),
                              tf.reduce_sum(tf.square(tf.sub(r_op_projection_b, true_assignment_b))))
            
            init = tf.initialize_all_variables()
            
            
            sess.run(init)
            a,b,c,d= sess.run([r, r_op, r_op_loss, tf.transpose(r_op_projection_a)])
            a_lst = []
            b_lst = []
            c_lst = []
            d_lst = []
            
            a_lst.append(a)
            b_lst.append(b)
            c_lst.append(c)
            d_lst.append(d)
            
            print "initial r: {}. r_op = sqrt(average degree) : {} . Loss associated with r_op: {}. r_op assignments {}.".format(a, b, c, d)
            for i in range(iterations):   
                try: sess.run(train)
                except: 
                    pass
                
                if i%print_ratio==0:  
                    #print i
                    try:
                        a,b,c,d = sess.run([r, loss, tf.gradients(loss, r), tf.transpose(projected_a)]) 
                        a_lst.append(a)
                        b_lst.append(b)
                        c_lst.append(c)
                        d_lst.append(d)
                    except:
                        a,b,c,d = 0, 0, 0, 0 
                        a_lst.append(a)
                        b_lst.append(b)
                        c_lst.append(c)
                        d_lst.append(d)
                    #print "current r: {}, current loss: {}, gradient of loss/r is {} and current assignments (up to sign) {}.".format(a,b,c,d)  

    d = {"r_value": a_lst, "loss": b_lst, "gradient_loss_r": c_lst, "projection": d_lst}
    d = pd.DataFrame(d)
    d.to_csv("/Users/xiangli/Desktop/clusternet/plot_data/r{}rate{}p{}q{}iterations{}step{}.csv".format(mean, l_rate, p, q, iterations, print_ratio))
    return  d
                

                

r_list = [i for i in range(-6, 7, 2)]
p_q_pairs = [(0.4, 0.05)]
l_rate_lst = [10**(-i)/3 for i in range(3, 6, 1)]



for l in range(len(l_rate_lst)):
    for k in range(len(r_list)):
        for j in range(len(p_q_pairs)):
            test_svm_cluster(communities = 2, group_size = 10, seed=100, seed_r=2000, p=p_q_pairs[j][0],
                               	q=p_q_pairs[j][1], name='grassman_with_2d', projection_dim=2, iterations=5000, 
                     	print_ratio=100, l_rate=l_rate_lst[l], mean=r_list[k], sd=0.2)

