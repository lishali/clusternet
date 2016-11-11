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

communities = 2
group_size = 10




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






def learn_average_deg_variable(communities = 2, group_size = 10, seed_v=None, projection_dim=2, l_rate=0.00000001, mean=0.3, sd=0.1):
    """testing to see if the loss will decrease backproping through very simple function"""
    
    #now p and q will be generated from a range of 
    
    X = tf.placeholder(dtype=tf.float64, shape=[communities*group_size, communities*group_size])
    
    B = tf.cast(X, dtype = tf.float64)
    
    Diag = tf.diag(tf.reduce_sum(B,0))
    Diag = tf.cast(Diag, tf.float64)
    
    #by symmetry I should make this a bit more constrained.  so

    v =  tf.Variable(tf.random_normal(shape=[communities*group_size,1], mean=mean,
                                 stddev=sd, dtype=tf.float64,
                                 seed=seed_v, name=None))
    
     
    
    degree = tf.cast(communities*group_size, dtype=tf.float64)
    r_param = tf.div(tf.cast(1.0, dtype=tf.float64), degree)*tf.matmul(tf.transpose(v), tf.matmul(Diag, v))

    
    BH = (tf.square(r_param)-1)*tf.diag(tf.ones(shape=[communities*group_size], dtype=tf.float64))-tf.mul(r_param, B)+Diag 
    

    with tf.Session() as sess:
        g = tf.get_default_graph()
        
        with g.gradient_override_map({'SelfAdjointEigV2': 'grassman_with_2d'}):
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
            
            optimizer = tf.train.AdamOptimizer(l_rate)
            
            train = optimizer.minimize(loss, var_list=[v])

            eigenvec_grad = tf.gradients(eigenvec, v)
            loss_grad = tf.gradients(loss, v)
            
            r_op, target = target_subspace(adj=B, groupsize=group_size, communities=communities, diag=Diag, dim_proj=projection_dim)
            r_diff = (r_op-r_param) #difference between r_op and r_param is how close we are to the average degree
            
            
            init = tf.initialize_all_variables()
            
            
            sess.run(init)
            a,r, b, avg_deg= sess.run([v, r_param, r_diff, r_op], feed_dict={X:data[0]})
            a_lst = []
            r_lst = []
            r_diff_list = []
            b_lst = []
            c_lst = []
            d_lst = []
            avg_deg_lst = []
            data_lst = []
            r_param_grad_lst = []
            
            
            a_lst.append(a)
            r_lst.append(r)
            r_diff_list.append(b)
            b_lst.append(None)
            c_lst.append(None)
            d_lst.append(None)
            avg_deg_lst.append(avg_deg)
            data_lst.append(None)
            r_param_grad_lst.append(None)
            
            print "initial v: {}. r_param: {}, difference between r_param and sqrt average deg {}.".format(a, r, b)
            for i in range(len(data)):   
                try:
                    sess.run(train, feed_dict={X:data[i]})
                    #if i%print_ratio==0:  
                    #print i
                        #try:
                    a,r, k, b,c,d, r_param_grad, avg_deg = sess.run([v, r_param, r_diff, loss, tf.gradients(loss, v), tf.transpose(projected_a), tf.gradients(loss, r_param), r_op], feed_dict={X:data[i]}) 
                    a_lst.append(a)
                    r_lst.append(r)
                    r_diff_list.append(k)
                    b_lst.append(b)
                    c_lst.append(c)
                    d_lst.append(d)
                    r_param_grad_lst.append(r_param_grad)
                    avg_deg_lst.append(avg_deg)
                    data_lst.append(data[i])
                    
                    print "step: {}: loss: {}, avg_deg: {} r_param:{}, r_diff: {}, r_param gradient: {}".format(i, b, avg_deg, r, k, r_param_grad)
                    
                            
                except: 
                    a,r, k, b,c,d, r_param_grad, avg_deg = None, None, None, None, None, None, None, None, None
                    a_lst.append(a)
                    r_lst.append(r)
                    r_diff_list.append(k)
                    b_lst.append(b)
                    c_lst.append(c)
                    d_lst.append(d)
                    r_param_grad_lst.append(r_param_grad)
                    avg_deg_lst.append(avg_deg)
                    data_lst.append(data[i])
                    print "step:{} not sucessful".format(i)
                    pass
                


    d = {"v": a_lst, "r_param": r_lst, "r_diff": r_diff_list, "loss": b_lst, "gradient_loss_v": c_lst, "projection": d_lst, 
        "r_param_grad": r_param_grad_lst, "avg_deg": avg_deg_lst, "data":data_lst }
    d = pd.DataFrame(d)
    easy_size = len(data)
    d.to_csv("/accounts/grad/janetlishali/clusternet/r_array_op/mean{}l_rate{}data_size{}p_min{}p_max{}hard_ratio{}.csv".format(mean, l_rate, easy_size+hard_size, p_min, p_max, hard_ratio))
    return  d
                
                
                
easy_size=5000
hard_size=1
p_min = 0.4
p_max = 0.41
hard_ratio = 0.1


data_easy = [np.asarray(balanced_stochastic_blockmodel(communities, group_size, p, 0.1*p)).astype(np.double) for p in np.linspace(p_min, p_max,easy_size)]
data_hard = [np.asarray(balanced_stochastic_blockmodel(communities, group_size, p, hard_ratio*p)).astype(np.double) for q in np.linspace(p_min, p_max,hard_size)]

data = data_easy+data_hard
np.random.shuffle(data)
                
                
                
mean_list = [-4]
l_rate_lst = [0.001]#,0.0001,0.00001, 0.000001]



for l in range(len(l_rate_lst)):
    for k in range(len(mean_list)):
            learn_average_deg_variable(communities = 2, group_size = 10, projection_dim=2, l_rate=l_rate_lst[l], mean=mean_list[k], sd=0.2)
                
                
