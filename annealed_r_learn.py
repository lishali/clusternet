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

mean_list = [-2]
l_rate_lst = [0.001]#,0.0001,0.00001, 0.000001]

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

            #using Laplacian
            Laplacian = Diag-B
            eigenval_baseline, eigenvec_baseline = tf.self_adjoint_eig(Laplacian)
            project_baseline = tf.slice(eigenvec_baseline, [0,0], [communities*group_size, projection_dim])

            projected_a_baseline = tf.matmul(tf.matmul(project_baseline, tf.transpose(project_baseline)), true_assignment_a)#tf.transpose(true_assignment_a))
            projected_b_baseline = tf.matmul(tf.matmul(project_baseline, tf.transpose(project_baseline)), true_assignment_b)
            loss_baseline = tf.minimum(tf.reduce_sum(tf.square(tf.sub(projected_a_baseline, true_assignment_a))),
                              tf.reduce_sum(tf.square(tf.sub(projected_b_baseline, true_assignment_b))))
            


            
            r_op, target = target_subspace(adj=B, groupsize=group_size, communities=communities, diag=Diag, dim_proj=projection_dim)
            r_diff = (r_op-r_param) #difference between r_op and r_param is how close we are to the average degree
            
            
            init = tf.initialize_all_variables()
            
            
            sess.run(init)
            V, R_param, R_diff, avg_deg, Loss_baseline= sess.run([v, r_param, r_diff, r_op, loss_baseline], feed_dict={X:data[0]})
            # V_lst = np.zeros(len(data), np.float64)
            V_lst = [0.0]*len(data)
            R_param_lst = np.zeros(len(data), np.float64)
            R_diff_list = np.zeros(len(data), np.float64)
            loss_list = [0.0]*len(data)
            true_assignment_lst = np.zeros(len(data), np.float64)
            avg_deg_lst = np.zeros(len(data), np.float64)
#            data_lst = []
            r_param_grad_lst = [0.0]*len(data)
            loss_baseline = np.zeros(len(data), np.float64)
            average_loss = np.zeros(len(data), np.float64)
                      
            V_lst[0]=V
            R_param_lst[0]=R_param
            R_diff_list[0]=R_diff
            loss_list[0]=None
            true_assignment_lst[0]=None
            avg_deg_lst[0]=avg_deg
 #           data_lst.append(None)
            r_param_grad_lst[0]=None
            loss_baseline[0]=Loss_baseline
            average_loss[0]=None
            
  #          print "initial v: {}. r_param: {}, difference between r_param and sqrt average deg {}.".format(a, r, b)
            for i in range(1, len(data)):   
                try:
                    sess.run(train, feed_dict={X:data[i]})
                    #if i%print_ratio==0:  
                    #print i
                        #try:
                    if i%2==0:

                        V, R_param, R_diff, Loss, Loss_baseline, r_param_grad, Tru_assign, avg_deg = sess.run([v, 
                            r_param, r_diff, loss, loss_baseline, tf.gradients(loss, r_param),
                            tf.transpose(projected_a), r_op], feed_dict={X:data[i]}) 

                        #for j in range(len(data)):
                         #   loss_tmp = np.zeros(len(data), np.float64)
                          #  tmp = sess.run(loss, feed_dict={X:data[j]})
                           # loss_tmp[j](tmp)

                        #np.mean(loss_tmp)

                        #average_loss = np.mean()

                        V_lst[j]=V
                        R_param_lst[j]=R_param
                        R_diff_list[j]=R_diff
                        loss_list[j]=Loss
                        loss_baseline[j]=Loss_baseline
                        true_assignment_lst[j]=Tru_assign
                        r_param_grad_lst[j]=r_param_grad
                        avg_deg_lst[j]=avg_deg
                    #data_lst.append(data[i])
                    
                        print "step {}: Loss:{}, Loss_baseline: {}, avg_deg: {}".format(j, Loss, Loss_baseline, avg_deg)
                    
                            
                except: 
                        V, R_param, R_diff, Loss, Loss_baseline, r_param_grad, Tru_assign, avg_deg = None, None, None, None, None, None, None, None
                        V_lst[i]=V
                        R_param_lst[i]=R_param
                        R_diff_list[i]=R_diff
                        loss_list[i]=Loss
                        loss_baseline[i]=Loss_baseline
                        true_assignment_lst[i]=Tru_assign
                        r_param_grad_lst[i]=r_param_grad
                        avg_deg_lst[i]=avg_deg
                    #data_lst.append(data[i])
                        print "step:{} not sucessful".format(i)
                        pass
                


    d = {"v": V_lst, "r_param": R_param_lst, "r_diff": R_diff_list, "loss": loss_list, "loss_baseline": loss_baseline, "R_grad_list": r_param_grad_lst, 
        "true_assignmnet": true_assignment_lst, "avg_deg": avg_deg_lst}#,"data":data_lst }
    d = pd.DataFrame(d)
    easy_size = len(data)
    #d.to_csv("mean{}l_rate{}data_size{}p_min{}p_max{}hard_ratio{}.csv".format(mean, l_rate, easy_size+hard_size, p_min, p_max, hard_ratio))
    return  d
                
            #/accounts/grad/janetlishali/clusternet/r_array_op    
                
easy_size=20
hard_size=1
p_min = 0.4
p_max = 0.41

hard_ratio = 0.1


data_easy = [np.asarray(balanced_stochastic_blockmodel(communities, group_size, p, 0.1*p)).astype(np.double) for p in np.linspace(p_min, p_max,easy_size)]
data_hard = [np.asarray(balanced_stochastic_blockmodel(communities, group_size, p, hard_ratio*p)).astype(np.double) for p in np.linspace(p_min, p_max,hard_size)]

data1 = [np.asarray(balanced_stochastic_blockmodel(communities, group_size, p, 0.1*p)).astype(np.double) for p in np.linspace(p_min-0.1, p_max-0.1, np.int_(np.ceil(easy_size/2)))]
data_h1 = [np.asarray(balanced_stochastic_blockmodel(communities, group_size, p, hard_ratio*p)).astype(np.double) for p in np.linspace(p_min-0.1, p_max-0.1,hard_size)]

data2 = [np.asarray(balanced_stochastic_blockmodel(communities, group_size, p, 0.1*p)).astype(np.double) for p in np.linspace(p_min-0.15, p_max-0.15,np.int_(np.ceil(easy_size/3)))]
data_h2 = [np.asarray(balanced_stochastic_blockmodel(communities, group_size, p, hard_ratio*p)).astype(np.double) for p in np.linspace(p_min-0.15, p_max-0.15,hard_size)]





data = data_easy+data_hard
np.random.shuffle(data)
a = data1+data_h1+data[1:np.int_(np.ceil(easy_size/2)):1]

np.random.shuffle(a)

b = data1[1:np.int_(np.ceil(easy_size/3)):1]+data2+data_h2+data[1:np.int_(np.ceil(easy_size/3)):1]
np.random.shuffle(b)
data = data+a+b                
                
                





for l in range(len(l_rate_lst)):
    for k in range(len(mean_list)):
            learn_average_deg_variable(communities = 2, group_size = 10, projection_dim=2, l_rate=l_rate_lst[l], mean=mean_list[k], sd=0.2)
                
                
