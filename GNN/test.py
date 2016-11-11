
import tensorflow as tf
import networkx as nx
import numpy as np
import pandas as pd


def Diffusion(A, F):
    """finds the diffusion of F via A"""
    #F can be a batched signal
    return tf.batch_matmul(A, F)

def Diag(A, F):
    """multiplies F by diagonal vector"""
    diag_matrix = tf.expand_dims(tf.reduce_sum(Adj, 1), 1)
    return tf.mul(diag_matrix, F)

def Diffusion_normed(A, F):
    """D^-1W"""
    return tf.mul(tf.div(1.0, Diag(A, F)), tf.batch_matmul(A, F))


def balanced_stochastic_blockmodel(communities=2, groupsize=3, p_in=1.0, p_out=0.0, seed=None):
    """gives dense adjacency matrix representaiton of randomly generated SBM with balanced community size"""

    G = nx.planted_partition_graph(l=communities, k=groupsize, p_in=p_in, p_out =p_out, seed=seed)
    A = nx.adjacency_matrix(G).todense()
    
    return A

def batch_vm2(x, m):
    [input_size, output_size] = m.get_shape().as_list()

    input_shape = tf.shape(x)
    batch_rank = input_shape.get_shape()[0].value - 1
    batch_shape = input_shape[:batch_rank]
    output_shape = tf.concat(0, [batch_shape, [output_size]])

    x = tf.reshape(x, [-1, input_size])
    y = tf.matmul(x, m)

    y = tf.reshape(y, output_shape)

    return y



###################
################### HYPERPARAMETERS



def GNN(signal_dim = 10, batch_size = 2, SD=1, communities=2, group_size=10, 
    p_min = 0.5, p_max = 0.5, Mean = 1, Mean_signal=1, l_rate = 0.0000001, Size=100000, seed=None):


    """ First implement of GNN"""
    dim = communities*group_size

    DATA1 = [np.asarray(balanced_stochastic_blockmodel(communities, group_size, p, 0.1*p, seed=seed)).astype(np.double) for p in np.linspace(p_min, p_max, Size)]
    DATA2 = [np.asarray(balanced_stochastic_blockmodel(communities, group_size, p, 0.1*p, seed=seed+23235)).astype(np.double) for p in np.linspace(p_min, p_max, Size)]
    DATA3 = [np.asarray(balanced_stochastic_blockmodel(communities, group_size, p, 0.1*p, seed=seed+25452)).astype(np.double) for p in np.linspace(p_min, p_max, Size)]
    DATA4 = [np.asarray(balanced_stochastic_blockmodel(communities, group_size, p, 0.1*p, seed=seed+234)).astype(np.double) for p in np.linspace(p_min, p_max, Size)]
    DATA5 = [np.asarray(balanced_stochastic_blockmodel(communities, group_size, p, 0.1*p, seed=seed+54)).astype(np.double) for p in np.linspace(p_min, p_max, Size)]
    DATA6 = [np.asarray(balanced_stochastic_blockmodel(communities, group_size, p, 0.1*p, seed=seed+25438352)).astype(np.double) for p in np.linspace(p_min, p_max, Size)]
    
    DATA = DATA1+DATA2+DATA3

    np.random.shuffle(DATA)
    Signal = SD*np.random.randn(signal_dim, dim) + Mean_signal
    TRUE_A = np.append(np.zeros([batch_size, group_size], dtype=float),np.ones([batch_size, group_size], dtype=float), axis = 1)
    TRUE_B = 1-TRUE_A

    Adj = tf.placeholder(dtype=tf.float32, shape=[None, communities*group_size, communities*group_size])
    Adj_mod = tf.reshape(tf.transpose(Adj, perm = [1,0,2]), [dim, batch_size*dim])#preparing it to be multiplied by F to broadcast

    F = tf.placeholder(dtype=tf.float32, shape = [signal_dim, dim])

    #first diffusion step without cascading (unnormed version)
    Diff_1 = tf.reshape(tf.transpose(tf.matmul(F, Adj_mod)), shape=[batch_size, dim, signal_dim]) #shape=[batch_size, signal_dim, dim])

    diag_inv = tf.div(1.0, tf.reduce_sum(Adj, 2))
    diag_inv_batch = tf.matrix_diag(diag_inv) #to use in subsequent layers
    Diag_1 = tf.mul(tf.expand_dims(diag_inv, 1), F)


    C_a = tf.Variable(tf.random_normal([signal_dim, signal_dim], stddev=1.0, mean=Mean))
    C_b = tf.Variable(tf.random_normal([signal_dim, signal_dim], stddev=1.0, mean=Mean))

    #treat this as the new Adj_mod
    A1 = tf.matmul(C_a, tf.reshape(tf.transpose(Diag_1, perm=[1, 0,2]), [signal_dim, batch_size*dim]))
    B1 = tf.matmul(C_b, tf.reshape(tf.transpose(Diff_1, perm=[2, 0,1]), [signal_dim, batch_size*dim]))

    #transform it back into the 3-D tensor it is
    #Psi_1 = tf.transpose(tf.reshape(A1 + B1, shape = [signal_dim, batch_size, dim]), perm=[1,0,2])
    #relu also added
    Psi_1 = tf.transpose(tf.reshape(tf.nn.relu(A1 + B1), shape = [signal_dim, batch_size, dim]), perm=[1,2,0])


 ###################
    ###################
    Diff_2 = tf.batch_matmul(Adj, Psi_1)
    Diag_2 = tf.batch_matmul(diag_inv_batch, Psi_1)
    #we change the constants for now but let's keep these the same for another model
    C_a_1 = tf.Variable(tf.random_normal([signal_dim, signal_dim], stddev=1.0, mean=Mean))
    C_b_1 = tf.Variable(tf.random_normal([signal_dim, signal_dim], stddev=1.0, mean=Mean))

    A2 = tf.matmul(C_a_1, tf.reshape(tf.transpose(Diag_2, perm=[2, 0,1]), [signal_dim, batch_size*dim]))
    B2 = tf.matmul(C_b_1, tf.reshape(tf.transpose(Diff_2, perm=[2, 0,1]), [signal_dim, batch_size*dim]))

    Psi_2 = tf.transpose(tf.reshape(tf.nn.relu(A2 + B2), shape = [signal_dim, batch_size, dim]), perm=[1,2,0])

    ##################
    ##################
    Diff_3 = tf.batch_matmul(Adj, Psi_2)
    Diag_3 = tf.batch_matmul(diag_inv_batch, Psi_2)

    C_a_2 = tf.Variable(tf.random_normal([signal_dim, signal_dim], stddev=1.0, mean=Mean))
    C_b_2 = tf.Variable(tf.random_normal([signal_dim, signal_dim], stddev=1.0, mean=Mean))

    A3 = tf.matmul(C_a_2, tf.reshape(tf.transpose(Diag_3, perm=[2, 0,1]), [signal_dim, batch_size*dim]))
    B3 = tf.matmul(C_b_2, tf.reshape(tf.transpose(Diff_3, perm=[2, 0,1]), [signal_dim, batch_size*dim]))

    Psi_3 = tf.transpose(tf.reshape(tf.nn.relu(A3 + B3), shape = [signal_dim, batch_size, dim]), perm=[1,2,0])

    ##################
    ##################
    Diff_4 = tf.batch_matmul(Adj, Psi_3)
    Diag_4 = tf.batch_matmul(diag_inv_batch, Psi_3)

    C_a_3 = tf.Variable(tf.random_normal([signal_dim, signal_dim], stddev=1.0, mean=Mean))
    C_b_3 = tf.Variable(tf.random_normal([signal_dim, signal_dim], stddev=1.0, mean=Mean))

    A4 = tf.matmul(C_a_3, tf.reshape(tf.transpose(Diag_4, perm=[2, 0,1]), [signal_dim, batch_size*dim]))
    B4 = tf.matmul(C_b_3, tf.reshape(tf.transpose(Diff_4, perm=[2, 0,1]), [signal_dim, batch_size*dim]))

    Psi_4 = tf.transpose(tf.reshape(tf.nn.relu(A4 + B4), shape = [signal_dim, batch_size, dim]), perm=[1,2,0])


    ###################
    ###################



    B_reduce = tf.Variable(tf.random_normal([signal_dim, 2], stddev=1.0, mean=0.0))

    Y_hat = tf.nn.relu(batch_vm2(Psi_4, B_reduce))
    test = tf.nn.softmax(Y_hat)
    ##################
    ################## TRUE ASSIGNMENTlton

    #true_assignment_a = tf.expand_dims(tf.concat(0, [tf.zeros([group_size], dtype=tf.float32),
     #                                     tf.ones([group_size], dtype=tf.float32)]), 1)
    #true_assignment_b = tf.expand_dims(tf.concat(0, [tf.ones([group_size], dtype=tf.float32),
     #                                     tf.zeros([group_size], dtype=tf.float32)]), 1)

    true_assignment_a= tf.placeholder(dtype=tf.int32, shape = [batch_size, dim])
    true_assignment_b= tf.placeholder(dtype=tf.int32, shape = [batch_size, dim])
    
    #true_a_prob = true_assignment_a 1-true_assignment_a
    a = tf.nn.sparse_softmax_cross_entropy_with_logits(Y_hat, true_assignment_a)
    
    b = tf.nn.sparse_softmax_cross_entropy_with_logits(Y_hat, true_assignment_b)

    loss_a = tf.reduce_sum(a)
    loss_b = tf.reduce_sum(b)
    loss = tf.minimum(loss_a, loss_b)
    
    optimizer = tf.train.AdamOptimizer(l_rate)
    train = optimizer.minimize(loss, var_list=[C_a, C_b, C_a_1, C_a_2, C_a_3, C_b_1, C_b_2, C_b_3, B_reduce])

    ##################

    init = tf.initialize_all_variables()
    #init_F = tf.initialize_variables(var_list=[F])

    with tf.Session() as sess:
        iterations = Size/batch_size
        sess.run(init)
        loss_lst = [None]*iterations
        lossA_lst = [None]*iterations
        lossB_lst = [None]*iterations
        #Psi_6_lst = []
        variable_lst = []
        #Cb_list = []
        
        for i in xrange(iterations):
            #sess.run(init_F)
            sess.run(train, feed_dict={Adj: DATA[i:i+batch_size], F: Signal, true_assignment_a: TRUE_A, true_assignment_b: TRUE_B})
            loss_printed, loss_A, loss_B = sess.run([loss, loss_a, loss_b], 
                feed_dict={Adj: DATA[i:i+batch_size], F: Signal, true_assignment_a: TRUE_A, true_assignment_b: TRUE_B})
            loss_lst[i]=loss_printed

            lossA_lst[i]=loss_A
            lossB_lst[i]=loss_B
            if i%20==0:
                print i, "loss", loss_printed
                d = {"loss": loss_lst, "loss_A": lossA_lst, "loss_B": lossB_lst}
                d = pd.DataFrame(d)
                d.to_csv("~/Desktop/clusternet/GNN/data/dataConstant/Size{}Mean{}l_rate{}Mean_signal{}group_size{}batch_size{}p_min{}p_max{}.csv".format(Size, Mean, l_rate, Mean_signal, group_size, batch_size, p_min, p_max))
               
            if i==iterations-1:
                print "these are the variables after training:"
                print "mean:", m, "l rate:", l, "Mean_signal:", n, 

                f, a, b, a1, a2, a3, b1, b2, b3, b_reduce = sess.run([F, C_a, C_b, C_a_1, C_a_2, C_a_3, C_b_1, C_b_2, C_b_3, B_reduce], 
                    feed_dict={Adj: DATA[i:i+batch_size], F: Signal, true_assignment_a: TRUE_A, true_assignment_b: TRUE_B})

                variable_lst = variable_lst+[f, a, b, a1, a2, a3, b1, b2, b3, b_reduce]
                d_var = {"vars_a_b_ai_bi_b_reduce": variable_lst, "header": ["F", "Ca", "Cb", "Ca1", "Ca2", "Ca3", 
                "Cb1", "Cb2", "Cb3", "B_reduce"]}

                d_var = pd.DataFrame(d_var)
                d_var.to_csv("~/Desktop/clusternet/GNN/data/dataConstant/ModelPARAMS_Size{}Mean{}l_rate{}Mean_signal{}group_size{}batch_size{}p_min{}p_max{}.csv".format(Size, Mean, l_rate, Mean_signal, group_size, batch_size, p_min, p_max))
           
                #print d_var

iterated_list = [(m, s, l, n, g, b, p) for m in [0] for 
     s in [1] for 
     l in [10**-i for i in [3]] for #xrange(6, 14, 4)] for 
     n in [0] for 
     g in [20] for
     b in [1] for 
     p in [0.3, 0.4, 0.5]]

k = np.random.randint(1, 1000)

def GNN_seq_2000(v):
    return GNN(Size=500, Mean=v[0], SD=v[1], l_rate=v[2], Mean_signal = v[3], group_size=v[4], batch_size = v[5],
               p_min = v[6], p_max = v[6], seed=k)
for i in iterated_list:
    GNN_seq_2000(i)      
  
