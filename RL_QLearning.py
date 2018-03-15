import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import linalg as LA

def tau(s,a):
    if s==0 or s==4:  return(s)
    else:      return(s+a)
def rho(s,a):
    return(s==1 and a==0)+2*(s==3 and a==1)
def calc_policy(Q):
    policy=np.zeros(5)
    for s in range(0,5):
        action_idx=np.argmax(Q[s,:])
        policy[s]=2*action_idx-1
        policy[0]=policy[4]=0
    return policy.astype(int)
def idx(a):
    return(int((a+1)/2))

#These lines establish the feed-forward part of the network used to choose actions
# Here we initialize the variables for our neural network architecture
Weights = tf.Variable(tf.random_uniform([5,2],0,0.01))
inputData = tf.placeholder(shape=[1,5],dtype=tf.float32)
Qoutput = tf.matmul(inputData,Weights)
nextQ = tf.placeholder(shape=[1,2],dtype=tf.float32)
predictedValues = tf.argmax(Qoutput,1)
netLoss = tf.reduce_sum(tf.square(nextQ - Qoutput))
trainingModel = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updatedModel = trainingModel.minimize(netLoss)
arr = np.array( [[ 0.,   1.,   0.5,  0.5,  0. ], [ 0.,   0.5,  1.,   2.,   0. ]])

init = tf.global_variables_initializer()

# Set learning parameters
discountFactor = .05
e = 0.1
num_episodes = 670
loss = []
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        # Randomly pickup a state
        s = np.random.randint(0,5)
        rAll = 0
        j = 0
        while j < 5:
            j+=1
            a,allQ = sess.run([predictedValues,Qoutput],feed_dict={inputData:np.identity(5)[s:s+1]})
            sNew = tau(s,a[0])
            Q1 = sess.run(Qoutput,feed_dict={inputData:np.identity(5)[sNew:sNew+1]})
            maxQ1 = np.max(Q1)
            targetQ = allQ 
            r = rho(s,a[0])
            targetQ[0,a[0]] = r + discountFactor*maxQ1
            # Training Phase
            sess.run([updatedModel,Weights],feed_dict={inputData:np.identity(5)[s:s+1],nextQ:targetQ})
            s = sNew
        loss.append(LA.norm(tf.global_variables()[0].eval().T-arr))
    policy = calc_policy(tf.global_variables()[0].eval())
    print('Best Policy :- '+str(policy))

plt.suptitle('Deep Q Learning Cost Function')
plt.xlabel('Episodes')
plt.ylabel('Loss')
plt.plot(loss)
plt.show()
