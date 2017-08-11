
# coding: utf-8

# In[3]:


import numpy as np

#BUILD PCA

np.random.seed(1)
#create dataset

mu_vec1=np.array([0,0,0])#mean [3]

#covariance=measure how changes in one variable are associated in the changes of a second variable 
#(how 1 variable change in relation to another variable). How 2 features are related
cov_mat1=np.array([[1,0,0],[0,1,0],[0,0,1]])#covariance[3,3]

#multivariate_normal (gaussian distribution [distrib of possibilities] in higher dimension),
#MEAN: define the center of the distribution, COVARIANCE define the width of stretch of the data(the distribution) 
class1_sample=np.random.multivariate_normal(mu_vec1,cov_mat1,20).T #[3x20], t=flip the matrix [20x3]->[3x20]


# In[4]:


class1_sample


# In[8]:


mu_vec2=np.array([1,1,1])
cov_mat2=np.array([[1,0,0],[0,1,0],[0,0,1]])

class2_sample=np.random.multivariate_normal(mu_vec2,cov_mat2,20).T


# In[9]:


class2_sample


# In[16]:


#PLOTTING THE DATA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#figure (so the space where data are plotted), width and height
fig=plt.figure(figsize=(8,8))

#3d subplot
#subplot grid parameters, 1x1grid,1st subplot
#ax=fig.add_subplot(111, projection='3d')

ax=Axes3D(fig)
#fontsize
plt.rcParams['legend.fontsize']=10
#plot samples
ax.plot(class1_sample[0,:],class1_sample[1,:],class1_sample[2,:], 
        'o', color='blue', alpha=0.5,label='class1')

ax.plot(class2_sample[0,:],class2_sample[1,:],class2_sample[2,:],
       '^', alpha=0.5, color='green', label='class2')

ax.legend(loc='upper right')

plt.show()


# In[40]:


#step3 --- merge data in 1 dataset

all_samples=np.concatenate((class1_sample, class2_sample), axis=1)# [3x40] dataset is created


# In[41]:


#step4 --- compute the dimensional mean  for each feature in order to computer the covariance matrix

mean_x=np.mean(all_samples[0,:])
mean_y=np.mean(all_samples[1,:])
mean_z=np.mean(all_samples[2,:])

#3d mean vector
mean_vector=np.array([[mean_x] , [mean_y], [mean_z]])
mean_vector


# In[43]:


cov_mat=np.zeros((3,3))
#cov_mat = np.cov([all_samples[0,:],all_samples[1,:],all_samples[2,:]])
print(all_samples.shape[1])
for i in range((all_samples.shape[1])):
    cov_mat += (all_samples[:,i].reshape(3,1)-mean_vector).dot((all_samples[:,i].reshape(3,1)))

print('covariance matrix:', cov_mat)


# In[ ]:





# In[ ]:




