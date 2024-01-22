"""
Author: Pierce Rotman
Professor: Pashaie
Course: CAP 6673
Date: 17 October 2023

Description: Use eigenvalue decomposition to perform PCA on data set containing images of faces. Then, see if this
PCA can be used to classify images using two people as an example. 
"""


import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

#Load data and visualize image sets
faces = scipy.io.loadmat("/Users/piercerotman/Documents/"+
                         "MastersProgram/Machine_Learning/allFaces.mat")

nfaces = np.array(faces['nfaces'])
faces = np.array(faces['faces'])
n = 192
m=168

#Look at first image of each person
person = np.zeros((n,m,36))
plt.figure(1)
for i in range(36):
    person[:, :, i] = faces[:, sum(nfaces[0,0:i-1])].reshape(m,n).transpose()
    plt.subplot(6,6,i+1)
    plt.imshow(person[:,:,i], cmap = 'gray')
    plt.axis('off')
    plt.gca().set_aspect(0.8802, adjustable = 'box')

plt.show()

#Look at all images of person 1
snapshot = np.zeros((n,m,64))
for i in range(len(nfaces[0,:])):
    subset = faces[:,sum(nfaces[0,0:i]):sum(nfaces[0,0:i+1])]
    for j in range(nfaces[0,i]):
        snapshot[:,:,j] = subset[:,j].reshape(m,n).transpose()

plt.figure(1)
for i in range(64):
    plt.subplot(8,8, i+1)
    plt.imshow(snapshot[:,:,i], cmap = 'gray')
    plt.axis('off')
    plt.gca().set_aspect(0.8802, adjustable = 'box')
plt.show()


#Calculate and visualize the average face for all images
average_face = np.sum(faces[:, 0:sum(nfaces[0,0:36])], axis=1) / sum(nfaces[0,0:36])
plt.figure(1)
plt.imshow(average_face.reshape(168,192), cmap = 'gray')
plt.show()

#Center data
X = np.array([faces[:,i]-average_face for i in range(sum(nfaces[0,0:36]))])

# Perform SVD on centered data
U, S, V = np.linalg.svd(X.transpose(), full_matrices=False)

#Visualize first 54 eigenfaces
plt.figure(1)
for i in range(54):
    plt.subplot(8,7, i+1)
    plt.imshow(U[:,i].reshape(168,192).T, cmap = 'gray')
    plt.axis('off')
    plt.gca().set_aspect(0.8802, adjustable = 'box')
plt.show()

#Verify orthogonality of some eigenfaces
firstfifth = U[:,0].dot(U[:,4].transpose())
print(firstfifth)

tenthfifteenth = U[:,9].dot(U[:,14].transpose())
print(tenthfifteenth)

#Decompose person 37 into eigenfaces and see how many are needed to identify person
p37 = faces[:,sum(nfaces[0, 0:36])]

rs = [5, 10, 200, 800, 100]
for r in rs:
    V = np.zeros(32256)
    for i in range(r):
        V += U[:,i].transpose().dot(p37)*(U[:,i])
    
    plt.figure(1)
    plt.title(f"Number of eigenfaces: {r}")
    plt.imshow(V.reshape(168,192).T, cmap = 'gray')
    plt.show()


#Plot singular values on logarithmic scale
plt.figure(1)
plt.plot(S)
plt.semilogy()
plt.xlabel("Singular Value")
plt.ylabel("Magnitude (logarithmic scale)")
plt.title("Magnitude of Singular Value")
plt.show()

#Decompose person 2 and person 7 into eigenfaces 5 and 6 and plot these decompositions
p2 = faces[:, sum(nfaces[0,0:1]): sum(nfaces[0,0:2])]
p7 = faces[:, sum(nfaces[0,0:6]): sum(nfaces[0,0:7])]

p2e5 = []
for i in range(62):
    p2e5.append(p2[:,i].transpose().dot(U[:,4]))
p2e6 = []
for i in range(62):
    p2e6.append(p2[:,i].transpose().dot(U[:,5]))

p7e5 = []
for i in range(64):
    p7e5.append(p7[:,i].transpose().dot(U[:,4]))
p7e6 = []
for i in range(64):
    p7e6.append(p7[:,i].transpose().dot(U[:,5]))

x1 = p2e5 + p7e5
x2 = p2e6 + p7e6
X = np.array([x1, x2]).transpose()

plt.scatter(x=X[:62,0], y=X[:62,1], color = 'black', label = 'Person 2')
plt.scatter(x=X[62:,0], y=X[62:,1], color = 'red', label = 'Person 7')
plt.xlabel('Eigenface 5')
plt.ylabel('Eigenface 6')
plt.title("6th vs 5th Eigenface for Person 2 and 7")
plt.legend()
plt.show()

#Perform SVM to classify images for person 5 and 6
import seaborn as sns
d = [-1 for i in range(62)] + [1 for i in range(64)]
Y = np.array([d]).transpose()

m = X[:,0].size
midm = int(m/2)
n = X[1,:].size

def poly(x1, x2, order):
    """
    Calculates polynomial kernel for two vectors of X, given order
    Args:
        x1 (numpy.ndarray): value1
        x2 (numpy.ndarray): value2
        order (int): order of polynomial
    Returns:
        (float) poly of x1, x2
    """
    return (1 - np.dot(x1, x2)) ** order



#Create variable and Q matrix
alphas = np.zeros(126)
ones = np.ones(126)
Q = np.zeros((126,126))
for i in range(126):
    for j in range(126):
        Q[i,j] = Y[i,:]*Y[j,:]*poly(X[i,:],X[j,:], 1)



#Make KKT conditions
constraints = {'type': 'eq', 'fun': lambda alphas: Y.transpose().dot(alphas)}
def function(alphas):
    return -0.005 * alphas.transpose().dot(Q).dot(alphas) - ones.transpose().dot(alphas)



minimization = scipy.optimize.minimize(function, alphas, constraints = constraints, bounds = [(0, 1) for i in range(m)])
optimized_alphas = minimization.x
sv_indices = np.where(optimized_alphas > 0.001)[0]

#find b:
def decision_boundary(x):
    """
    Calculates decision function
    Args:
        x (numpy.ndarray): value to calculate decision function for
    Returns:
        (float) decision function value
    """
    return sum([optimized_alphas[i] * Y[i,:][0] * poly(X[i,:], x, 1) for i in sv_indices])


W = np.zeros((2,1))
for i in sv_indices:
    W[0] += optimized_alphas[i] * Y.flatten()[i] * X[i,0]
    W[1] += optimized_alphas[i] * Y.flatten()[i] * X[i,1]

b = 1 - decision_boundary(X[sv_indices[1],:])
fig, ax = plt.subplots()
plt.scatter(x=X[:62,0], y=X[:62,1], color = 'black', label = 'Person 2')
plt.scatter(x=X[62:,0], y=X[62:,1], color = 'red', label = 'Person 7')
plt.xlabel('Eigenface 5')
plt.ylabel('Eigenface 6')
plt.title("Person 2 and Person 7 With SVM Classifier")
plt.scatter(x=X[23,0], y=X[23,1], color='green', marker = 'x')
plt.scatter(x=X[47,0], y=X[47,1], color='green', marker = 'x')
plt.scatter(x=X[125,0], y=X[125,1], color='green', marker = 'x')
plt.legend()
sns.lineplot(x = X[:,0], y = -(W[0]*X[:,0]-b)/W[1])
plt.show()
