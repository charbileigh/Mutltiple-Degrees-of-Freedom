#libraries
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from scipy.linalg import eigh

n_steps = 500 #number of time steps
end_time = 1 # last time, in seconds
time = np.linspace(0,end_time,n_steps) #time vector
delta_t = time[1] - time[0] #timestep size, python is 0-indexed

#define system
dummy = np.zeros((2, 1))
print(dummy)
dof = int(input("How many degree/degrees of freedom will the system have:\n"))
print("\nThe chosen degree/degrees of freedom are:\n" + str(dof) +"\n")

# Fzero = np.zeros((dof, 1)) #force amplitude
# Omega = np.zeros((dof, 1)) #force angular speed

#define matrices

# M = np.zeros((dof, dof))
# C = np.zeros((dof, dof))
# K = np.zeros((dof, dof))

print('\nIn order to enter all your inputs for the column vectors, enter a value seperated by a space for the exact amount of series present in the system.\n')
print("If uncertain about how to input values please refer to READ ME text")

# for i in range(0, dof):
#     print('\nEnter the ' + str(i) + "th vector (column) of the mass matrix\n")
#     mi = list(map(float,input().split()))
#     M[:,i] = mi
M = np.array([[1, 0], [0, 1]])
print("\nThe mass matrix is:\n")
print(M)
print(type(M))
print("\n")

# for j in range(0, dof):
#     print('\nEnter the ' + str(j) + "th vector (column) of the damping matrix\n")
#     cj = list(map(float,input().split()))
#     C[:,j] = cj
C = np.array([[400, -200], [-200, 200]])
print("\nThe damping matrix is:\n")
print(C)
print(type(C))
print("\n")

# for k in range(0, dof):
#     print('\nEnter the ' + str(k) + "th vector (column) of the stiffness matrix\n")
#     kk = list(map(float,input().split()))
#     K[:,k] = kk
K = np.array([[1000, -500], [-500, 500]])
print("\nThe stiffness matrix is:\n")
print(K)
print(type(K))
print("\n")

# print("Enter the Forcezero vector:\n")
# fj = list(map(float,input().split()))
# Fzero[:,0] = fj
Fzero = np.array([[5], [0]])
print("\nThe forcezero vector is:\n")
print(Fzero)
print("\n")

print("\nEnter the Forced Frequency vector assosociated with the force vector:\n")
# oj = list(map(float,input().split()))
# Omega[:,0] = oj
Omega = np.array([[1], [0]])
print("\nThe Forced Frequency vector is:\n")
print(Omega)
print("\n")

Minv = np.linalg.inv(M)
Kinv = np.linalg.inv(K)

F = np.multiply(Fzero, np.sin(np.multiply(time,Omega)))

#empty vectors to store solution
x = np.zeros((dof, n_steps))
v = np.zeros((dof, n_steps))
a = np.zeros((dof, n_steps))

#initial conditions
print("Enter the initial conditions for position:\n")
xj = list(map(float,input().split()))
print("xj")
print(xj)
x[:,1] = xj
print("\nThe initial conditions for position are:\n")
print(x)
print("\n")

print("Please enter the initial conditions for velocity")
vj = list(map(float,input().split()))
print("vj")
print(vj)
v[:,1] = vj
print("\nThe initial conditions for velocity are:\n")
print(v)
print("\n")

a[:,1] = np.dot(Minv, (F[:, 0] - np.dot(C,v[:, 1]) - np.dot(K, x[: ,1])))


#solve for t(-1)
x[: ,0] = x[:, 1] - np.multiply(delta_t ,v[: ,1]) + np.multiply((delta_t**2)/2, a[:, 1])#changed the code here in order to accomodate the change created by moving everything backwards

#pre-loop calculations to improve effciency
A1 = M/(delta_t**2) 
A2 = C/(2*delta_t)
A3 = A1 + A2
A4 = K - 2*A1
A5 = A1 - A2
A6 = 1/(2*delta_t)
A7 = 1/(delta_t**2)

A3inv = np.linalg.inv(A3) #inverse of contant matrix A3

#looping over all timesteps
for n in range(n_steps-2):
    
    x[:,n+2] = np.dot(A3inv,(F[:,n+1] - np.dot(A4,x[:,n+1]) - np.dot(A5,x[:, n])))
    
xt = x.T #transponse of displacement
print("x transpose")
print(xt)
xt_1 = xt[:,0] #displacement of mass 1
xt_2 = xt[:,1] #displacement of mass 2
print("xt_2")
print(xt_2)


#exact solution for an underdamped system
z1 = 1.0514*np.cos(13.82*time) + 0.06156*np.sin(13.82*time) + 0.01414*np.cos(time)
z2 =  -1.7012*np.cos(36.18*time) + 0.01453*np.sin(36.18*time) - 0.003252*np.cos(time)

x_exact_1 = 0.5271*z1 - 0.8507*z2
x_exact_2 = 0.8507*z1 + 0.5257*z2

#plot results
plt.figure(0)
plt.subplot(312)
plt.plot(time, xt_1,'-r', time, x_exact_1,'--k')
plt.xlabel('time - $t (s)$')
plt.ylabel('X_exact1 vs X_1 \n $(m)$\n ')
plt.title('Comparison of CDM and Analytical solutions')
plt.subplot(313)
plt.plot(time, xt_2,'-r', time, x_exact_2,'k--')
plt.xlabel('time - $t (s)$')
plt.ylabel('X_exact2 vs X_2 \n $(m)$\n ')
plt.legend(['CDM','Analytical'], loc='best')
plt.tight_layout(rect=[0, 0.0001, 1, 2])
plt.show()
