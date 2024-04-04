import numpy as np
import sympy as sy
from scipy.optimize import fsolve
import control as ctrl
import control.matlab as ctrlmatlab
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import ss2tf




def biononlin(x,t,mu_max,Ks,Y,u,d):
    X=x[0]; S=x[1]

    par=[mu_max, Ks, Y]; u=Df; d=Sf
    mu=mu_max*S/(Ks+S)
    
    dXdt=-Df*X+mu*X
    dSdt=Df*(Sf-S)-mu*X/Y
    dxdt = [dXdt,dSdt]
    return dxdt


mu_max = 0.5; Ks = 0.1; Y = 0.4
Df = 0.35; Sf = 1.0 

x0 = [0.1,0.1]

t = np.arange(0,60,0.0001)

x = odeint(biononlin,x0,t,args=(mu_max,Ks,Y,Df,Sf))


# Plotagem da modelo não linear
plt.plot(t, x[:, 0], 'b-', label='X')
plt.plot(t, x[:, 1], 'r-', label='S')
plt.ylabel('Concentrações')
plt.xlabel('Tempo')
plt.show()


# Pontos de equilíbrio
xss = fsolve(biononlin, x0, args=(t, mu_max, Ks, Y, Df, Sf))


# Linearizar

mu_max, Ks, Y, Df, Sf, S, X = sy.symbols('mu_max, Ks, Y, Df, Sf, S, X')
mu = (mu_max*S)/(Ks + S)

dXdt = -Df*X + mu*X
dSdt =  Df*(Sf - S) - (mu*X)/Y
F = sy.Matrix([dXdt, dSdt])
A_ss = F.jacobian([X, S])
B_ss = F.jacobian([Df, Sf])


A = np.asarray(A_ss.evalf(subs={mu_max:0.5, Ks: 0.1, Y: 0.4, Df: 0.35, Sf: 1.0, X: xss[0], S: xss[1]}))
B = np.asarray(B_ss.evalf(subs={mu_max:0.5, Ks: 0.1, Y: 0.4, Df: 0.35, Sf: 1.0, X: xss[0], S: xss[1]}))
C = np.matrix([[1, 0], [0, 1]])
D = np.matrix([[0, 0], [0, 0]])

biolin = ctrl.ss(A, B, C, D)
yout, T, xout = ctrlmatlab.lsim(biolin, 0, t, x0-xss)

Xlin = np.array(xout[:,0]) + xss[0]
Slin = np.array(xout[:,1]) + xss[1]

# Comparação de modelos lineares
plt.plot(T, Xlin, 'r-', label='X_lin(t)')
plt.plot(t, x[:,0], 'b-', label='X_nl(t)')
plt.ylabel('concentrações')
plt.xlabel('Tempo')
plt.legend(loc='best')
plt.show()

plt.plot(T, Slin, 'r-', label='S_lin(t)')
plt.plot(t, x[:,1], 'b-', label='S_nl(t)')
plt.ylabel('concentrações')
plt.xlabel('Tempo')
plt.legend(loc='best')
plt.show()

A = np.array([[5.55111512312578e-17, 0.138000000000000], [-0.875000000000000, -0.695000000000000]])
B = np.array([[-0.30, 0], [0.766666666666667, 0.350000000000000]])

print(A)
print(B)
print(C)
print(D)

tf = ss2tf(A, B, C, D, 0)

print(tf)

