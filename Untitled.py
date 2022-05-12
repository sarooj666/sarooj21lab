#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib;pyplot as plt


# In[ ]:


# Question # 2

Plot the first derivative of the following function over $[-1,1]$ by using the forward, backward and central differences.

Compare the plots with the theoretical one at $h = 0.1, 0.01$ and $0.001$

$f(x) = 0.1x^5 - 0.2x^3 + 0.1x - 0.2$


# In[19]:


f = lambda x: 0.1*x**5 - 0.2*x**3 + 0.1*x-0.2
h = 0.001
x = np.linspace(-1,1) # will create an array of elemets between -1 and 1 having 20 equal 
#farword

dff1 = (f(x)-f(x-h))/h
dff2 = (f(x)-2*f(x-h)+f(x-2*h))/h**2
 #plot 
plt.plot (x,f(x), '-k', x,dff1, '--b',x,dff2,'-.r')
plt.xlabel('x')
plt.ylabel('y')

plt.legend (["f(x)", "f'(x)", "f''(x)"])
plt.grid()


# In[20]:


f = lambda x: 0.1*x**5 - 0.2*x**3 + 0.1*x-0.2
h = 0.001
x = np.linspace(-1,1) # will create an array of elemets between -1 and 1 having 20 equal 
dff1 = (f(x+h)-f(x))/h
dff2 = (f(x+2*h)-2*f(x+h)+f(x))/h**2
#backward

 #plot 
plt.plot (x,f(x), '-k', x,dff1, '--b',x,dff2,'-.r')
plt.xlabel('x')
plt.ylabel('y')

plt.legend (["f(x)", "f'(x)", "f''(x)"])
plt.grid()


# In[21]:


f = lambda x: 0.1*x**5 - 0.2*x**3 + 0.1*x-0.2
h = 0.001
x = np.linspace(-1,1) # will create an array of elemets between -1 and 1 having 20 equal 
#central

dff1 = (f(x+h) - f(x-h))/(2*h)
dff2 = ((f(x+h) - 2*f(x)) + f(x-h))/h**2
 #plot 
plt.plot (x,f(x), '-k', x,dff1, '--b',x,dff2,'-.r')
plt.xlabel('x')
plt.ylabel('y')

plt.legend (["f(x)", "f'(x)", "f''(x)"])
plt.grid()


# In[38]:


f=lambda x: 0.1*x*5 - 0.2*x*3 + 0.1*x - 0.2
x = 0.6
h = 0.1
df1 = -0.0512
df2 = -0.288
print("\t f'(x)\t\t err\t\t f''(x)\t\t err")
dff1 = (f(x+h)-f(x))/h
dff2 = (f(x+2*h)-2*f(x+h)+f(x))/h**2
print("FFD\t% f\t% f\t% f\t% f"%(dff1,dff1-df1,dff2,dff2-df2))
dff1 = (f(x)-f(x-h))/h
dff2 = (f(x)-2*f(x-h)+f(x-2*h))/h**2
print("BFD\t% f\t% f\t% f\t% f"%(dff1,dff1-df1,dff2,dff2-df2))
dff1 = (f(x+h)-f(x-h))/(2*h)
dff2 = (f(x+h)-2*f(x)+f(x-h))/h**2
print("CFD\t% 0.8f\t% 0.8f\t% 0.8f\t% 0.8f"%(dff1,dff1-df1,dff2,dff2-df2))


# In[23]:


f = lambda x: 0.1*x**5 - 0.2*x**3 + 0.1*x-0.2
h = 0.01
x = np.linspace(-1,1) # will create an array of elemets between -1 and 1 having 20 equal 
#farword

dff1 = (f(x)-f(x-h))/h
dff2 = (f(x)-2*f(x-h)+f(x-2*h))/h**2
 #plot 
plt.plot (x,f(x), '-k', x,dff1, '--b',x,dff2,'-.r')
plt.xlabel('x')
plt.ylabel('y')

plt.legend (["f(x)", "f'(x)", "f''(x)"])
plt.grid()


# In[30]:


f = lambda x: 0.1*x**5 - 0.2*x**3 + 0.1*x-0.2
h = 0.01
x = np.linspace(-1,1) # will create an array of elemets between -1 and 1 having 20 equal 
dff1 = (f(x+h)-f(x))/h
dff2 = (f(x+2*h)-2*f(x+h)+f(x))/h**2
#backward

 #plot 
plt.plot (x,f(x), '-k', x,dff1, '--g',x,dff2,'-r')
plt.xlabel('x')
plt.ylabel('y')

plt.legend (["f(x)", "f'(x)", "f''(x)"])
plt.grid()


# In[25]:


f = lambda x: 0.1*x**5 - 0.2*x**3 + 0.1*x-0.2
h = 0.01
x = np.linspace(-1,1) # will create an array of elemets between -1 and 1 having 20 equal 
#central

dff1 = (f(x+h) - f(x-h))/(2*h)
dff2 = ((f(x+h) - 2*f(x)) + f(x-h))/h**2
 #plot 
plt.plot (x,f(x), '-k', x,dff1, '--b',x,dff2,'-.r')
plt.xlabel('x')
plt.ylabel('y')

plt.legend (["f(x)", "f'(x)", "f''(x)"])
plt.grid()


# In[34]:


f=lambda x: 0.1*x*5 - 0.2*x*3 + 0.1*x - 0.2
x = 0.6
h = 0.01
df1 = -0.0512
df2 = -0.288
print("\t f'(x)\t\t err\t\t f''(x)\t\t err")
dff1 = (f(x+h)-f(x))/h
dff2 = (f(x+2*h)-2*f(x+h)+f(x))/h**2
print("FFD\t% f\t% f\t% f\t% f"%(dff1,dff1-df1,dff2,dff2-df2))
dff1 = (f(x)-f(x-h))/h
dff2 = (f(x)-2*f(x-h)+f(x-2*h))/h**2
print("BFD\t% f\t% f\t% f\t% f"%(dff1,dff1-df1,dff2,dff2-df2))
dff1 = (f(x+h)-f(x-h))/(2*h)
dff2 = (f(x+h)-2*f(x)+f(x-h))/h**2
print("CFD\t% f\t% f\t% f\t% f"%(dff1,dff1-df1,dff2,dff2-df2))


# In[26]:


f = lambda x: 0.1*x**5 - 0.2*x**3 + 0.1*x-0.2
h = 0.1
x = np.linspace(-1,1) # will create an array of elemets between -1 and 1 having 20 equal 
#farword

dff1 = (f(x)-f(x-h))/h
dff2 = (f(x)-2*f(x-h)+f(x-2*h))/h**2
 #plot 
plt.plot (x,f(x), '-k', x,dff1, '--b',x,dff2,'-.r')
plt.xlabel('x')
plt.ylabel('y')

plt.legend (["f(x)", "f'(x)", "f''(x)"])
plt.grid()


# In[27]:


f = lambda x: 0.1*x**5 - 0.2*x**3 + 0.1*x-0.2
h = 0.1
x = np.linspace(-1,1) # will create an array of elemets between -1 and 1 having 20 equal 
dff1 = (f(x+h)-f(x))/h
dff2 = (f(x+2*h)-2*f(x+h)+f(x))/h**2
#backward

 #plot 
plt.plot (x,f(x), '-k', x,dff1, '--b',x,dff2,'-.r')
plt.xlabel('x')
plt.ylabel('y')

plt.legend (["f(x)", "f'(x)", "f''(x)"])
plt.grid()


# In[32]:


f = lambda x: 0.1*x**5 - 0.2*x**3 + 0.1*x-0.2
h = 0.1
x = np.linspace(-1,1) # will create an array of elemets between -1 and 1 having 20 equal 
#central

dff1 = (f(x+h) - f(x-h))/(2*h)
dff2 = ((f(x+h) - 2*f(x)) + f(x-h))/h**2
 #plot 
plt.plot (x,f(x), '-k', x,dff1, '--r',x,dff2,'-g')
plt.xlabel('x')
plt.ylabel('y')

plt.legend (["f(x)", "f'(x)", "f''(x)"])
plt.grid()


# In[35]:


f=lambda x: 0.1*x*5 - 0.2*x*3 + 0.1*x - 0.2
x = 0.6
h = 0.001
df1 = -0.0512
df2 = -0.288
print("\t f'(x)\t\t err\t\t f''(x)\t\t err")
dff1 = (f(x+h)-f(x))/h
dff2 = (f(x+2*h)-2*f(x+h)+f(x))/h**2
print("FFD\t% f\t% f\t% f\t% f"%(dff1,dff1-df1,dff2,dff2-df2))
dff1 = (f(x)-f(x-h))/h
dff2 = (f(x)-2*f(x-h)+f(x-2*h))/h**2
print("BFD\t% f\t% f\t% f\t% f"%(dff1,dff1-df1,dff2,dff2-df2))
dff1 = (f(x+h)-f(x-h))/(2*h)
dff2 = (f(x+h)-2*f(x)+f(x-h))/h**2
print("CFD\t% f\t% f\t% f\t% f"%(dff1,dff1-df1,dff2,dff2-df2))


# In[45]:


from math import sin
def newton(fn,dfn,x,tol,maxiter):
    for i in range(maxiter):
        xnew = x - fn(x)/dfn(x)
        if abs(xnew-x)<tol:                          
            break
        x = xnew
    return xnew, i

y = lambda x: 2*x**2 - 5*x + 3 
dy = lambda x : 4*x - 5

x, n = newton(y, dy, 5, 0.0001, 100)
print('the root is %.3f at %d iterations.'%(x,n))


# In[ ]:




