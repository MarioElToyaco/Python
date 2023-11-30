import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive
import time
from scipy.optimize import curve_fit
from matplotlib.axes._axes import _log as matplotlib_axes_logger

M_sun = 1.98892e33            #in grams
M_galaxia = 1e12*M_sun        #in grams
M_disco = 5e10*M_sun          #in grams
Un_año = 365.2422*24*3600     #in seconds

#Units
Unit_time = 9.8e8*Un_año    #in seconds 
Unit_velocity = 1e5         #cm/s
Unit_length = 3.085678e21   #cm
Unit_mass = 1.989e33        #M_sun
G = 4.302e-6                #kpc*((km/s)**2)/Msun

#NFW profile
C = 12.5            
R_v = 250           # kpc
R_s = 20            # kpc
rho_0 = 5932371     # Msun/kpc^3

t0 = 0        
tf = 2*Unit_time  
steps = 1000    
dt = tf/steps  

#The enclosed mass is defined for any radious
def enclosed_mass(rho_0, R_s, R_max):
    M = 4*np.pi*rho_0*R_s**3*(np.log((R_s+R_max)/R_s) - R_max/(R_s + R_max))
    return M

#The Euler's and Rungee Krutta's numeric integration methods are defined
def euler(problem, t, x, y, z, vx, vy, vz, h):
    sol = problem(t, x, y, z, vx, vy, vz)
    
    xi = x + h*sol[0]    ;  yi = y + h*sol[1]    ;  zi = z + h*sol[2]
    vxi = vx + h*sol[3]  ;  vyi = vy + h*sol[4]  ;  vzi = vz + h*sol[5]
    return xi, yi, zi, vxi, vyi, vzi
  
def RK2(problem, t, x, y, z, vx, vy, vz, h):
    k1 = problem(t, x, y, z, vx, vy, vz)
    k2 = problem(t + h/2, x + k1[0]*h/2, y + k1[1]*h/2, z + k1[2]*h/2, vx + k1[3]*h/2, vy + k1[4]*h/2, vz + k1[5]*h/2)
    
    xi = x + k2[0]*h
    yi = y + k2[1]*h
    zi = z + k2[2]*h
    vxi = vx + k2[3]*h
    vyi = vy + k2[4]*h
    vzi = vz + k2[5]*h
    return xi, yi, zi, vxi, vyi, vzi

def RK4(problem, t, x, y, z, vx, vy, vz, h):
    k1 = problem(t, x, y, z, vx, vy, vz)
    k2 = problem(t + h/2., x + k1[0]*h/2, y + k1[1]*h/2, z + k1[2]*h/2, vx + k1[3]*h/2, vy + k1[4]*h/2, vz + k1[5]*h/2)
    k3 = problem(t + h/2., x + k2[0]*h/2, y + k2[1]*h/2, z + k2[2]*h/2, vx + k2[3]*h/2, vy + k2[4]*h/2, vz + k2[5]*h/2)
    k4 = problem(t + h, x + k3[0]*h, y + k3[1]*h, z + k3[2]*h, vx + k3[3]*h, vy + k3[4]*h, vz + k3[5]*h)
        
    xi = x + (h/6.)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    yi = y + (h/6.)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    zi = z + (h/6.)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
    vxi = vx + (h/6.)*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
    vyi = vy + (h/6.)*(k1[4] + 2*k2[4] + 2*k3[4] + k4[4])
    vzi = vz + (h/6.)*(k1[5] + 2*k2[5] + 2*k3[5] + k4[5])
    return xi, yi, zi, vxi, vyi, vzi

#The integration function is defined
def integrator(problem, method, t0, tf, h):
    t = [t0]
    x = [x0]  ;  vx = [vx0*Unit_velocity/Unit_length]
    y = [y0]  ;  vy = [vy0*Unit_velocity/Unit_length]
    z = [z0]  ;  vz = [vz0*Unit_velocity/Unit_length]

    while t[-1] < tf:
        sol = method(problem, t[-1], x[-1], y[-1], z[-1], vx[-1], vy[-1], vz[-1], h)
        sol_x = sol[0]   ;  sol_y = sol[1]   ;  sol_z = sol[2]
        sol_vx = sol[3]  ;  sol_vy = sol[4]  ;  sol_vz = sol[5]
        x.append(sol_x)    ;  y.append(sol_y)    ;  z.append(sol_z)
        vx.append(sol_vx)  ;  vy.append(sol_vy)  ;  vz.append(sol_vz)
        t.append(t[-1] + h)
        
    return np.array(t), np.array(x), np.array(y), np.array(z)

## Part 1: The Sun revolving around the Centre of the Galaxy.

#The orbit function is defined (no particle interaction)
def orbita(t, x, y, z, vx, vy, vz):
    r = np.sqrt(x**2 + y**2 + z**2)
    mass = enclosed_mass(rho_0, R_s, r)
    ax = (-x*G*mass/r**3)*(Unit_velocity/Unit_length)**2
    ay = (-y*G*mass/r**3)*(Unit_velocity/Unit_length)**2
    az = (-z*G*mass/r**3)*(Unit_velocity/Unit_length)**2
    return vx, vy, vz, ax, ay, az

#Variables
a0 = np.array([8])       # kpc
phi0 = np.array([0])     # radianes

#Initial position
x0 = a0*np.cos(phi0)    # kpc
y0 = a0*np.sin(phi0)    # kpc
z0 = np.array([0])      # kpc
R0 = np.sqrt(x0**2 + y0**2 + z0**2) #kpc

#Initial velocity
vx0 = np.array([0])     # km/s
vy0 = np.array([127])   # km/s
vz0 = np.array([0])     # km/s

t, x_eul, y_eul, z_eul = integrator(orbita, euler, t0, tf, dt)
t, x_RK2, y_RK2, z_RK2 = integrator(orbita, RK2, t0, tf, dt)
t, x_RK4, y_RK4, z_RK4 = integrator(orbita, RK4, t0, tf, dt)

total_time = (tf - t0)/Un_año
skip = 10


#Euler
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(221, projection = '3d')
ax.scatter(0, 0, 0, c = 'k')
ax.scatter(x_eul[::skip], y_eul[::skip], z_eul[::skip],s=4)
ax.set_title('Euler en 1.96*10⁹')
ax.set_xlabel('X [kpc]')
ax.set_ylabel('Y [kpc]')
ax.set_zlabel('Z [kpc]')
ax2 = fig.add_subplot(222)
ax2.plot(0, 0, 'ko')
ax2.plot(x_eul, y_eul)
ax2.set_title('Euler en 1.96*10⁹')
ax2.set_xlabel('X [kpc]')
ax2.set_ylabel('Y [kpc]')
ax2.legend()

#RK2
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(221, projection = '3d')
ax.scatter(0, 0, 0, c = 'k')
ax.scatter(x_RK2[::skip], y_RK2[::skip], z_RK2[::skip],s=4)
ax.set_title('RK2 en 1.96*10⁹')
ax.set_xlabel('X [kpc]')
ax.set_ylabel('Y [kpc]')
ax.set_zlabel('Z [kpc]')
ax2 = fig.add_subplot(222)
ax2.plot(0, 0, 'ko')
ax2.plot(x_RK2, y_RK2)
ax2.set_title('RK2 en 1.96*10⁹')
ax2.set_xlabel('X [kpc]')
ax2.set_ylabel('Y [kpc]')
ax2.legend()

#RK4
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(221, projection = '3d')
ax.scatter(0, 0, 0, c = 'k')
ax.scatter(x_RK4[::skip], y_RK4[::skip], z_RK4[::skip],s=4)
ax.set_title('RK4 en 1.96*10⁹')
ax.set_xlabel('X [kpc]')
ax.set_ylabel('Y [kpc]')
ax.set_zlabel('Z [kpc]')
ax2 = fig.add_subplot(222)
ax2.plot(0, 0, 'ko')
ax2.plot(x_RK4, y_RK4)
ax2.set_title('RK4 en 1.96*10⁹')
ax2.set_xlabel('X [kpc]')
ax2.set_ylabel('Y [kpc]')
ax2.legend()


#Part 2: A Disk revolving around the Centre of the Galaxy.

#10 Particles
start_time = time.time() 
m, x0, y0, z0, vx0, vy0, vz0 = np.loadtxt('disk10.txt', unpack = True)
bodies_1 = len(m)

t, x_RK4, y_RK4, z_RK4 = integrator(orbita, RK4, t0, tf, dt)
t_10_1 = time.time() - start_time

skip_s = 1
skip_p = 20

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(221, projection = '3d')
ax.scatter(0, 0, 0, c = 'k')
ax2 = fig.add_subplot(222)
ax2.plot(0, 0, 'ko')
for i in range(0, bodies_1, skip_s):
    ax.scatter(x_RK4[:,i][::skip_p], y_RK4[:,i][::skip_p], z_RK4[:,i][::skip_p],s=4)
    ax2.plot(x_RK4[:,i], y_RK4[:,i], '-', linewidth = 0.5, label = 'Star '+str(i))

ax.set_title('RK4 en 1.96*10⁹ (10 particles)')  
ax.set_xlabel('X [kpc]')
ax.set_ylabel('Y [kpc]')
ax.set_zlabel('Z [kpc]')
ax2.set_title('RK4 en 1.96*10⁹ (10 particles)')
ax2.set_xlabel('X [kpc]')
ax2.set_ylabel('X [kpc]')
print('%.2f' % t_10_1, ' s')

#100 Particles
start_time = time.time() 
m, x0, y0, z0, vx0, vy0, vz0 = np.loadtxt('disk100.txt', unpack = True)
bodies_2 = len(m)

t, x_RK4, y_RK4, z_RK4 = integrator(orbita, RK4, t0, tf, dt)
t_100_1 = time.time() - start_time

skip_s = 10
skip_p = 20

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(221, projection = '3d')
ax.scatter(0, 0, 0, c = 'k')
ax2 = fig.add_subplot(222)
ax2.plot(0, 0, 'ko')
for i in range(0, bodies_2, skip_s):
    ax.scatter(x_RK4[:,i][::skip_p], y_RK4[:,i][::skip_p], z_RK4[:,i][::skip_p],s=4)
    ax2.plot(x_RK4[:,i], y_RK4[:,i], '-', linewidth = 0.5)

ax.set_title('RK4 en 1.96*10⁹ (100 particles)')  
ax.set_xlabel('X [kpc]')
ax.set_ylabel('Y [kpc]')
ax.set_zlabel('Z [kpc]')
ax2.set_title('RK4 en 1.96*10⁹ (100 particles)')
ax2.set_xlabel('X [kpc]')
ax2.set_ylabel('X [kpc]')
print('%.2f' % t_100_1, ' s')

#1000 Particles
start_time = time.time() 
m, x0, y0, z0, vx0, vy0, vz0 = np.loadtxt('disk1000.txt', unpack = True)
bodies_3 = len(m)

t, x_RK4, y_RK4, z_RK4 = integrator(orbita, RK4, t0, tf, dt)
t_1000_1 = time.time() - start_time

skip_s = 100
skip_p = 20

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(221, projection = '3d')
ax.scatter(0, 0, 0, c = 'k')
ax2 = fig.add_subplot(222)
ax2.plot(0, 0, 'ko')
for i in range(0, bodies_3, skip_s):
    ax.scatter(x_RK4[:,i][::skip_p], y_RK4[:,i][::skip_p], z_RK4[:,i][::skip_p],s=4)
    ax2.plot(x_RK4[:,i], y_RK4[:,i], '-', linewidth = 0.5)

ax.set_title('RK4 in 1.96*10⁹ (1000 particles)')  
ax.set_xlabel('X [kpc]')
ax.set_ylabel('Y [kpc]')
ax.set_zlabel('Z [kpc]')
ax2.set_title('RK4 in 1.96*10⁹ (1000 particles)')
ax2.set_xlabel('X [kpc]')
ax2.set_ylabel('X [kpc]')
print('%.2f' % t_1000_1, ' s')

#Relation between the computing time and the particle number
N = np.array([bodies_1, bodies_2, bodies_3])
t_1 = np.array([t_10_1, t_100_1, t_1000_1])

def degree(x, a, b, n):
    return a*x**n + b

a_1, b_1, n_1 = curve_fit(degree, N, t_1)[0]
fit_1 = degree(N, a_1, b_1, n_1)
print('n =', n_1)

fig = plt.figure(figsize=(10,6))
plt.plot(N, t_1, 'ko', label = 'Data')
plt.plot(N, fit_1, 'r-', label = 'Fit')
plt.legend()
plt.title('Computational time vs Number of particles simulated')
plt.xlabel('N')
plt.ylabel('t [s]')

#Part 3: A Disk revolving around the Centre of the Galaxy with “self gravity”.

#The orbital function is defined, now with interaction between particles.
def Nbody(t, x, y, z, vx, vy, vz):
    r = np.sqrt(x**2 + y**2 + z**2 + soft**2)
    m_in = enclosed_mass(rho_0, R_s, r)
    
    xi = np.full([bodies, bodies], x).transpose()
    yi = np.full([bodies, bodies], y).transpose()
    zi = np.full([bodies, bodies], z).transpose()
        
    ex = [k != r for k in r]
    
    dist = np.sqrt((x-xi*ex)**2 + (y-yi*ex)**2 + (z-zi*ex)**2 + soft**2)

    ax = - G*(sum((x - xi*ex)*m*ex/dist**3) + x*m_in/r**3)*(Unit_velocity/Unit_length)**2
    ay = - G*(sum((y - yi*ex)*m*ex/dist**3) + y*m_in/r**3)*(Unit_velocity/Unit_length)**2
    az = - G*(sum((z - zi*ex)*m*ex/dist**3) + z*m_in/r**3)*(Unit_velocity/Unit_length)**2
    return vx, vy, vz, ax, ay, az

#10 Particles
start_time = time.time() 
m, x0, y0, z0, vx0, vy0, vz0 = np.loadtxt('disk10.txt', unpack = True)    # 10 stars
bodies = len(m)
soft = 1

t, x_RK4, y_RK4, z_RK4 = integrator(Nbody, RK4, t0, tf, dt)
t_10_2 = time.time() - start_time

skip_s = 1
skip_p = 15

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(221, projection = '3d')
ax.scatter(0, 0, 0, c = 'k')
ax2 = fig.add_subplot(222)
ax2.plot(0, 0, 'ko')
for i in range(0, bodies_1, skip_s):
    ax.scatter(x_RK4[:,i][::skip_p], y_RK4[:,i][::skip_p], z_RK4[:,i][::skip_p],s=4)
    ax2.plot(x_RK4[:,i], y_RK4[:,i], '-', linewidth = 0.5, label = 'Star '+str(i))

ax.set_title('RK4 en 1.96*10⁹ (10 particles)')  
ax.set_xlabel('X [kpc]')
ax.set_ylabel('Y [kpc]')
ax.set_zlabel('Z [kpc]')
ax2.set_title('RK4 en 1.96*10⁹ (10 particles)')
ax2.set_xlabel('X [kpc]')
ax2.set_ylabel('X [kpc]')
print('%.2f' % t_10_2, ' s')

#100 Particles
start_time = time.time() 
m, x0, y0, z0, vx0, vy0, vz0 = np.loadtxt('disk100.txt', unpack = True)    # 10 stars
bodies = len(m)
soft = 1

t, x_RK4, y_RK4, z_RK4 = integrator(Nbody, RK4, t0, tf, dt)
t_100_2 = time.time() - start_time

skip_s = 15
skip_p = 10

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(221, projection = '3d')
ax.scatter(0, 0, 0, c = 'k')
ax2 = fig.add_subplot(222)
ax2.plot(0, 0, 'ko')
for i in range(0, bodies_2, skip_s):
    ax.scatter(x_RK4[:,i][::skip_p], y_RK4[:,i][::skip_p], z_RK4[:,i][::skip_p],s=4)
    ax2.plot(x_RK4[:,i], y_RK4[:,i], '-', linewidth = 0.5)

ax.set_title('RK4 en 1.96*10⁹ (100 particles)')  
ax.set_xlabel('X [kpc]')
ax.set_ylabel('Y [kpc]')
ax.set_zlabel('Z [kpc]')
ax2.set_title('RK4 en 1.96*10⁹ (100 particles)')
ax2.set_xlabel('X [kpc]')
ax2.set_ylabel('X [kpc]')
print('%.2f' % t_100_2, ' s')

#1000 Particles
start_time = time.time() 
m, x0, y0, z0, vx0, vy0, vz0 = np.loadtxt('disk1000.txt', unpack = True)    # 10 stars
bodies = len(m)
soft = 1

t, x_RK4, y_RK4, z_RK4 = integrator(Nbody, RK4, t0, tf, dt)
t_1000_2 = time.time() - start_time

skip_s = 100
skip_p = 20

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(221, projection = '3d')
ax.scatter(0, 0, 0, c = 'k')
ax2 = fig.add_subplot(222)
ax2.plot(0, 0, 'ko')
for i in range(0, bodies_3, skip_s):
    ax.scatter(x_RK4[:,i][::skip_p], y_RK4[:,i][::skip_p], z_RK4[:,i][::skip_p],s=4)
    ax2.plot(x_RK4[:,i], y_RK4[:,i], '-', linewidth = 0.5)

ax.set_title('RK4 in 1.96*10⁹ (1000 particles)')  
ax.set_xlabel('X [kpc]')
ax.set_ylabel('Y [kpc]')
ax.set_zlabel('Z [kpc]')
ax2.set_title('RK4 in 1.96*10⁹ (1000 particles)')
ax2.set_xlabel('X [kpc]')
ax2.set_ylabel('X [kpc]')
print('%.2f' % t_1000_1, ' s')

#Relation between the computing time and the particle number (with self gravity)
t_2 = np.array([t_10_2, t_100_2, t_1000_2])

a_2, b_2, n_2 = curve_fit(degree, N, t_2)[0]
fit_2 = degree(N, a_2, b_2, n_2)
print('n =', n_2)

fig = plt.figure(figsize=(10,6))
plt.plot(N, t_2, 'ko', label = 'Data')
plt.plot(N, fit_2, 'r-', label = 'Fit')
plt.legend()
plt.title('Computational time vs Number of particles simulated')
plt.xlabel('N')
plt.ylabel('t [s]')
