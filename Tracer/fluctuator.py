import numpy as np

def gust_simple(wind,tke,rng,cf=1):
    sigma = np.sqrt(tke * (2.0/3.0))            # std of turbulent kinetic energy with correction factor cf
    eta = rng.normal(0.0, 1.0, size=3)          # draw three random numbers     
    gust = sigma * eta                          # gaussian distributed gust components u',v',w'
    wind_new = wind + gust * cf                 # effective wind u,v,w = mean u,v,w + gust u',v',w'
    return wind_new

def gust_OU(wind, tke, gust_old, rng, dt=0.01, Tg=0.1, cf=1):                   # Ornstein-Uhlenbeck inspired model
    sigma = np.sqrt(tke * (2.0/3.0))                                            # std of turbulent kinetic energy
    eta = rng.normal(0.0, 1.0, size=3)                                          # draw three random numbers
    gust_new = gust_old*(1.0 - dt/Tg) + np.sqrt(2.0*sigma**2*dt/Tg) * eta       # new gust = decaying previous gust term + new random component
    wind_new = wind + gust_new * cf                                             # effective wind u,v,w = mean u,v,w + gust u',v',w'
    return wind_new, gust_new

def gust_Langevin(wind, tke, epsilon, gust_old, rng: np.random.Generator, dt=0.01, C0 = 2.1, cf=1):      # Langevin implementation
    gamma = (3.0/4.0) * C0 * (epsilon/(tke))                                        # decay term based on engineering factor C0, tke, and epsilon
    eta = rng.normal(0.0, 1.0, size=3)                                              # random number draw 
    gust_new = gust_old * (1.0 - gamma * dt) + np.sqrt(C0 * epsilon * dt) * eta     
    wind_new = wind + gust_new * cf                  
    return wind_new, gust_new