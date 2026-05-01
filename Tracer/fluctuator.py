from windfield import WindField 
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

def gust_Langevin(wind, tke, epsilon, gust_old, rng, dt=0.01, C0 = 2.1, cf=1):      # Langevin implementation
    gamma = (3.0/4.0) * C0 * (epsilon/(tke))                                        # decay term based on engineering factor C0, tke, and epsilon
    eta = rng.normal(0.0, 1.0, size=3)                                              # random number draw 
    gust_new = gust_old * (1.0 - gamma * dt) + np.sqrt(C0 * epsilon * dt) * eta     
    wind_new = wind + gust_new *cf                  
    return wind_new, gust_new

# --- Usage Example ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sim_field = WindField(profile="log", direction=0, U_ref=6, z_ref=10, z0=0.03)
    point = np.array([1, 1, 20])

    wind = np.array(sim_field.get_velocity_at(x=point[0], y=point[1], z=point[2]), dtype=float)
    tke = sim_field.get_tke_at(x=point[0], y=point[1], z=point[2], dtype=float)
    epsilon = sim_field.get_epsilon_at(x=point[0], y=point[1], z=point[2], dtype=float)

    rng = np.random.default_rng(42)

    n_steps = 1000
    time = np.linspace(0, n_steps*0.01, n_steps)

    # storage arrays
    simple_hist = np.zeros((n_steps, 3))
    ou_hist = np.zeros((n_steps, 3))
    lang_hist = np.zeros((n_steps, 3))

    gust_state_ou = np.zeros(3)
    gust_state_lang = np.zeros(3)

    for i in range(n_steps):

        simple_hist[i] = gust_simple(wind, tke, rng)

        ou_wind, gust_state_ou = gust_OU(wind, tke, gust_state_ou, rng)
        ou_hist[i] = ou_wind

        lang_wind, gust_state_lang = gust_Langevin(wind, tke, epsilon, gust_state_lang, rng, C0=2.1)
        lang_hist[i] = lang_wind

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    labels = ['u-component', 'v-component', 'w-component']

    for j in range(3):
        axs[j].plot(time, simple_hist[:, j], label='Simple Gaussian')
        axs[j].plot(time, ou_hist[:, j], label='OU gust')
        axs[j].plot(time, lang_hist[:, j], label='Langevin gust')
        axs[j].axhline(wind[j], linestyle='--')
        axs[j].set_ylabel(labels[j])
        axs[j].legend()
        axs[j].grid(True)

    axs[2].set_xlabel('Time [s]')
    plt.suptitle('Comparison of Simple Gaussian Gust vs Ornstein-Uhlenbeck Gust')
    plt.tight_layout()
    plt.show()

# %%
