import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

rng = np.random.default_rng(1234)


##Added Hill Non-Linearity to CaMK2 next step is to add the corect dynamics
##Added distance based attenuation of signal in the code , requires sanity check

# -------- Parameters --------
params = {
    "n_syn": 30,

    # Weight parameters
    "w_init": 0.4,
    "w_min": 0.0,
    "w_max": 4.0,
    "eta": 0.22,
    "sigma_w": 0.02,

    # Spine Ca2+ (NMDAR)
    "alpha_nmda": 1.0,
    "tau_ca_spine": 0.2,
    "sigma_ca_spine": 0.05,
    "mg_block": 0.062,

    # Voltage
    "v_rest": -70,
    "v_peak": -15,

    # IP3 & ER
    "alpha_ip3_post": 3.0,
    "tau_ip3": 6.0,
    "ca_store_max": 8.0, #18/12/25
    "alpha_release": 3.5,
    "tau_store_refill": 40.0,

    # Dendritic Ca2+
    "tau_ca_dend": 3.0,
    "sigma_dend": 0.04,

    # CaMKII dynamics (delayed & stochastic)
    "alpha_camkii": 1.5, #18/12/25
    "tau_camkii": 8.2, #17/12/25
    "camkii_theta": 0.6, #18/12/25
    "camkii_delay": 15.0,
    "sigma_camkii": 0.08,

    # Eligibility trace (decaying)
    "tau_elig": 20.0,

    # Synaptic tag (phosphatase / PP1)
    "alpha_pp1": 1.0,
    "tau_pp1": 30.0,
    "pp1_baseline": 1.0,
    "sigma_pp1": 0.03,

    # BTSP timing kernel (Gaussian)
    "tau_btsp": 2.5,
    
    # Distance-based attenuation
    "lambda_d": 150.0,  # Space constant in µm
}
#Switch for Camkii activation
def camkii_activation(ca, theta, n=4):
    return ca**n / (ca**n + theta**n)


# ===== Optimized Simulation Function =====
def run_btsp(
    offset,
    active_synapses,
    T=120.0,  # Reduced from 120s
    dt=0.01,  # Increased from 0.01
    return_traces=False
):
    """
    offset: (t_post - t_pre), timing difference
    active_synapses: list of syn indices with presynaptic input
    """
    time = np.arange(0, T, dt)
    n = len(time)
    n_syn = params["n_syn"]
    
    # Create boolean mask for active synapses (vectorized operations)
    is_active = np.zeros(n_syn, dtype=bool)
    is_active[active_synapses] = True

    #18/12/25
    # FIXED: Synapse distances
    syn_distances = np.linspace(1, 300, n_syn)  # Evenly spaced from 1 to 300 µm
    # Alternatively, use random distances:
    #syn_distances = np.sort(rng.uniform(1, 300, n_syn))

    # Local states (vectorized)
    Ca_sp = np.zeros((n_syn, n))
    elig = np.zeros((n_syn, n))
    PP1 = np.ones((n_syn, n)) * params["pp1_baseline"]
    w = np.ones((n_syn, n)) * params["w_init"]

    # Global states
    IP3 = np.zeros(n)
    Ca_store = np.ones(n) * params["ca_store_max"]
    Ca_dend = np.zeros(n)
    CaMKII = np.zeros(n)

    # Event times
    t_pre = 10.0
    t_post = 10.0 + offset

    # Precompute BTSP timing kernel factor
    timing_kernel = np.exp(-(offset**2) / (2 * params["tau_btsp"]**2))

    # Preallocate random numbers for noise (much faster)
    noise_ca_spine = rng.normal(0, 1, (n_syn, n)) * params["sigma_ca_spine"]
    noise_pp1 = rng.normal(0, 1, (n_syn, n)) * params["sigma_pp1"]
    noise_w = rng.normal(0, 1, (n_syn, n)) * params["sigma_w"]
    noise_dend = rng.normal(0, 1, n) * params["sigma_dend"]
    noise_camkii = rng.normal(0, 1, n) * params["sigma_camkii"]

    camkii_started = False
    start_time = 0.0

    for i in range(n - 1):
        t = time[i]

        # Postsynaptic plateau
        post = 1.0 if abs(t - t_post) < dt else 0.0
        
        # FIXED: Distance-based voltage attenuation
        #18/12/25
        V_soma = params["v_rest"] + post * (params["v_peak"] - params["v_rest"])
        attenuation = np.exp(-syn_distances / params["lambda_d"])*4 #19/12/25 Increased the degree of attenutation
        #V_spine = params["v_rest"] + post * (params["v_peak"] - params["v_rest"]) * attenuation
        V_spine = params["v_rest"] + (V_soma - params["v_rest"]) * attenuation
        
        # NMDA voltage dependence (vectorized for all synapses)
        nmda_v = 1.0 / (1.0 + params["mg_block"] * np.exp(-0.062 * V_spine)) #18/12/25

        # ---- Global signals ----
        dIP3 = (-IP3[i] / params["tau_ip3"] + params["alpha_ip3_post"] * post) * dt 
        release = params["alpha_release"] * IP3[i] * Ca_store[i]#np.exp(params['tau_ca_spine']/(1+Ca_dend[i])) #19/12/25
        refill = (params["ca_store_max"] - Ca_store[i]) / params["tau_store_refill"]
        dCa_store = (refill - release) * dt
        dCa_store += noise_dend[i] * np.sqrt(dt)

        dCa_dend = (-Ca_dend[i] / params["tau_ca_dend"] + release) * dt
        dCa_dend += noise_dend[i] * np.sqrt(dt)

        # Delayed CaMKII (fixed bug: >= instead of >)
        delay_steps = int(params["camkii_delay"] / dt)
        drive = Ca_dend[i - delay_steps] if i >= delay_steps else 0.0
        dCaMKII = (
            -CaMKII[i] / params["tau_camkii"]
            + params["alpha_camkii"] * camkii_activation(drive, params["camkii_theta"]) #18/12/15 #Hill non linearity previous code max(drive - params["camkii_theta"], 0)

        ) * dt #The rate of Calmodulin concentration directly scales with dendritic Calcium
        dCaMKII += noise_camkii[i] * np.sqrt(dt*drive) #18/12/25 # Logic being if dendritic Ca goes down than less binding for Calmodulin(Rate goes down)

        # ---- Vectorized synapse updates ----
        pre = 1.0 if abs(t - t_pre) < dt else 0.0
        
        #
        dCa = -Ca_sp[:, i] / params["tau_ca_spine"] * dt
        dCa[is_active] += params["alpha_nmda"] * pre * nmda_v[is_active] * dt #18/12/25
        dCa += noise_ca_spine[:, i] * np.sqrt(dt)
        Ca_sp[:, i + 1] = np.maximum(Ca_sp[:, i] + dCa, 0)

        # Eligibility (vectorized)
        delig = -elig[:, i] / params["tau_elig"] * dt
        delig[is_active] += (timing_kernel * Ca_sp[is_active, i]) / params["tau_elig"] * dt
        elig[:, i + 1] = np.maximum(elig[:, i] + delig, 0)

        # Synaptic tag (PP1 inhibition) (vectorized)
        dPP1 = (params["pp1_baseline"] - PP1[:, i]) / params["tau_pp1"] * dt
        dPP1[is_active] -= params["alpha_pp1"] * Ca_sp[is_active, i] * dt
        dPP1 += noise_pp1[:, i] * np.sqrt(dt)
        PP1[:, i + 1] = np.maximum(PP1[:, i] + dPP1, 0)

        # Weight update (vectorized with masking)
        dw = np.zeros(n_syn)
        
        # Check conditions for plasticity 
        plasticity_mask = (
            is_active & 
            (CaMKII[i] > 0) & 
            (elig[:, i] > 0) & 
            (PP1[:, i] < 0.75) #18/12/25
        )
        
        if np.any(plasticity_mask):
            if not camkii_started:
                camkii_started = True
                start_time = t
            
            timing_gain = np.exp(-(t - start_time) / 30.0)
            current_W=w[plasticity_mask,i]
            dw[plasticity_mask] = (
                params["eta"]
                * timing_kernel
                * timing_gain
                * CaMKII[i]
                *(params["w_max"]-current_W)#18/12/25
                * elig[plasticity_mask, i]
                * dt
            )
            
            dw += noise_w[:, i] *np.sqrt(dt) * plasticity_mask#-(params["w_max"]-w[:,i])

        

        w[:, i + 1] = np.clip(w[:, i] + dw, params["w_min"], params["w_max"])

        # Update global states
        IP3[i + 1] = max(IP3[i] + dIP3, 0)
        Ca_store[i + 1] = np.clip(Ca_store[i] + dCa_store, 0, params["ca_store_max"])
        Ca_dend[i + 1] = max(Ca_dend[i] + dCa_dend, 0)
        CaMKII[i + 1] = max(CaMKII[i] + dCaMKII, 0)

    if return_traces:
        return time, w, Ca_sp, PP1, elig, CaMKII, Ca_dend, IP3, Ca_store, syn_distances,nmda_v #18/12/25

    return w[:, -1] - w[:, 0], syn_distances #18/12/25


# ===== Plotting: Timing Curve =====
print("Running timing curve simulation...")
active_syn = [3, 7, 12] #18/12/25
offsets = np.linspace(-6, 6, 13)
n_trials = 20  # Reduced from 30

mean_dw, sem_dw = [], []
for idx, off in enumerate(offsets):
    print(f"Offset {idx+1}/{len(offsets)}: {off:.2f}s")
    trials = []
    for _ in range(n_trials):
        dw, _ = run_btsp(off, active_syn)
        trials.append(dw[active_syn].mean())
    trials = np.array(trials)
    mean_dw.append(trials.mean())
    sem_dw.append(trials.std(ddof=1) / np.sqrt(n_trials))

plt.figure(figsize=(6, 4))
plt.errorbar(offsets, mean_dw, yerr=sem_dw, fmt='o-', capsize=4)
plt.axvline(0, color='gray', linestyle='--')
plt.xlabel("Post − Pre timing (s)")
plt.ylabel("Δw (mean ± SEM)")
plt.title("BTSP Timing Curve (Optimized)")
plt.tight_layout()
plt.show()



# ===== NEW: Distance vs Weight Change Analysis =====
print("Running distance analysis...")
dw_all, distances = run_btsp(2.0, list(range(params["n_syn"])))

plt.figure(figsize=(6, 4))
plt.scatter(distances, dw_all, alpha=0.7)
plt.xlabel("Synapse Distance (µm)")
plt.ylabel("Δw")
plt.title("Weight Change vs Distance")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ===== Plotting: Active vs Inactive Traces =====
print("Running trace simulation...")
time, w_tr, Ca_sp, PP1, elig, CaMKII, Ca_dend, IP3, Ca_store, distances,nmda = run_btsp(2.0, active_syn, return_traces=True)
inactive_syn = [0]


fig, ax = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
ax[0].plot(time, Ca_sp[active_syn[0]], label=f"Active (d={distances[active_syn[0]]:.1f}µm)", linewidth=2)
#ax[0].plot(time, Ca_sp[inactive_syn[0]], color='gray', alpha=0.7, label=f"Inactive (d={distances[inactive_syn[0]]:.1f}µm)")
ax[0].set_ylabel("Spine Ca²⁺")
ax[0].legend()

ax[1].plot(time, PP1[active_syn[0]], label="Active", linewidth=2)
ax[1].plot(time, PP1[inactive_syn[0]], color='gray', alpha=0.7, label="Inactive")
ax[1].set_ylabel("PP1")
ax[1].legend()

ax[2].plot(time, elig[active_syn[0]], label="Active", linewidth=2)
ax[2].plot(time, elig[inactive_syn[0]], color='gray', alpha=0.7, label="Inactive")
ax[2].set_ylabel("Eligibility")
ax[2].set_xlabel("Time (s)")
ax[2].legend()

fig1, ax1 = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
ax1[0].plot(time, CaMKII, label="CaMKII", linewidth=2)
ax1[0].plot(time, Ca_dend, color='gray', label="Dendritic Ca", linewidth=2)
ax1[0].set_ylabel("Dendritic Ca²⁺ vs CaMKII")
ax1[0].legend()

ax1[1].plot(time, IP3, label="IP3", linewidth=2)
ax1[1].set_ylabel("IP3")
ax1[1].legend()

ax1[2].plot(time, Ca_store, label="Ca Store", linewidth=2)
ax1[2].set_ylabel("Ca from internal stores")
ax1[2].set_xlabel("Time (s)")
ax1[2].legend()

plt.tight_layout()
plt.show()
print("Done!")
