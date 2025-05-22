import numpy as np
import matplotlib.pyplot as plt
import common
import timeit

INTEGRATOR = common.leapfrog
INITIAL_CONDITION = "pyth-3-body"
TF = 1000.0 * 365.24  # 1000 giorni ≈ ~2.7 anni
DT = 1.0
OUTPUT_INTERVAL = 0.01
NUM_STEPS = int(TF / DT)
TOLERANCE = 1e-13


def compute_velocity_distribution(sol_v, time_index=-1):
    v_mag = np.linalg.norm(sol_v[time_index], axis=1)
    plt.hist(v_mag, bins=30, density=True, color="skyblue", edgecolor="black")
    plt.title("Distribuzione delle velocità (tempo finale)")
    plt.xlabel("Velocità (AU/day)")
    plt.ylabel("Densità")
    plt.grid(True)
    plt.show()


def compute_velocity_autocorrelation(sol_v, particle_idx=0):
    v = sol_v[:, particle_idx, 0]  # componente x della velocità
    v -= np.mean(v)
    autocorr = np.correlate(v, v, mode='full') / np.max(np.correlate(v, v, mode='full'))
    mid = len(autocorr) // 2
    plt.plot(autocorr[mid:])
    plt.title(f"Autocorrelazione velocità - Particella {particle_idx}")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelazione")
    plt.grid(True)
    plt.show()


def compute_trajectory_divergence(system, dt, num_steps):
    # Crea una copia leggermente perturbata
    perturbation = 1e-5
    system2 = system.copy()
    system2.x[0, 0] += perturbation  # perturbazione posizione corpo 0

    a = np.zeros((system.num_particles, 3))
    a2 = np.zeros_like(a)

    d_list = []

    for _ in range(num_steps):
        INTEGRATOR(a, system, dt)
        INTEGRATOR(a2, system2, dt)
        d = np.linalg.norm(system.x - system2.x)
        d_list.append(d)

    plt.plot(np.arange(num_steps) * dt, d_list)
    plt.yscale("log")
    plt.title("Divergenza delle traiettorie (caos)")
    plt.xlabel("Tempo (giorni)")
    plt.ylabel("||x1 - x2|| (AU)")
    plt.grid(True)
    plt.show()


def main():
    system, labels, colors, legend = common.get_initial_conditions(INITIAL_CONDITION)

    a = np.zeros((system.num_particles, 3))
    sol_size = int(TF // OUTPUT_INTERVAL + 2)
    sol_x = np.zeros((sol_size, system.num_particles, 3))
    sol_v = np.zeros((sol_size, system.num_particles, 3))
    sol_t = np.zeros(sol_size)

    sol_x[0] = system.x
    sol_v[0] = system.v
    sol_t[0] = 0.0

    output_count = 1
    next_output_time = output_count * OUTPUT_INTERVAL

    start = timeit.default_timer()

    for i in range(NUM_STEPS):
        INTEGRATOR(a, system, DT)
        current_time = i * DT

        if current_time >= next_output_time:
            sol_x[output_count] = system.x
            sol_v[output_count] = system.v
            sol_t[output_count] = current_time
            output_count += 1
            next_output_time = output_count * OUTPUT_INTERVAL

    end = timeit.default_timer()

    sol_x = sol_x[:output_count]
    sol_v = sol_v[:output_count]
    sol_t = sol_t[:output_count]

    print(f"Simulazione completata in {end - start:.2f} s")

    # Analisi
    compute_velocity_distribution(sol_v)
    compute_velocity_autocorrelation(sol_v, particle_idx=0)
    compute_trajectory_divergence(system=system, dt=DT, num_steps=500)


if __name__ == "__main__":
    main()
