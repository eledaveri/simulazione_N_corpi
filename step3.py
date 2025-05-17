import common
import numpy as np
import timeit

INITIAL_CONDITION = "solar_system"
OUTPUT_INTERVAL = 0.1 * 365.24
# Default units is AU, days, and M_sun
TF = 200.0 * 365.24  # years to days
DT = 1.0
OUTPUT_INTERVAL = 0.1 * 365.24  # years to days
NUM_STEPS = int(TF / DT)

def main () -> None:
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
    
    #lancio la simulazione
    common.print_simulation_info_fixed_step_size(system=system, tf=TF, dt=DT, num_steps=NUM_STEPS, output_interval=OUTPUT_INTERVAL, sol_size=sol_size)
    next_output_time = output_count * OUTPUT_INTERVAL
    start = timeit.default_timer()
    for i in range(NUM_STEPS):
        common.euler(a, system, DT)
        current_time = i * DT
        if current_time >= next_output_time:
            sol_x[output_count] = system.x
            sol_v[output_count] = system.v
            sol_t[output_count] = current_time

            output_count += 1
            next_output_time = output_count * OUTPUT_INTERVAL

            print(f"Current time: {current_time:.2f} days", end="\r")

    sol_x = sol_x[:output_count]
    sol_v = sol_v[:output_count]
    sol_t = sol_t[:output_count]

    end = timeit.default_timer()

    print()
    print(f"Done! Runtime: {end - start:.3g} seconds, Solution size: {output_count}")
    common.plot_trajectory(
        sol_x=sol_x,
        labels=labels,
        colors=colors,
        legend=legend,
    )
    
if __name__ == "__main__":
    main()