import time
import datetime

from neural_networks.data import project_data_root
import numpy as np
import original

current_datetime = datetime.datetime.now().isoformat()
output_dir = project_data_root / 'figures' / 'dho_r_runtime' / current_datetime
output_dir.mkdir(exist_ok=True, parents=True)

repeat_samples = 4
original_csv_out = open(output_dir.joinpath('original.csv'), "w")
original_csv_out.write("r,time,setup_time\n")

iterations = 500


def sample_original(r: int):
    print(f"Running original with r={r}")
    start_setup = time.time()
    for _ in range(repeat_samples):
        sys = original.dho(1.0, 1.0, 1.0, r)

    setup_time = (time.time() - start_setup) / repeat_samples

    start = time.time()
    for _ in range(repeat_samples):
        sys(iterations)

    time_elapsed = (time.time() - start) / repeat_samples
    original_csv_out.write(f"{r},{time_elapsed},{setup_time}\n")
    original_csv_out.flush()

    return time_elapsed


r_values = np.arange(10, 20, 1)
original_times = np.array([sample_original(int(n)) for n in r_values])
original_csv_out.close()
