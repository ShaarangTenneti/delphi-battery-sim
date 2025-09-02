from dataclasses import dataclass
from collections import namedtuple 

@dataclass(frozen=True)
class BatteryParams:
    capacity: float = 2.0 #mWh
    charge_efficiency: float = 0.96 
    discharge_efficiency: float = 0.96
    max_charge_rate: float = 1.04 #mW
    max_discharge_rate: float = 0.96 #mW

@dataclass
class DispatchParams:
    commitment_intervals: int = 1   
    horizon_intervals:    int = 144
    time_period_hours:    float = 5/60

from dataclasses import dataclass
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary, PULP_CBC_CMD

def solve_battery_lp(
    prices,
    battery_params: BatteryParams,
    initial_energy: float = 0.0,
    time_period_hours: float = 5 / 60,
    committed_decisions={"charge": [], "discharge": []}
):
    n = len(prices)
    model = LpProblem("Battery_Optimisation", LpMaximize)

    # Decision variables for each period (0 to n-1)
    charge = []
    discharge = []
    mode = []
    for t in range(n):
        charge_var = LpVariable(f"charge_{t}", 0, battery_params.max_charge_rate)
        discharge_var = LpVariable(f"discharge_{t}", 0, battery_params.max_discharge_rate)
        mode_var = LpVariable(f"mode_{t}", cat=LpBinary)
        charge.append(charge_var)
        discharge.append(discharge_var)
        mode.append(mode_var)

    # --- pin the locked ones ---
    L = len(committed_decisions["charge"])
    if L != len(committed_decisions["discharge"]) or L > n:
        raise ValueError("charge/discharge lengths must match and be ≤ n")
    for t in range(L):
        charge[t].lowBound = charge[t].upBound = committed_decisions["charge"][t]
        discharge[t].lowBound = discharge[t].upBound = committed_decisions["discharge"][t]

    # Define energy variables for t = 0, 1, ..., n (n+1 variables)
    # These automatically enforce SoC stays between 0 and battery capacity.
    energy = []
    for t in range(n + 1):
        energy_var = LpVariable(f"energy_{t}", 0, battery_params.capacity)
        energy.append(energy_var)

    # Set the initial SoC
    model += energy[0] == initial_energy

    # Update SoC for each period (from t to t+1)
    for t in range(n):
        energy_to_battery = charge[t] * battery_params.charge_efficiency * time_period_hours
        energy_from_battery = discharge[t] * (1 / battery_params.discharge_efficiency) * time_period_hours
        model += energy[t+1] == energy[t] + energy_to_battery - energy_from_battery

    # Objective: profit = revenue from export - cost of import
    objective_terms = []
    for t in range(n):
        revenue = discharge[t] * prices[t] * time_period_hours
        cost = charge[t] * prices[t] * time_period_hours
        objective_terms.append(revenue - cost)
    model += lpSum(objective_terms)

    # Prevent simultaneous charging and discharging
    for t in range(n):
        model += charge[t] <= battery_params.max_charge_rate * mode[t]
        model += discharge[t] <= battery_params.max_discharge_rate * (1 - mode[t])

    # (No terminal constraint forcing energy[n] to 0; it’s free to be anywhere within [0, capacity])
    
    # Solve the problem
    solver = PULP_CBC_CMD(msg=0, threads=4, timeLimit=30, gapRel=0.01)
    model.solve(solver)


    return {
        "SoC": [e.varValue for e in energy],       # SoC over time (n+1 values)
        "charge": [c.varValue for c in charge],         # kW imported per period
        "discharge": [d.varValue for d in discharge],   # kW exported per period
        "total_profit": model.objective.value()      # total profit ($)
    }


# --------------------------------------------------------------------
# Fully Optimal Solver
# --------------------------------------------------------------------
def solve_fully_optimal(df, battery_params, time_period_hours):
    """Solves the battery optimization problem over the entire horizon."""
    start_time = time.time()
    print("Solving fully optimal model...")
   
    # Extract prices and timestamps
    prices = df["0"].values
    timestamps = list(df.index)
   
    # Solve the optimization problem
    solution = solve_battery_lp(
        prices=prices,
        battery_params=battery_params,
        initial_energy=0,
        time_period_hours=time_period_hours
    )
   
    solve_time = time.time() - start_time
    print(f"Solved in {solve_time:.2f}s")

    soc = solution["SoC"]
    charge = np.array(solution["charge"])
    discharge = np.array(solution["discharge"])

    # Profit based on grid-level power
    profit_step = prices * (discharge - charge) * time_period_hours

    return pd.DataFrame({
        "datetime": timestamps,
        "price_actual": prices,
        "charge_mw": charge,  # grid-side
        "discharge_mw": discharge,  # grid-side
        "soc_open": soc[:-1],
        "soc_close": soc[1:],
        "profit_step": profit_step,
        "solve_time": solve_time
    })

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, PULP_CBC_CMD
from tqdm import tqdm
import time

# --------------------------------------------------------------------
# Rolling-Horizon Solver
# --------------------------------------------------------------------
def solve_rolling_horizon(model_df, battery_params, dispatch_params: DispatchParams, perfect_results, log_file=None):

    start_time = time.time()
    print("Solving rolling horizon")
    
    N = len(model_df)
    results = []
    current_soc = 0.0  # Initial SoC
    end_of_sim_period_penalty = 0

    L = dispatch_params.commitment_intervals
    charge_queue     = [0.0] * L
    discharge_queue  = [0.0] * L

    for i in range(N):
        t_iter_start = time.time()


        #battery should end with 0 soc
        if (i + dispatch_params.horizon_intervals > N):
            end_of_sim_period_penalty = i + dispatch_params.horizon_intervals - N
            
        forecast_cols = [str(k) for k in range(0, dispatch_params.horizon_intervals + 1 - end_of_sim_period_penalty)]
        forecast_prices = model_df[forecast_cols].iloc[i].values

        # --- safety clip to avoid rounding over-fill -----------------
        trim_committed_charge_if_full(
            soc_now=current_soc,
            charge_queue=charge_queue,
            battery_params=battery_params,
            time_period_hours=dispatch_params.time_period_hours,
        )

        committed_dict = {
            "charge":     charge_queue,
            "discharge":  discharge_queue}
        
        #For the current period
        solution = solve_battery_lp(
                    prices=forecast_prices,
                    battery_params=battery_params,
                    initial_energy=current_soc,
                    time_period_hours=dispatch_params.time_period_hours,
                    committed_decisions=committed_dict
                )
        
        charge = solution["charge"][0]
        discharge = solution["discharge"][0]
        soc_open = solution["SoC"][0]
        soc_close = solution["SoC"][1]
        price_actual = model_df["0"].iloc[i]
        profit_step = price_actual * (discharge - charge) * dispatch_params.time_period_hours
        solve_time = time.time() - t_iter_start

        current_soc = soc_close
        
        results.append({
                "datetime": model_df.index[i],
                "price_actual": price_actual,
                "charge_mw": charge,
                "discharge_mw": discharge,
                "soc_open": soc_open,
                "soc_close": soc_close,
                "profit_step": profit_step,
                "solve_time": solve_time
            })
        
        # pop the commitment that just executed
        charge_queue.pop(0)
        discharge_queue.pop(0)
        
        # push the LP s decision for t+L onto the queue (if it exists)
        if len(solution["charge"]) > L:
            charge_queue.append(    solution["charge"][L]    )
            discharge_queue.append( solution["discharge"][L] )

    results_df = pd.DataFrame(results)
    total_time = time.time() - start_time
    total_model_profit = results_df["profit_step"].sum()
    optimal_profit = perfect_results["profit_step"].sum()
    percentage_profit_of_optimal = (total_model_profit / optimal_profit) * 100  # Multiply by 100 here

    period_log = (
        f"\nPeriod: {results_df['datetime'].min()} - {results_df['datetime'].max()}\n"
        f"Model Profit:        ${total_model_profit:,.2f}\n"
        f"Optimal Profit:      ${optimal_profit:,.2f}\n"
        f"Percentage of Optimal: {percentage_profit_of_optimal:.2f}%\n"
        f"Done. Total time: {total_time:.2f}s.\n"
    )

    log(period_log, logfile=log_file)

    return results_df


def trim_committed_charge_if_full(
    soc_now: float,
    charge_queue: list[float],
    battery_params: BatteryParams,
    time_period_hours: float,
    eps: float = 1e-6,
) -> None:
    """
    When the batterys state-of-charge is near its upper limit, a committed
    charge can push it just beyond capacity due to floating-point rounding.
    That tiny overshoot can cause the LP to become infeasible. We only need
    to trim charged amounts when SoC is close to full; the solver handles
    near-zero SoC rounding gracefully, so no special handling is required
    for discharging.
    """

    if not charge_queue:
        return

    # how much energy (kWh) that queued charge would add this step
    added_kwh = charge_queue[0] * battery_params.charge_efficiency * time_period_hours

    # if it doesn’t even reach the cap, bail out
    if soc_now + added_kwh < battery_params.capacity:
        return

    # compute the tiny overflow beyond cap
    surplus_kwh = soc_now + added_kwh - battery_params.capacity

    # if it’s within eps of cap, fix it
    if surplus_kwh <= eps:
        # convert that surplus back to kW for this period
        delta_kw = surplus_kwh / (battery_params.charge_efficiency * time_period_hours)
        old = charge_queue[0]
        new = old - delta_kw

        # only trim if it actually changes
        if abs(old - new) > 1e-9:
            # print(
            #     f"⚠️  trimming committed charge "
            #     f"from {old:.6f} kW to {new:.6f} kW (avoid over-fill)"
            # )
            charge_queue[0] = new
        return

    # if it overshoots by more than eps, something’s wrong
    raise RuntimeError("Committed charge would over-fill battery")


# --------------------------------------------------------------------
# Constraint-Checking Function
# --------------------------------------------------------------------
def check_solution_constraints(df_solution, battery_params, tol=1e-6):
    """Validates the solution against battery constraints."""
    simultaneous = df_solution.query(f"charge_mw > {tol} and discharge_mw > {tol}")
    if not simultaneous.empty:
        print("\nSimultaneous charging and discharging detected:")
        print(simultaneous[["datetime", "charge_mw", "discharge_mw"]])
        raise ValueError("Simultaneous charging and discharging occurred")

    if abs(df_solution["soc_open"].iloc[0]) > tol:
        raise ValueError(f"Initial SoC not zero: {df_solution['soc_open'].iloc[0]}")

    if (df_solution[["soc_open", "soc_close"]] < -tol).any().any() or \
       (df_solution[["soc_open", "soc_close"]] > battery_params.capacity + tol).any().any():
        raise ValueError("SoC out of bounds")

    if abs(df_solution["soc_close"].iloc[-1]) > tol:
        raise ValueError(f"Final SoC not zero: {df_solution['soc_close'].iloc[-1]}")


import pandas as pd
import numpy as np
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time
from tqdm import tqdm
from statistics import mean

# --------------------------------------------------------------------
# Runs the rolling in chunks (Extended to include the freeze version)
# --------------------------------------------------------------------


def run_parallel_chunks(
    model_df,
    battery_params: BatteryParams,
    dispatch_params: DispatchParams,
    perfect_results,
    chunk_duration_days,
    max_workers=None,
    log_file=None
):
    intervals_per_day = int(24 / dispatch_params.time_period_hours)
    intervals_per_chunk_duration = chunk_duration_days * intervals_per_day
    number_of_chunks = math.ceil(len(model_df) / intervals_per_chunk_duration)

    print(f"Splitting dataset into {number_of_chunks} chunks of ~{chunk_duration_days} days each...")

    data_chunks = {}  # dictionary now

    for chunk_id in range(number_of_chunks):
        chunk_start_idx = chunk_id * intervals_per_chunk_duration
        chunk_end_idx = min((chunk_id + 1) * intervals_per_chunk_duration, len(model_df))
        df_chunk = model_df.iloc[chunk_start_idx:chunk_end_idx].copy()
        df_chunk.index = pd.date_range(
            start=df_chunk.index[0],
            periods=len(df_chunk),
            freq=f"{int(dispatch_params.time_period_hours * 60)}min"
        )
        optimal_chunk = perfect_results.iloc[chunk_start_idx:chunk_end_idx]
        data_chunks[chunk_id] = (df_chunk, optimal_chunk)

    results_by_chunk = {}
    percentage_perfect_by_chunk = {}

    start_time = time.time()

    pbar = tqdm(total=number_of_chunks, desc="Solving Chunks")

    # Track which future corresponds to which chunk_id
    future_to_chunkid = {}

    with ProcessPoolExecutor(max_workers=max_workers or os.cpu_count()) as executor:
        futures = []
        for chunk_id, (df_chunk, opt_chunk) in data_chunks.items():
            future = executor.submit(
                solve_rolling_horizon,
                model_df=df_chunk,
                battery_params=battery_params,
                dispatch_params=dispatch_params,
                perfect_results=opt_chunk,
                log_file=log_file
            )
            future_to_chunkid[future] = chunk_id
            futures.append(future)

        for future in as_completed(futures):
            chunk_id = future_to_chunkid[future]
            result = future.result()
            results_by_chunk[chunk_id] = result

            df_chunk, optimal_chunk = data_chunks[chunk_id]

            chunk_profit = result["profit_step"].sum()
            optimal_profit = optimal_chunk["profit_step"].sum()
            perc_perfect = chunk_profit / optimal_profit

            percentage_perfect_by_chunk[chunk_id] = perc_perfect
            average_perc_perfect = mean(percentage_perfect_by_chunk.values())

            pbar.set_postfix({"Avg % Perfect": f"{average_perc_perfect * 100:.2f}%"})
            pbar.update(1)

            print(f"[Chunk {chunk_id}] Done - % Perfect: {perc_perfect * 100:.2f}%")

    pbar.close()

    final_result = pd.concat([results_by_chunk[i] for i in sorted(results_by_chunk.keys())]).reset_index(drop=True)

    total_time = time.time() - start_time
    print(f"All chunks completed in {total_time:.2f} seconds.")

    return final_result


# --------------------------------------------------------------------
# Comparison Function (Extended to include the freeze version)
# --------------------------------------------------------------------
def run_comparison(df, battery_params: BatteryParams, dispatach_params: DispatchParams, chunk_size=10):
    """Runs both solvers (fully optimal vs rolling with freeze), validates, and plots results."""

    print("Running Fully Optimal Solver...")
    perfect_results = solve_fully_optimal(df, battery_params, dispatach_params.time_period_hours)
    perfect_profit = perfect_results["profit_step"].sum()
    print(f"Fully Optimal:      ${perfect_profit:,.2f}")
    print("\nRunning Rolling-Horizon chunks")

    log_filename = "chunk_log.txt"
    if os.path.exists(log_filename):
        os.remove(log_filename)  # Clear old log if it exists

    model_results = run_parallel_chunks(df, battery_params, dispatach_params, perfect_results=perfect_results, chunk_duration_days=chunk_size, log_file=log_filename)
    print("\nValidating solutions...")
    for name, result in [
        ("Perfect", perfect_results),
        ("Model", model_results)
    ]:
        try:
            check_solution_constraints(result, battery_params)
            print(f"✓ {name}: VALID")
        except ValueError as e:
            print(f"✗ {name}: INVALID - {e}")

    model_profit = model_results["profit_step"].sum()
   
    percentage_of_perfect = (model_profit / perfect_profit) * 100
       
    print(f"\nProfit Comparison:")
    print(f"Fully Optimal:      ${perfect_profit:,.2f}")
    print(f"Rolling (freeze):   ${model_profit:,.2f}")
    print(f"Percentage of Perfect: {percentage_of_perfect:.2f}%")
   
    diff = abs(perfect_profit - model_profit)
    print(f"Absolute Difference: ${diff:,.2f}")
    if diff < 1e-6:
        print("Both solutions match! ✓")
    else:
        print(f"Model achieves {percentage_of_perfect:.2f}% of optimal performance.")

    return perfect_results, model_results

# --------------------------------------------------------------------
# Get Data
# --------------------------------------------------------------------

def get_data(name, start=None, end=None):
    df = pd.read_csv(
        name + ".csv",
        parse_dates=[0],
        dayfirst=True,
        index_col=0
    )

    if start is not None and end is not None:
        df_slice = df.loc[start:end]
        print(f"Full data shape: {df.shape}")
        print(f"Sliced data   : {df_slice.shape}  ({start} to {end})")
        return df_slice
    else:
        print(f"Data shape: {df.shape}")
        return df
    
import os
from datetime import datetime


def log(message, logfile=None):
    print(message)
    if logfile:
        with open(logfile, "a") as f:
            f.write(message + "\n")

def plot_profit_log(filename, output_folder=None):
    import re
    import pandas as pd
    import matplotlib.pyplot as plt

    # --- Load the text log ---
    with open(filename, "r") as f:
        text = f.read()

    # --- Extract data ---
    pattern = re.compile(
        r"Period: ([\d\- :]+) - ([\d\- :]+).*?"
        r"Model Profit:\s*\$([\d,\.]+).*?"
        r"Optimal Profit:\s*\$([\d,\.]+).*?"
        r"Percentage of Optimal:\s*([\d\.]+)%",
        re.DOTALL
    )

    data = []
    for match in pattern.findall(text):
        start, end, model_profit, optimal_profit, percentage = match
        data.append({
            "start": pd.to_datetime(start.strip()),
            "end": pd.to_datetime(end.strip()),
            "model_profit": float(model_profit.replace(",", "")),
            "optimal_profit": float(optimal_profit.replace(",", "")),
            "percentage": float(percentage)
        })

    df = pd.DataFrame(data)
    df = df.sort_values("start")

    # --- Plot % Perfect ---
    plt.figure()
    plt.plot(df["start"], df["percentage"], marker="o")
    plt.title("Percentage of Optimal Profit Over Time")
    plt.ylabel("Percentage (%)")
    plt.xlabel("Period Start Date")
    plt.grid(True)
    plt.tight_layout()

    if output_folder:
        plt.savefig(os.path.join(output_folder, "percentage_of_optimal.png"))
    else:
        plt.show()

    # --- Plot Model vs Optimal Profit ---
    plt.figure()
    plt.plot(df["start"], df["model_profit"], label="Model Profit", marker="o")
    plt.plot(df["start"], df["optimal_profit"], label="Optimal Profit", marker="o")
    plt.title("Profit Comparison Over Time")
    plt.ylabel("Profit ($)")
    plt.xlabel("Period Start Date")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if output_folder:
        plt.savefig(os.path.join(output_folder, "profit_comparison.png"))
    else:
        plt.show()



import shutil
# --------------------------------------------------------------------
# Save Results
# --------------------------------------------------------------------

def save_results(perfect_results, model_results, model_name):
    # Create "results" folder if it doesn't exist
    base_folder = "results"
    os.makedirs(base_folder, exist_ok=True)

    # Create timestamped subfolder
    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    subfolder = os.path.join(base_folder, f"{now_str}_{model_name}")
    os.makedirs(subfolder, exist_ok=True)

    # Save CSVs
    perfect_path = os.path.join(subfolder, "perfect_results.csv")
    model_path = os.path.join(subfolder, "model_results.csv")

    perfect_results.to_csv(perfect_path)
    model_results.to_csv(model_path)

    # ----- Save Summary TXT -----
    perfect_profit = perfect_results["profit_step"].sum()
    model_profit = model_results["profit_step"].sum()
    percentage_of_perfect = (model_profit / perfect_profit) * 100
    absolute_diff = abs(perfect_profit - model_profit)

    summary_text = (
        f"Profit Comparison:\n"
        f"Start Date: {perfect_results.iloc[0]['datetime']}     End Date: {perfect_results.iloc[-1]['datetime']}\n"
        f"Fully Optimal:      ${perfect_profit:,.2f}\n"
        f"Rolling (freeze):   ${model_profit:,.2f}\n"
        f"Percentage of Perfect: {percentage_of_perfect:.2f}%\n"
        f"Absolute Difference: ${absolute_diff:,.2f}\n"
    )

    summary_path = os.path.join(subfolder, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)


    # ----- Save chuvk log and plot -----
    log_src = "chunk_log.txt"
    log_dest = os.path.join(subfolder, "chunk_log.txt")
    if os.path.exists(log_src):
        shutil.copyfile(log_src, log_dest)
    plot_profit_log(log_dest, output_folder=subfolder)

    print(f"Results saved to {subfolder}")

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

if __name__ == '__main__':
    name = "aemo_SA1_model_median_with_aemo"
    #start, end = "2024-12-21", "2024-12-26"
    data_df = get_data(name)#, start, end)

    BATTERY_PARAMS = BatteryParams()
    DISPATCH_PARAMS = DispatchParams()

    perfect_results, model_results = run_comparison(data_df, BATTERY_PARAMS, DISPATCH_PARAMS, 10)

    save_results(perfect_results, model_results, model_name=name)
