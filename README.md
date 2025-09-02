delphi-battery-sim is a Python framework for simulating battery dispatch using linear programming. It supports both a fully optimal solver (with perfect foresight) and a rolling-horizon solver (realistic lookahead), allowing you to benchmark strategies against the theoretical maximum. Results include state-of-charge (SoC), charge/discharge flows, and profit comparisons.

Usage
	1.	Update the file name in main (e.g. name = "aemo_SA1_model_median_with_aemo") to point to your CSV.
 
	2.	Optionally set start and end dates to slice the dataset.
 
	3.	Adjust battery parameters in BatteryParams if needed.

 
  @dataclass(frozen=True)
  class BatteryParams:
      capacity: float = 2.0          # MWh, usable energy
      charge_efficiency: float = 0.96
      discharge_efficiency: float = 0.96
      max_charge_rate: float = 1.04  # MW to/from grid
      max_discharge_rate: float = 0.96  # MW to/from grid


Important: max_charge_rate and max_discharge_rate are defined with respect to the grid, not the battery’s internal transfer rate.

Output
	•	results/ folder contains timestamped subfolders with:
	•	CSVs of “perfect” and “model” results
	•	Profit summaries and %-of-optimal metrics
	•	Plots comparing rolling vs. fully optimal performance
    
