# Graph-based Methods for Discrete Choice
This repository accompanies the paper: 
> Graph-based Methods for Discrete Choice.</br>
> Kiran Tomlinson and Austin R. Benson.</br>
> *Network Science*, 2023.

## Contents
This repository includes one directory:
- `code/`: all code needed to process data, implement our methods, run our experiments, and plot results
In addition, the following directories containing our results and data can be downloaded from https://osf.io/egj4q/:
- `raw-data/`: raw data files for the election data
- `data/`: processed versions of our datasets
- `results/`: our experiment result files
The instructions below assume these directories are placed alongside `code/`.

## Data
In https://osf.io/egj4q/, we provide our processed versions of the us-elec-2016, ca-elec-2016,
and ca-elec-2020 datasets as well as the original raw data 
(original sources: https://statewidedatabase.org/election.html, https://github.com/mkearney/presidential_election_county_results_2016, and https://github.com/000Justin000/gnn-residual-correlation/tree/master/datasets/election). In the code, they are named election-2016,
ca-election-2016, and ca-election-2020. The Friends and Family datasets
are not licensed for redistribution so we have not included them here, but they can be requested from 
http://realitycommons.media.mit.edu/friendsdataset.html and our data
processing code is included (see reproducibility instructions below).
Our California data is built from the 2016 and 2020 General Election data
SOV (statement of vote) and REG (registration) files, per-county codebooks,
and precinct SRPREC_SHP files. These are all available at 
https://statewidedatabase.org/election.html, but we include them for completeness.


## Reproducibility
We provide instructions for replicating all experiments.

### 1. Setup
Download the `raw-data/`, `data/`, and `results/` folders from https://osf.io/egj4q/ and place them alongside `code/`.
After requesting the Friends and Family dataset as described above, place the Friends and Family files in `raw-data/` (in
particular, `App.csv`, `AppRunning.csv`, and `BluetoothProximity.csv`).
Then, modify `config.yml` so that `parsed_data_dir` contains the path to
the `data/` directory included in this repo, `data_dir` contains the path
to `raw-data/`, and `results_dir` contains the path to the
`results/`.

### 2. Friends and Family data processing
Simply run `python3 parse-data.py` to generate the processed `app-install` and 
`app-usage` directories in `data/`.

### 3. Experiments
First, set the number of threads you want to run on (we used 100)
```
export NUM_THREADS=[number of threads to run on]
```
To run the synthetic data sample complexity experiment:
```
python3 expriments.py --network-convergence --threads $NUM_THREADS
```
To run all the election experiments:
```
python3 experiments.py --election-social election-2016 --threads $NUM_THREADS
python3 experiments.py --election-adjacency ca-election-2016 --threads $NUM_THREADS
python3 experiments.py --election-adjacency ca-election-2020 --threads $NUM_THREADS
python3 experiments.py --election-social election-2016 --propagation --threads $NUM_THREADS
python3 experiments.py --election-adjacency ca-election-2016 --propagation --threads $NUM_THREADS
python3 experiments.py --election-adjacency ca-election-2020 --propagation --threads $NUM_THREADS
```
To run all the app experiments:
```
python3 experiments.py --masked --datasets app-usage app-install --per-item-utilities --app-prox-thresholds 10 --threads $NUM_THREADS 
python3 experiments.py --propagation --datasets app-usage app-install --app-prox-thresholds 10 --threads $NUM_THREADS 
```
To run the runtime experiment:
```
python3 expriments.py --timing
```

### 4. Plotting
Simply run 
```
python3 plot.py
```
This will print out data for all the tables and save plots to `code/plots/`.




