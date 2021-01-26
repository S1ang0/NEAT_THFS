import neat
import os
import multiprocessing as mp
import timeit
import datetime
import gzip
import pickle
import fileinput
from shutil import copyfile

from lib.des.thfs_model import Model
import lib.toolbox.importer as importer
import lib.toolbox.allocation_functions as alloc
import lib.toolbox.sequencing_functions as seq
from lib.toolbox import visualize

### Settings ################################################################################################################################################################################################

# Define allocation mode: Job allocation ("job") or family allocation ("family")
ALLOCATION_OBJECTIVE = "job"

# Define sequencing mode: Pre-sequencing ("pre") or post-sequencing ("post")
SEQUENCING_OBJECTIVE = "pre"

# Fitness function
FITNESS_FUNCTION = lambda model: model.total_tardiness * (-1)

# Initial fitness to beat
INITIAL_FITNESS = -9999999

# Term to increase the best fitness value to be beaten in the next NEAT session
INCREASE_FITNESS_BY = 1

# Number of generations per NEAT session to beat the fitness of the previous session
NUM_GENERATIONS = 30

# Datasets for training and evaluation
DATASETS_OVERALL = ["dataset 1", "dataset 2", "dataset 3", "dataset 4"]
DATASETS_FOR_TRAINING = ["dataset 2", "dataset 3"]
DATASETS_FOR_TESTING = ["dataset 1", "dataset 4"]
NUM_JOBS = None  # Set to None, if all jobs of each

# Constants (usually not necessary to adapt)
AOI_ALLOCATION_FUNCTION = alloc.aoi_allocation_min_workload

#############################################################################################################################################################################################################

### NeuroEvolution of Augmenting Topologies #################################################################################################################################################################

def eval_fitness_alloc(genome, config):

    # Parse ANN
    neat_ann = neat.nn.FeedForwardNetwork.create(genome, config)

    # If possible: Parse sequencing ANN from best genome of last session
    try:
        gen, conf = importer.restore_genome("best_sequencing_genome", exp_dir)
        seq_ann = neat.nn.FeedForwardNetwork.create(gen, conf)
    except:
        seq_ann = None

    # Initialize fitness
    fitness_overall = 0

    # Evaluate NEAT genomes on datasets for training
    for dataset in train_datasets.values():
        model = run_simulation(dataset, neat_ann, seq_ann)
        fitness_overall += FITNESS_FUNCTION(model)

    return fitness_overall


def eval_fitness_seq(genome, config):

    # Parse ANN
    neat_ann = neat.nn.FeedForwardNetwork.create(genome, config)

    # If possible: Parse allocation ANN from best genome of last session
    try:
        gen, conf = importer.restore_genome("best_allocation_genome", exp_dir)
        alloc_ann = neat.nn.FeedForwardNetwork.create(gen, conf)
    except:
        alloc_ann = None

    # Initialize fitness
    fitness_overall = 0

    # Evaluate NEAT genomes on datasets for training
    for dataset in train_datasets.values():
        model = run_simulation(dataset, alloc_ann, neat_ann)
        fitness_overall += FITNESS_FUNCTION(model)

    return fitness_overall


def run_simulation(dataset, alloc_ann, seq_ann):

    # Determine sequencing function and initial job sequence
    if SEQUENCING_OBJECTIVE == "pre":
        post_sequencing_function = None
        if seq_ann is not None:
            sequence = seq.pre_sequencing_4_input_neurons(dataset, seq_ann)
        else:
            sequence = seq.fifo_pre_sequencing(dataset)
    else:
        sequence = seq.fifo_pre_sequencing(dataset)
        if seq_ann is not None:
            post_sequencing_function = seq.post_sequencing_5_input_neurons
        else:
            post_sequencing_function = seq.fifo_post_sequencing

    # Determine allocation function and eventually dictionary for family-smd mapping
    if ALLOCATION_OBJECTIVE == "family":
        smd_allocation_function = alloc.smd_allocation_based_on_family_smd_mapping
        if alloc_ann is not None:
            alloc_dict = alloc.map_families_to_smds_17_input_neurons(dataset, alloc_ann)
        else:
            alloc_dict = alloc.map_families_to_smds_equal(dataset)
        for job in sequence:
            job["alloc_to_smd"] = alloc_dict[job["family"]]
    else:
        if alloc_ann is not None:
            smd_allocation_function = alloc.smd_allocation_9_input_neurons
        else:
            smd_allocation_function = alloc.smd_allocation_min_workload

    # Create and run model
    model = Model(
        sequence=sequence,
        dataset=dataset,
        smd_allocation_function=smd_allocation_function,
        aoi_allocation_function=AOI_ALLOCATION_FUNCTION,
        post_sequencing_function=post_sequencing_function,
        smd_allocation_ann=alloc_ann,
        post_sequencing_ann=seq_ann,
    )

    # Return terminated model
    return model


def init_neat():

    # Get config file depending on current mode
    config_file = os.path.join(
        exp_dir, alloc_config if mode == "allocation" else seq_config
    )

    # Update fitness in config file
    threshold_line = False
    for line in fileinput.input(config_file, inplace=1):
        if line.startswith("fitness_criterion"):
            threshold_line = True
        elif line.startswith("fitness_threshold") or line.startswith(
            "no_fitness_termination"
        ):
            continue
        else:
            if threshold_line:
                print(
                    "fitness_threshold      = {}".format(
                        best_fitness + INCREASE_FITNESS_BY
                    )
                )
                print("no_fitness_termination = False")
            threshold_line = False
        print(line.strip())

    # Parse NEAT configuration
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )

    # Initialize population with NEAT configuration
    population = neat.Population(config)

    # Add statistical reporters to population
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(1))

    # Save initial size and generation of population
    initial_pop_size = len(population.population)

    # Create workers
    evaluators = neat.ParallelEvaluator(
        mp.cpu_count(), eval_fitness_alloc if mode == "allocation" else eval_fitness_seq
    )

    # Run NEAT
    print("\n### Initialize {} population ###".format(mode))
    winner = population.run(evaluators.evaluate, NUM_GENERATIONS)

    # Print best genome
    print("\nBest genome:\n{!s}".format(winner))

    # Calculate approximal number of performed simulation runs
    num_runs = round(
        ((initial_pop_size + len(population.population)) / 2) * population.generation
    )

    # Prepare next NEAT session
    init_succeed = prepare_next_session(True, winner, config)

    return init_succeed, num_runs, stats if init_succeed else None


def run_last_cp():

    # Restore last checkpoint of allocation or sequencing population
    population = neat.Checkpointer.restore_checkpoint("last_cp_{}".format(mode))

    # Adjust fitness threshold of population
    population.config.fitness_threshold = best_fitness + INCREASE_FITNESS_BY

    # Add statistical reporters to population
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(stats_alloc if mode == "allocation" else stats_seq)
    population.add_reporter(neat.Checkpointer(1))

    # Save initial size and generation of population
    initial_pop_size = len(population.population)
    initial_gen = population.generation

    # Create workers
    evaluators = neat.ParallelEvaluator(
        mp.cpu_count(), eval_fitness_alloc if mode == "allocation" else eval_fitness_seq
    )

    # Run NEAT
    print("\n### Improve {} population ###".format(mode))
    winner = population.run(evaluators.evaluate, NUM_GENERATIONS)

    # Print best genome
    print("\nBest genome:\n{!s}".format(winner))

    # Calculate approximal number of performed simulation runs
    num_runs = round(
        ((initial_pop_size + len(population.population)) / 2)
        * (population.generation - initial_gen)
    )

    # Prepare next NEAT session
    abort = prepare_next_session(False, winner, population.config)

    return abort, num_runs


def prepare_next_session(init, genome, config):

    # Declare best_fitness as global variable
    global best_fitness

    # Save last checkpoint
    max_cp = -1
    num_cp = 0
    last_cp_file = None
    for filename in os.listdir(exp_dir):
        if "neat-checkpoint" in filename:
            num_cp += 1
            current_cp = int(filename.split("-")[2])
            if current_cp > max_cp:
                last_cp_file = filename
                max_cp = current_cp
    if last_cp_file:
        if os.path.isfile("last_cp_{}".format(mode)):
            os.remove("last_cp_{}".format(mode))
        os.rename(last_cp_file, "last_cp_{}".format(mode))
        init_succeed = True
    else:
        init_succeed = False

    # Delete other checkpoints
    for filename in os.listdir(exp_dir):
        if "neat-checkpoint" in filename:
            os.remove(filename)

    # Save new best genome if it is better than the last best genome
    if genome.fitness >= best_fitness:
        filename = "best_{}_genome.bin".format(mode)
        try:
            os.remove(filename)
        except:
            pass
        with gzip.open(filename, "w", compresslevel=2) as f:
            data = (genome, config)
            pickle.dump(data, f, 2)
        print(
            "\n--- New best {} genome with fitness {} (before {}) ---".format(
                mode, genome.fitness, best_fitness
            )
        )
        best_fitness = genome.fitness
    else:
        print(
            "\n--- Best {} genome rejected, because fitness ({}) is worse than best_fitness ({}) ---".format(
                mode, genome.fitness, best_fitness
            )
        )

    # If this function was called from init_neat(), ...
    if init:
        # ...the returned value indicates whether or not a population was succesfully initialized
        return init_succeed
    # Otherwise, this function was called from run_last_cp()
    # In this case, the returned value indicates if NEAT for for the current mode shall be aborted
    elif num_cp >= NUM_GENERATIONS:
        # NEAT for the current mode will be aborted if the number of checkpoints meets or exceeds NUM_GENERATIONS
        abort = True
    else:
        abort = False

    return abort


def final_report():

    alloc_gen, alloc_conf = importer.restore_genome("best_allocation_genome", exp_dir)
    alloc_ann = neat.nn.FeedForwardNetwork.create(alloc_gen, alloc_conf)

    seq_gen, seq_conf = importer.restore_genome("best_sequencing_genome", exp_dir)
    seq_ann = neat.nn.FeedForwardNetwork.create(seq_gen, seq_conf)

    print(
        "\n### FINAL REPORT ###",
        "\n\nBest allocation genome:\n{!s}".format(alloc_gen),
        "\n\nBest sequencing genome:\n{!s}".format(seq_gen),
    )

    for dataset in all_datasets.items():
        model = run_simulation(dataset[1], alloc_ann, seq_ann)
        print(
            "\n--- Performance on {} ------------------------".format(dataset[0]),
            "\nMakespan: ",
            model.makespan,
            "\nTotal tardiness: ",
            model.total_tardiness,
            "\nMajor setups: ",
            model.num_major_setups,
            "\nJobs processed: ",
            model.jobs_processed,
            "\n-----------------------------------------------------",
        )

    print(
        "\nComputational time: ",
        timeit.default_timer() - start,
        "\nApproximal number of simulation runs (allocation | sequencing | overall): {} | {} | {}\n".format(
            num_runs_alloc, num_runs_seq, num_runs_alloc + num_runs_seq
        ),
    )

    # visualize.draw_net(alloc_conf, alloc_gen, False, filename="alloc_net", objective=ALLOCATION_OBJECTIVE) # works only of graphviz binaries are installed
    visualize.plot_stats(
        stats_alloc, ylog=False, view=False, filename="avg_fitness_alloc.svg"
    )
    visualize.plot_species(stats_alloc, view=False, filename="speciation_alloc.svg")

    # visualize.draw_net(seq_conf, seq_gen, False, filename="seq_net", objective=SEQUENCING_OBJECTIVE) # works only of graphviz binaries are installed
    visualize.plot_stats(
        stats_seq, ylog=False, view=False, filename="avg_fitness_seq.svg"
    )
    visualize.plot_species(stats_seq, view=False, filename="speciation_seq.svg")

#############################################################################################################################################################################################################

### Execution ###############################################################################################################################################################################################

# Set timer
start = timeit.default_timer()

# Get working directory for importing problem instances
this_dir = os.path.dirname(__file__)

# Data preparation
all_datasets = importer.problem_datasets(this_dir, NUM_JOBS)
train_datasets = {key: all_datasets[key] for key in DATASETS_FOR_TRAINING}
test_datasets = {key: all_datasets[key] for key in DATASETS_FOR_TESTING}

# Get config file for allocation
if ALLOCATION_OBJECTIVE == "job":
    alloc_config = "config_job_allocation"
else:
    alloc_config = "config_family_allocation"

# Get config file for sequencing
if SEQUENCING_OBJECTIVE == "pre":
    seq_config = "config_pre_sequencing"
else:
    seq_config = "config_post_sequencing"

# Global variables and flags
best_fitness = INITIAL_FITNESS
stats_alloc = None
stats_seq = None
num_runs_alloc = 0
num_runs_seq = 0
abort_alloc = False
abort_seq = False
init_succeed_alloc = False
init_succeed_seq = False
mode = "allocation"

if __name__ == "__main__":
    # Get current datetime
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    # Create experiment folder
    exp_dir = os.path.join(
        this_dir,
        "exp\\{}_{}_{}_{}".format(
            now, "competitive_neat", SEQUENCING_OBJECTIVE, ALLOCATION_OBJECTIVE
        ),
    )
    os.makedirs(exp_dir)
    # Copy config files to experiment folder
    copyfile(os.path.join(this_dir, alloc_config), os.path.join(exp_dir, alloc_config))
    copyfile(os.path.join(this_dir, seq_config), os.path.join(exp_dir, seq_config))
    # Change working directory to experiment folder
    os.chdir(exp_dir)
    # Run competitive NEAT
    while abort_alloc == False or abort_seq == False:
        if mode == "allocation":
            if abort_alloc == False:
                if init_succeed_alloc == False:
                    init_succeed_alloc, num_runs, stats_alloc = init_neat()
                    if init_succeed_alloc:
                        print(
                            "\n--- Initialization of allocation population successful ---"
                        )
                    else:
                        print(
                            "\n--- Initialization of allocation population not successful, because not enough generations were evolved to create a checkpoint ---"
                        )
                else:
                    abort_alloc, num_runs = run_last_cp()
                    if abort_alloc == True:
                        print(
                            "\n--- Max. number of generations of NEAT for allocation reached --> Abort NEAT for allocation ---"
                        )
                num_runs_alloc += num_runs
            else:
                print(
                    "\n--- NEAT for allocation already aborted --> Switching back to NEAT for sequencing ---"
                )
            mode = "sequencing"
        else:
            if abort_seq == False:
                if init_succeed_seq == False:
                    init_succeed_seq, num_runs, stats_seq = init_neat()
                    if init_succeed_seq:
                        print(
                            "\n--- Initialization of sequencing population successful ---"
                        )
                    else:
                        print(
                            "\n--- Initialization of sequencing population not successful, because not enough generations were evolved to create a checkpoint ---"
                        )
                else:
                    abort_seq, num_runs = run_last_cp()
                    if abort_seq == True:
                        print(
                            "\n--- Max. number of generations of NEAT for allocation reached --> Abort NEAT for allocation ---"
                        )
                num_runs_seq += num_runs
            else:
                print(
                    "\n--- NEAT for sequencing already aborted --> Switching back to NEAT for allocation ---"
                )
            mode = "allocation"
    # Print final report
    final_report()

#############################################################################################################################################################################################################
