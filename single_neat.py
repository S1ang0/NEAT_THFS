import neat
import os
import multiprocessing as mp
import timeit
import datetime
import gzip
import pickle
from shutil import copyfile

from lib.des.thfs_model import Model
import lib.toolbox.importer as importer
import lib.toolbox.allocation_functions as alloc
import lib.toolbox.sequencing_functions as seq
from lib.toolbox import visualize

### Settings ################################################################################################################################################################################################

# Define for which task NEAT should find an ANN
OBJECTIVE = "pre_sequencing"  # possible values: ["pre_sequencing", "post_sequencing" "job_allocation", "family_allocation"]

#  Fitness function
FITNESS_FUNCTION = lambda model: model.total_tardiness * (-1)

# Number of generations
NUM_GENERATIONS = 100

# Define after how many generations a checkpoint shall be created
CREATE_CHECKPOINT_AFTER = 10

# Sequencing function (This option will be only applied if OBJECTIVE is not "pre_sequencing" or "post_sequencing" and if ANN_FOR_SEQUENCING is not given)
SEQUENCING_FUNCTION = seq.pre_sequencing_4_input_neurons

# Allocation functions (This option will be only applied if OBJECTIVE is not "job_allocation" or "family_allocation" and if ANN_FOR_ALLOCATION is not given)
SMD_ALLOCATION_FUNCTION = alloc.smd_allocation_min_workload

# Family-SMD-Mapping --> This option will be overwriten if:
#  1) OBJECTIVE is "family_allocation" or
#  2) ANN_FOR_ALLOCATION is given or
#  3) SMD_ALLOCATION_FUNCTION is not smd_allocation_based_on_family_smd_mapping
FAMILY_SMD_MAPPING = alloc.map_families_to_smds_equal

# Utilized ANNs
ANN_FOR_SEQUENCING = (
    None  # possible values: None or filename (without ".bin") of the ANN to be restored
)
ANN_FOR_ALLOCATION = (
    None  # possible values: None or filename (without ".bin") of the ANN to be restored
)

# Datasets for training and evaluation
DATASETS_OVERALL = ["dataset 1", "dataset 2", "dataset 3", "dataset 4"]
DATASETS_FOR_TRAINING = ["dataset 2", "dataset 3"]
DATASETS_FOR_TESTING = ["dataset 1", "dataset 4"]
NUM_JOBS = None  # Set to None if all jobs of each dataset shall be considered

# Constants (usually not necessary to adapt)
AOI_ALLOCATION_FUNCTION = alloc.aoi_allocation_min_workload

#############################################################################################################################################################################################################

### NeuroEvolution of Augmenting Topologies #################################################################################################################################################################

def eval_fitness(genome, config):

    # Parse ANN
    neat_ann = neat.nn.FeedForwardNetwork.create(genome, config)

    # Initialize fitness
    fitness_overall = 0

    # Evaluate NEAT genomes on datasets for training
    for dataset in train_datasets.values():
        model = run_simulation(dataset, neat_ann)
        fitness_overall += FITNESS_FUNCTION(model)

    return fitness_overall


def run_simulation(dataset, neat_ann):

    # Determine sequencing function, sequencing ANN and initial job sequence
    if OBJECTIVE == "pre_sequencing":
        sequence = SEQUENCING_FUNCTION(dataset, neat_ann)
    elif "pre" in SEQUENCING_FUNCTION.__name__:
        sequence = SEQUENCING_FUNCTION(dataset, sequencing_ann)
    else:
        sequence = seq.fifo_pre_sequencing(dataset)

    # Determine allocation function, allocation ANN and eventually dictionary for family-smd mapping
    if FAMILY_SMD_MAPPING:
        if OBJECTIVE == "family_allocation":
            alloc_dict = FAMILY_SMD_MAPPING(dataset, neat_ann)
        else:
            alloc_dict = FAMILY_SMD_MAPPING(dataset, allocation_ann)
        for job in sequence:
            job["alloc_to_smd"] = alloc_dict[job["family"]]

    # Create and run model
    model = Model(
        sequence=sequence,
        dataset=dataset,
        smd_allocation_function=SMD_ALLOCATION_FUNCTION,
        aoi_allocation_function=AOI_ALLOCATION_FUNCTION,
        post_sequencing_function=SEQUENCING_FUNCTION
        if "post" in SEQUENCING_FUNCTION.__name__
        else None,
        smd_allocation_ann=neat_ann
        if OBJECTIVE == "job_allocation"
        else allocation_ann,
        post_sequencing_ann=neat_ann
        if OBJECTIVE == "post_sequencing"
        else sequencing_ann,
    )

    # Return terminated model
    return model


def run_neat(config_file):

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
    population.add_reporter(neat.Checkpointer(CREATE_CHECKPOINT_AFTER))

    # Save initial size and generation of population
    initial_pop_size = len(population.population)

    # Create workers
    evaluators = neat.ParallelEvaluator(1, eval_fitness)

    # Run NEAT
    winner = population.run(evaluators.evaluate, NUM_GENERATIONS)

    # Save best genome
    now = datetime.datetime.now()
    filename = "{}_winner_{}_genome.bin".format(
        now.strftime("%Y-%m-%d_%H.%M.%S"), OBJECTIVE
    )
    with gzip.open(filename, "w", compresslevel=2) as f:
        data = (winner, config)
        pickle.dump(data, f, 2)

    # Parse artificial neural network from best genome
    winner_ann = neat.nn.FeedForwardNetwork.create(winner, config)

    # Create final report
    print("\n### FINAL REPORT ###", "\n\nBest genome:\n{!s}".format(winner))

    for dataset in all_datasets.items():
        model = run_simulation(dataset[1], winner_ann)
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
        "\nApproximal number of simulation runs: {}\n".format(
            round(
                ((initial_pop_size + len(population.population)) / 2)
                * population.generation
            )
        ),
    )

    # visualize.draw_net(config, winner, True, objective=OBJECTIVE) # works only of graphviz binaries are installed
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

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

# pylint: disable=E1135

# Adapt sequencing function if sequencing OBJECTIVE or ANN_FOR_SEQUENCING is given
if (
    OBJECTIVE == "pre_sequencing"
    or ANN_FOR_SEQUENCING
    and "pre_sequencing" in ANN_FOR_SEQUENCING
):
    SEQUENCING_FUNCTION = seq.pre_sequencing_4_input_neurons
    if __name__ == "__main__":
        print(
            "\n-------------------------------------------------------------------------------------------------------------",
            "\nSEQUENCING_FUNCTION automatically set to {}, because OBJECTIVE is".format(
                SEQUENCING_FUNCTION.__name__
            ),
            "\n'pre_sequencing' or ANN_FOR_SEQUENCING contains the path to a genome for pre_sequencing",
            "\n-------------------------------------------------------------------------------------------------------------",
        )
elif (
    OBJECTIVE == "post_sequencing"
    or ANN_FOR_SEQUENCING
    and "post_sequencing" in ANN_FOR_SEQUENCING
):
    SEQUENCING_FUNCTION = seq.post_sequencing_5_input_neurons
    if __name__ == "__main__":
        print(
            "\n-------------------------------------------------------------------------------------------------------------",
            "\nSEQUENCING_FUNCTION automatically set to {}, because OBJECTIVE is".format(
                SEQUENCING_FUNCTION.__name__
            ),
            "\n'post_sequencing' or ANN_FOR_SEQUENCING contains the path to a genome for post_sequencing",
            "\n-------------------------------------------------------------------------------------------------------------",
        )
elif not ANN_FOR_SEQUENCING and "neuron" in SEQUENCING_FUNCTION.__name__:
    required_mode = (
        "pre_sequencing"
        if "pre_sequencing" in SEQUENCING_FUNCTION.__name__
        else "post_sequencing"
    )
    SEQUENCING_FUNCTION = seq.fifo_pre_sequencing
    if __name__ == "__main__":
        print(
            "\n-------------------------------------------------------------------------------------------------------------",
            "\nSEQUENCING_FUNCTION automatically set to {}, because ANN_FOR_SEQUENCING is not given.".format(
                SEQUENCING_FUNCTION.__name__
            ),
            "\nThe user selected SEQUENCING_FUNCTION is only valid if ANN_FOR_SEQUENCING is not None",
            "\nor if the OBJECTIVE is '{}'".format(required_mode),
            "\n-------------------------------------------------------------------------------------------------------------",
        )

# Adapt allocation function if allocation OBJECTIVE or ANN_FOR_ALLOCATION is given
if (
    OBJECTIVE == "job_allocation"
    or ANN_FOR_ALLOCATION
    and "job_allocation" in ANN_FOR_ALLOCATION
):
    SMD_ALLOCATION_FUNCTION = alloc.smd_allocation_9_input_neurons
    if __name__ == "__main__":
        print(
            "\n-------------------------------------------------------------------------------------------------------------",
            "\nSMD_ALLOCATION_FUNCTION automatically set to {}, because OBJECTIVE is".format(
                SMD_ALLOCATION_FUNCTION.__name__
            ),
            "\n'job_allocation' or ANN_FOR_SEQUENCING contains the path to a genome for job_allocation",
            "\n-------------------------------------------------------------------------------------------------------------",
        )
elif (
    OBJECTIVE == "family_allocation"
    or ANN_FOR_ALLOCATION
    and "family_allocation" in ANN_FOR_ALLOCATION
):
    SMD_ALLOCATION_FUNCTION = alloc.smd_allocation_based_on_family_smd_mapping
    FAMILY_SMD_MAPPING = alloc.map_families_to_smds_17_input_neurons
    if __name__ == "__main__":
        print(
            "\n-------------------------------------------------------------------------------------------------------------",
            "\nSMD_ALLOCATION_FUNCTION automatically set to ",
            SMD_ALLOCATION_FUNCTION.__name__,
            "\nand FAMILY_SMD_MAPPING automatically set to {},".format(
                FAMILY_SMD_MAPPING.__name__
            ),
            "\nbecause OBJECTIVE is 'family_allocation' or ANN_FOR_SEQUENCING contains the path to a genome",
            "\nfor family_allocation",
            "\n-------------------------------------------------------------------------------------------------------------",
        )
elif not ANN_FOR_ALLOCATION and "neuron" in SMD_ALLOCATION_FUNCTION.__name__:
    required_mode = (
        "job_allocation"
        if "job_allocation" in SMD_ALLOCATION_FUNCTION.__name__
        else "family_allocation"
    )
    SMD_ALLOCATION_FUNCTION = alloc.smd_allocation_min_workload
    if __name__ == "__main__":
        print(
            "\n-------------------------------------------------------------------------------------------------------------",
            "\nSMD_ALLOCATION_FUNCTION automatically set to {}, because ANN_FOR_ALLOCATION is not".format(
                SMD_ALLOCATION_FUNCTION.__name__
            ),
            "\ngiven. The user selected SMD_ALLOCATION_FUNCTION is only valid if ANN_FOR_SEQUENCING is not None",
            "\nor if the OBJECTIVE is '{}'".format(required_mode),
            "\n-------------------------------------------------------------------------------------------------------------",
        )

# pylint: enable=E1135

# Create ANNs
if ANN_FOR_SEQUENCING is not None:
    gen, conf = importer.restore_genome(ANN_FOR_SEQUENCING, this_dir)
    sequencing_ann = neat.nn.FeedForwardNetwork.create(gen, conf)
else:
    sequencing_ann = None

if ANN_FOR_ALLOCATION is not None:
    gen, conf = importer.restore_genome(ANN_FOR_ALLOCATION, this_dir)
    allocation_ann = neat.nn.FeedForwardNetwork.create(gen, conf)
else:
    allocation_ann = None

if __name__ == "__main__":
    # Get config file
    config_file = "config_{}".format(OBJECTIVE)
    config_path = os.path.join(this_dir, config_file)
    # Get current datetime
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    # Create experiment folder
    exp_dir = os.path.join(
        this_dir, "exp\\{}_{}_{}".format(now, "single_neat", OBJECTIVE)
    )
    os.makedirs(exp_dir)
    # Copy config file to experiment folder
    copyfile(os.path.join(this_dir, config_file), os.path.join(exp_dir, config_file))
    # Change working directory to experiment folder
    os.chdir(exp_dir)
    # Run NEAT
    run_neat(config_path)

#############################################################################################################################################################################################################
