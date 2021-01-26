import neat
import os
import multiprocessing
import timeit
import datetime
import gzip
import pickle

from lib.des.thfs_model import Model
import lib.toolbox.importer as importer
import lib.toolbox.allocation_functions as alloc
import lib.toolbox.sequencing_functions as seq
from lib.toolbox import visualize

### Settings ################################################################################################################################################################################################

# Animation and tracing options
ANIMATION = False
STEP_BY_STEP_EXECUTION = False
FREEZE_WINDOW_AT_ENDSIM = False
TRACING = False

# Sequencing function (This option will be only applied if ANN_FOR_SEQUENCING is not given)
SEQUENCING_FUNCTION = seq.edd_pre_sequencing

# Allocation functions (This option will be only applied if ANN_FOR_ALLOCATION is not given)
SMD_ALLOCATION_FUNCTION = alloc.smd_allocation_min_workload

# Family-SMD-Mapping --> This option will be no applied if:
#  1) ANN_FOR_ALLOCATION is given or
#  2) SMD_ALLOCATION_FUNCTION is not smd_allocation_based_on_family_smd_mapping
FAMILY_SMD_MAPPING = None

# Utilized ANNs
ANN_FOR_SEQUENCING = (
    None  # possible values: None or filename (without '.bin') of the ANN to be restored
)
ANN_FOR_ALLOCATION = (
    None  # possible values: None or filename (without '.bin') of the ANN to be restored
)

# Datasets for learning and evaluation
DATASETS_OVERALL = ["dataset 1", "dataset 2", "dataset 3", "dataset 4"]
DATASETS_FOR_EVALUATION = ["dataset 1", "dataset 2", "dataset 3", "dataset 4"]
N_JOBS = None  # Set to None, if all jobs of each dataset shall be considered

# Constants (usually not nescessary to adapt)
AOI_ALLOCATION_FUNCTION = alloc.aoi_allocation_min_workload

#############################################################################################################################################################################################################

### Execution ###############################################################################################################################################################################################

if __name__ == "__main__":

    # Get working directory
    local_dir = os.path.dirname(__file__)

    # Data preparation
    all_datasets = importer.problem_datasets(local_dir, N_JOBS)
    eval_datasets = {key: all_datasets[key] for key in DATASETS_FOR_EVALUATION}

    # pylint: disable=E1135

    # Adapt sequencing function if sequencing OBJECTIVE or ANN_FOR_SEQUENCING is given
    if ANN_FOR_SEQUENCING:
        if "pre_sequencing" in ANN_FOR_SEQUENCING:
            SEQUENCING_FUNCTION = seq.pre_sequencing_4_input_neurons
            print(
                "\n-------------------------------------------------------------------------------------------------------------",
                "\nSEQUENCING_FUNCTION automatically set to {}, because ANN_FOR_SEQUENCING".format(
                    SEQUENCING_FUNCTION.__name__
                ),
                "\ncontains the path to a genome for pre_sequencing",
                "\n-------------------------------------------------------------------------------------------------------------",
            )
        elif "post_sequencing" in ANN_FOR_SEQUENCING:
            SEQUENCING_FUNCTION = seq.post_sequencing_5_input_neurons
            print(
                "\n-------------------------------------------------------------------------------------------------------------",
                "\nSEQUENCING_FUNCTION automatically set to {}, because ANN_FOR_SEQUENCING".format(
                    SEQUENCING_FUNCTION.__name__
                ),
                "\ncontains the path to a genome for post_sequencing",
                "\n-------------------------------------------------------------------------------------------------------------",
            )
    elif "neuron" in SEQUENCING_FUNCTION.__name__:
        required_mode = (
            "pre_sequencing"
            if "pre_sequencing" in SEQUENCING_FUNCTION.__name__
            else "post_sequencing"
        )
        SEQUENCING_FUNCTION = seq.fifo_pre_sequencing
        print(
            "\n-------------------------------------------------------------------------------------------------------------",
            "\nSEQUENCING_FUNCTION automatically set to {}, because ANN_FOR_SEQUENCING is not given.".format(
                SEQUENCING_FUNCTION.__name__
            ),
            "\nThe user selected SEQUENCING_FUNCTION is only valid if ANN_FOR_SEQUENCING contains the path",
            "\nto a genome for",
            required_mode,
            "\n-------------------------------------------------------------------------------------------------------------",
        )

    # Adapt allocation function if allocation OBJECTIVE or ANN_FOR_ALLOCATION is given
    if ANN_FOR_ALLOCATION:
        if "job_allocation" in ANN_FOR_ALLOCATION:
            SMD_ALLOCATION_FUNCTION = alloc.smd_allocation_9_input_neurons
            print(
                "\n-------------------------------------------------------------------------------------------------------------",
                "\nSMD_ALLOCATION_FUNCTION automatically set to {}, because ANN_FOR_ALLOCATION".format(
                    SMD_ALLOCATION_FUNCTION.__name__
                ),
                "\ncontains the path to a genome for job_allocation",
                "\n-------------------------------------------------------------------------------------------------------------",
            )
        elif "family_allocation" in ANN_FOR_ALLOCATION:
            SMD_ALLOCATION_FUNCTION = alloc.smd_allocation_based_on_family_smd_mapping
            FAMILY_SMD_MAPPING = alloc.map_families_to_smds_17_input_neurons
            print(
                "\n-------------------------------------------------------------------------------------------------------------",
                "\nSMD_ALLOCATION_FUNCTION automatically set to ",
                SMD_ALLOCATION_FUNCTION.__name__,
                "\nand FAMILY_SMD_MAPPING automatically set to {},".format(
                    FAMILY_SMD_MAPPING.__name__
                ),
                "\nbecause ANN_FOR_SEQUENCING contains the path to a genome for family_allocation"
                "\n-------------------------------------------------------------------------------------------------------------",
            )
    elif "neuron" in SMD_ALLOCATION_FUNCTION.__name__:
        required_mode = (
            "job_allocation"
            if "job_allocation" in SMD_ALLOCATION_FUNCTION.__name__
            else "family_allocation"
        )
        SMD_ALLOCATION_FUNCTION = alloc.smd_allocation_min_workload
        print(
            "\n-------------------------------------------------------------------------------------------------------------",
            "\nSMD_ALLOCATION_FUNCTION automatically set to {}, because ANN_FOR_ALLOCATION is not".format(
                SMD_ALLOCATION_FUNCTION.__name__
            ),
            "\ngiven. The user selected SMD_ALLOCATION_FUNCTION is only valid if ANN_FOR_SEQUENCING contains the path",
            "\nto a genome for",
            required_mode,
            "\n-------------------------------------------------------------------------------------------------------------",
        )
    elif (
        SMD_ALLOCATION_FUNCTION is alloc.smd_allocation_based_on_family_smd_mapping
        and not FAMILY_SMD_MAPPING
        and not "map_families_to_smds" in FAMILY_SMD_MAPPING.__name__
    ):
        SMD_ALLOCATION_FUNCTION = alloc.smd_allocation_min_workload
        print(
            "\n-------------------------------------------------------------------------------------------------------------",
            "\nSMD_ALLOCATION_FUNCTION automatically set to {}, because the user selected function".format(
                SMD_ALLOCATION_FUNCTION.__name__
            ),
            "\n'smd_allocation_based_on_family_smd_mapping' requires a valid FAMILY_SMD_MAPPING function",
            "\n-------------------------------------------------------------------------------------------------------------",
        )

    # pylint: enable=E1135

    # Create ANNs
    if ANN_FOR_SEQUENCING is not None:
        gen, conf = importer.restore_genome(ANN_FOR_SEQUENCING, local_dir)
        sequencing_ann = neat.nn.FeedForwardNetwork.create(gen, conf)
    else:
        sequencing_ann = None

    if ANN_FOR_ALLOCATION is not None:
        gen, conf = importer.restore_genome(ANN_FOR_ALLOCATION, local_dir)
        allocation_ann = neat.nn.FeedForwardNetwork.create(gen, conf)
    else:
        allocation_ann = None

    def run_simulation(dataset):

        if "pre" in SEQUENCING_FUNCTION.__name__:
            sequence = SEQUENCING_FUNCTION(dataset, sequencing_ann)
        else:
            sequence = seq.fifo_pre_sequencing(dataset)

        if FAMILY_SMD_MAPPING:
            alloc_dict = FAMILY_SMD_MAPPING(dataset, allocation_ann)
            for job in sequence:
                job["alloc_to_smd"] = alloc_dict[job["family"]]

        model = Model(
            sequence=sequence,
            dataset=dataset,
            smd_allocation_function=SMD_ALLOCATION_FUNCTION,
            aoi_allocation_function=AOI_ALLOCATION_FUNCTION,
            post_sequencing_function=SEQUENCING_FUNCTION
            if "post" in SEQUENCING_FUNCTION.__name__
            else None,
            smd_allocation_ann=allocation_ann,
            post_sequencing_ann=sequencing_ann,
            animation=ANIMATION,
            step_by_step_execution=STEP_BY_STEP_EXECUTION,
            freeze_window_at_endsim=FREEZE_WINDOW_AT_ENDSIM,
            tracing=TRACING,
        )

        return model

    for dataset in eval_datasets.items():
        start = timeit.default_timer()
        model = run_simulation(dataset[1])
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
            "\nComputational time: ",
            timeit.default_timer() - start,
            "\n-----------------------------------------------------",
        )

#############################################################################################################################################################################################################
