import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler


def smd_allocation_9_input_neurons(self, job, *args, **kwargs):

    """
    Allocates the job to a SMD according to the decision of an artificial neural network.

    Input features:
    ---------------
    * job's due date
    * job's family
    * job's t_smd
    * job's t_aoi
    * workload of SMD 1,...,4 (4 input neurons)
    * unseized workload
    """

    # 1. Create feature vector for ANN
    job_features = [
        job.duedate_scaled,
        job.family_scaled,
        job.t_smd_scaled,
        job.t_aoi_scaled,
    ]
    smd_features = [[smd.calc_workload()] for smd in self.smds]
    smd_features.append([self.unseized_smd_workload])
    smd_features = MinMaxScaler().fit_transform(smd_features)
    smd_features.tolist()
    smd_features = [i[0] for i in smd_features]
    features = job_features + smd_features
    # 2. Forward propagation
    ann_output = self.smd_allocation_ann.activate(features)
    # 3. Create dictionary for SMD selection
    smd_dict = {i: smd for i, smd in zip(range(len(self.smds)), self.smds)}
    # 4. Choose ann_output with highest value and safe index
    max_index = ann_output.index(max(ann_output))
    # 5. Return selected SMD
    return smd_dict[max_index]


def smd_allocation_random(self, job, *args, **kwargs):

    """
    Allocates the job to a random SMD.
    """

    smd = random.randint(0, len(self.smds) - 1)
    return self.smds[smd]


def smd_allocation_min_workload(self, job, *args, **kwargs):

    """
    Allocates the job to the SMD with the minimal workload.
    """

    return min(self.smds, key=lambda smd: smd.calc_workload())


def smd_allocation_based_on_family_smd_mapping(self, job, *args, **kwargs):

    """
    Allocates the job to a SMD according to a predefined family-smd-mapping.
    """

    smd_dict = {i: smd for i, smd in zip(range(len(self.smds)), self.smds)}
    return smd_dict[job.alloc_to_smd]


def map_families_to_smds_17_input_neurons(dataset, ann, *args, **kwargs):

    """
    Maps families to smds according to the decisions of a artificial neural network.

    Input features:
    ---------------
    * sum of t_smd over all jobs of family
    * sum of t_aoi over all jobs of family
    * sum of due date over all jobs of family
    * min., mean and max. due date of family (3 inputs)
    * 25, 50 and 75 percent quantil of family's due date (3 inputs)
    * workload of SMD 1,...,4 (4 inputs)
    * #families of SMD 1,...,4 (4 inputs)
    """

    # 1. create input data
    df = (
        pd.DataFrame(dataset)
        .transpose()
        .drop(
            [
                "id",
                "scaled due date",
                "scaled family",
                "scaled t_smd",
                "scaled t_aoi",
                "alloc_to_smd",
            ],
            axis=1,
        )
    )
    # 2. sort input data according to job family index in ascending order and save job families as list
    df = df.sort_values("family")
    families = df["family"].drop_duplicates().tolist()
    # 3. create static feature vector for ann
    stats = [
        "sum t_smd",
        "sum t_aoi",
        "sum duedate",
        "min duedate",
        "mean duedate",
        "max duedate",
        "q25 duedate",
        "q50 duedate",
        "q75 duedate",
    ]
    family_features = pd.DataFrame(columns=stats)
    for family in families:
        record = []
        entries_w_family = df["family"] == family
        record.append(df.loc[entries_w_family, "t_smd"].sum())
        record.append(df.loc[entries_w_family, "t_aoi"].sum())
        record.append(df.loc[entries_w_family, "due date"].sum())
        record.append(df.loc[entries_w_family, "due date"].min())
        record.append(df.loc[entries_w_family, "due date"].mean())
        record.append(df.loc[entries_w_family, "due date"].max())
        record.append(df.loc[entries_w_family, "due date"].quantile(0.25))
        record.append(df.loc[entries_w_family, "due date"].median())
        record.append(df.loc[entries_w_family, "due date"].quantile(0.75))
        record = pd.DataFrame([record], columns=stats)
        family_features = family_features.append(record, ignore_index=True)
    # 4. min-max scaling of static features
    scaled_family_features = MinMaxScaler().fit_transform(family_features).tolist()
    # 5. initialize dynamic feature vectoor for  ann
    workload_features = [0 for i in range(4)]  # sum(t_smd 1,...,4)
    scaled_workload_features = [0 for i in range(4)]
    n_family_features = [0 for i in range(4)]  # n_families(SMD 1,...,4)
    scaled_n_family_features = [0 for i in range(4)]
    # 6. create allocation dictionary for mapping job families to SMDs
    alloc_dict = {}
    for fam, fam_feat, t_smd in zip(
        families, scaled_family_features, family_features["sum t_smd"]
    ):
        # 6.1 create feature vector
        features = fam_feat + scaled_workload_features + scaled_n_family_features
        # 6.2 forward propagation
        ann_output = ann.activate(features)
        # 6.3 find output neuron with max. activation
        max_index = ann_output.index(max(ann_output))
        # 6.4 update allocation dictionary
        alloc_dict[fam] = max_index
        # 6.5 update workload on SMDs
        workload_features[max_index] += t_smd
        scaled_workload_features = [[i] for i in workload_features]
        scaled_workload_features = (
            MinMaxScaler().fit_transform(scaled_workload_features).tolist()
        )
        scaled_workload_features = [i[0] for i in scaled_workload_features]
        # 6.6 update number of families on SMDs
        n_family_features[max_index] += 1
        scaled_n_family_features = [[i] for i in n_family_features]
        scaled_n_family_features = (
            MinMaxScaler().fit_transform(scaled_n_family_features).tolist()
        )
        scaled_n_family_features = [i[0] for i in scaled_n_family_features]
    # 7 return allocation dictionary
    return alloc_dict


def map_families_to_smds_equal(dataset, *args, **kwargs):

    """
    Maps job families evenly to SMD
    """

    # 1. create input data
    df = (
        pd.DataFrame(dataset)
        .transpose()
        .drop(
            [
                "id",
                "scaled due date",
                "scaled family",
                "scaled t_smd",
                "scaled t_aoi",
                "alloc_to_smd",
            ],
            axis=1,
        )
    )
    # 2. sort input data according to job family index in ascending order and save job families as list
    df = df.sort_values("family")
    families = df["family"].drop_duplicates().tolist()
    # 3. create allocation dictionary for mapping job families to SMDs
    alloc_dict = {}
    smd = 0
    for fam in families:
        alloc_dict[fam] = smd
        smd += 1 if smd < 3 else -3
    # 4 return allocation dictionary
    return alloc_dict


def map_families_to_smds_random(dataset):

    """
    Maps job families randomly to SMDs
    """

    # 1. create input data
    df = (
        pd.DataFrame(dataset)
        .transpose()
        .drop(
            [
                "id",
                "scaled due date",
                "scaled family",
                "scaled t_smd",
                "scaled t_aoi",
                "alloc_to_smd",
            ],
            axis=1,
        )
    )
    # 2. sort input data according to job family index in ascending order and save job families as list
    df = df.sort_values("family")
    families = df["family"].drop_duplicates().tolist()
    # 3. create allocation dictionary for mapping job families to SMDs
    alloc_dict = {fam: random.randint(0, 3) for fam in families}
    # 4 return allocation dictionary
    return alloc_dict


def aoi_allocation_random(self, job, *args, **kwargs):

    """
    Allocates the job to a random AOI
    """

    aoi = random.randint(0, len(self.aois) - 1)
    return self.aois[aoi]


def aoi_allocation_min_workload(self, job, *args, **kwargs):

    """
    Allocates the job to the AOI with the minimal workload
    """

    return min(self.aois, key=lambda aoi: aoi.calc_workload())
