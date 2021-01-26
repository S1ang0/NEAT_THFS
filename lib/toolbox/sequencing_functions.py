def pre_sequencing_4_input_neurons(dataset, ann, *args, **kwargs):

    """
    Generates an initial job sequence based on the priorization of
    an artificial neural network. The job sequence will be feed to the model.

    Input features:
    ---------------
    * job's due date
    * job's family
    * job's t_smd
    * job's t_aoi
    """

    sequence = []
    for job in dataset.values():
        features = [
            job["scaled due date"],
            job["scaled family"],
            job["scaled t_smd"],
            job["scaled t_aoi"],
        ]
        ann_output = ann.activate(features)
        job[9] = ann_output
        if sequence == []:
            sequence.append(job)
        else:
            sequencing_successful = False
            for previous_job in sequence:
                if job[9] > previous_job[9]:
                    sequence.insert(sequence.index(previous_job), job)
                    sequencing_successful = True
                    break
            if sequencing_successful == False:
                sequence.append(job)
    return sequence


def post_sequencing_5_input_neurons(self, job, smd, *args, **kwargs):

    """
    Inserts job in the buffer of the selected SMD according to the
    priorization of an artificial neural network.

    Input features:
    ---------------
    * job's due date
    * job's family
    * job's t_smd
    * job's t_aoi
    * setup type of allocated SMD
    """

    # 1. Create feature vector for ANN
    features = [
        job.duedate_scaled,
        job.family_scaled,
        job.t_smd_scaled,
        job.t_aoi_scaled,
        smd.setuptype_scaled,
    ]
    # 2. Forward propagation
    job.priority = self.post_sequencing_ann.activate(features)
    # 3. Insert job in sequence
    queue = smd.buffer
    if queue == []:
        queue.append(job)
    else:
        sequencing_successful = False
        for previous_job in queue:
            if job.priority > previous_job.priority:
                queue.insert(queue.index(previous_job), job)
                sequencing_successful = True
                break
        if sequencing_successful == False:
            queue.append(job)


def fifo_pre_sequencing(dataset, *args, **kwargs):

    """
    Generates an initial job sequence based on the first-in-first-out 
    dispatching strategy. The job sequence will be feed to the model.
    """

    sequence = [job for job in dataset.values()]

    return sequence


def fifo_post_sequencing(self, job, smd, *args, **kwargs):

    """
    Inserts job in the buffer of the selected SMD according to the
    first-in-first-out dispatching strategy
    """
    queue = smd.buffer

    queue.append(job)


def lifo_pre_sequencing(dataset, *args, **kwargs):

    """
    Generates an initial job sequence based on the last-in-first-out 
    dispatching strategy. The job sequence will be feed to the model.
    """

    sequence = []

    for job in dataset.values():
        sequence.insert(0, job)

    return sequence


def lifo_post_sequencing(self, job, smd, *args, **kwargs):

    """
    Inserts job in the buffer of the selected SMD according to the
    last-in-first-out dispatching strategy
    """

    queue = smd.buffer

    queue.insert(0, job)


def edd_pre_sequencing(dataset, *args, **kwargs):

    """
    Generates an initial job sequence based on the earliest-due-date 
    dispatching strategy. The job sequence will be feed to the model.
    """

    sequence = []

    for job in dataset.values():
        if sequence == []:
            sequence.append(job)
        else:
            sequencing_successful = False
            for previous_job in sequence:
                if job["due date"] < previous_job["due date"]:
                    sequence.insert(sequence.index(previous_job), job)
                    sequencing_successful = True
                    break
            if sequencing_successful == False:
                sequence.append(job)

    return sequence


def edd_post_sequencing(self, job, smd, *args, **kwargs):

    """
    Inserts job in the buffer of the selected SMD according to the
    earliest-due-date dispatching strategy
    """

    queue = smd.buffer

    if queue == []:
        queue.append(job)
    else:
        sequencing_successful = False
        for previous_job in queue:
            if job.duedate < previous_job.duedate:
                queue.insert(queue.index(previous_job), job)
                sequencing_successful = True
                break
        if sequencing_successful == False:
            queue.append(job)


def spt_pre_sequencing(dataset, *args, **kwargs):

    """
    Generates an initial job sequence based on the shortest-processing-time 
    dispatching strategy. The job sequence will be feed to the model.
    """

    sequence = []

    for job in dataset.values():
        if sequence == []:
            sequence.append(job)
        else:
            sequencing_successful = False
            for previous_job in sequence:
                if (job["t_smd"] + job["t_aoi"]) < (
                    previous_job["t_smd"] + previous_job["t_aoi"]
                ):
                    sequence.insert(sequence.index(previous_job), job)
                    sequencing_successful = True
                    break
            if sequencing_successful == False:
                sequence.append(job)

    return sequence


def spt_post_sequencing(self, job, smd, *args, **kwargs):

    """
    Inserts job in the buffer of the selected SMD according to the
    shortest-processing-time dispatching strategy
    """

    queue = smd.buffer

    if queue == []:
        queue.append(job)
    else:
        sequencing_successful = False
        for previous_job in queue:
            if (job.t_smd + job.t_aoi) < (previous_job.t_smd + previous_job.t_aoi):
                queue.insert(queue.index(previous_job), job)
                sequencing_successful = True
                break
        if sequencing_successful == False:
            queue.append(job)


def edd_x_spt_pre_sequencing(dataset, w_edd=0.5, w_spt=0.5, *args, **kwargs):

    """
    Generates an initial job sequence based on a weighted combination of
    the earliest-due-date and the shortest-processing-time
    dispatching strategy. The job sequence will be feed to the model.
    """

    sequence = [job for job in dataset.values()]

    # sort and rank according to edd
    sequence = sorted(sequence, key=lambda job: job["due date"])

    rank = 1

    prev_job = None
    for job in sequence:
        if prev_job and job["due date"] != prev_job["due date"]:
            rank += 1
        job["rank"] = rank
        prev_job = job

    # sort and rank according to spt and create joint edd_spt_rank
    sequence = sorted(sequence, key=lambda job: job["t_smd"] + job["t_aoi"])
    rank = 1
    prev_job = None
    for job in sequence:
        if (
            prev_job
            and job["t_smd"] + job["t_aoi"] != prev_job["t_smd"] + prev_job["t_aoi"]
        ):
            rank += 1
        job["rank"] = w_edd * job["rank"] + w_spt * rank

    # sort according to joint edd_spt_rank
    sequence = sorted(sequence, key=lambda job: job["rank"])

    return sequence


def edd_x_spt_post_sequencing(self, job, smd, w_edd=0.5, w_spt=0.5):

    """
    Inserts job in the buffer of the selected SMD according to a weighted
    combination of the earliest-due-date and the shortest-processing-time
    dispatching strategy. The job sequence will be feed to the model.
    """

    queue = smd.buffer

    if queue == []:
        queue.append(job)
    else:
        sequencing_successful = False
        for previous_job in queue:
            if ((w_edd * job.duedate) + (w_spt * (job.t_smd + job.t_aoi))) < (
                (w_edd * previous_job.duedate)
                + (w_spt * (previous_job.t_smd + previous_job.t_aoi))
            ):
                queue.insert(queue.index(previous_job), job)
                sequencing_successful = True
                break
        if sequencing_successful == False:
            queue.append(job)
