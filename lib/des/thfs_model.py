#############################################################################################################################################################################################################
### PRINTED CIRCUIT BOARDS PRODUCTION - A TWO-STAGE HYBRID FLOWSHOP SCHEDULING PROBLEM ######################################################################################################################
#############################################################################################################################################################################################################
### Author: Sebastian Lang (sebastian.lang@ovgu.de) #########################################################################################################################################################
#############################################################################################################################################################################################################

### Libraries ###############################################################################################################################################################################################

import salabim as sim
import pandas as pd
import math

#############################################################################################################################################################################################################

### Global Variables ########################################################################################################################################################################################

# Visualization
ENV_W = 1200
ENV_H = 600
REF_WIDTH = 110
GLOB_BUFFER_WIDTH = REF_WIDTH
GLOB_PROCESS_WIDTH = REF_WIDTH
REF_HEIGHT = 60
GLOB_SOURCE_DRAIN_RADIUS = (REF_HEIGHT) / 2
GLOB_BUFFER_HEIGHT = REF_HEIGHT - 20
GLOB_PROCESS_HEIGHT = REF_HEIGHT
GLOB_FONTSIZE = 12
X_0 = 50
Y_0 = 300
Y_GLOB_SOURCE_DRAIN = Y_0 + GLOB_SOURCE_DRAIN_RADIUS
Y_GLOB_BUFFER = Y_0 + ((REF_HEIGHT - GLOB_BUFFER_HEIGHT) / 2)
Y_GLOB_PROCESS = Y_0 + ((REF_HEIGHT - GLOB_PROCESS_HEIGHT) / 2)

#############################################################################################################################################################################################################

### Modeling Objects ########################################################################################################################################################################################


class Job(sim.Component):
    def setup(
        self,
        model,
        job_id,
        duedate,
        family,
        t_smd,
        t_aoi,
        duedate_scaled,
        family_scaled,
        t_smd_scaled,
        t_aoi_scaled,
        alloc_to_smd,
    ):

        # model
        self.model = model

        # initial attributes
        self.id = job_id
        self.duedate = duedate
        self.family = family
        self.t_smd = t_smd
        self.t_aoi = t_aoi
        self.duedate_scaled = duedate_scaled
        self.family_scaled = family_scaled
        self.t_smd_scaled = t_smd_scaled
        self.t_aoi_scaled = t_aoi_scaled
        self.alloc_to_smd = alloc_to_smd

        # flags
        self.selected_smd = None
        self.selected_aoi = None

        # visualization
        if self.model.env.animate() == True:
            self.img = sim.AnimateCircle(
                radius=10,
                x=X_0,
                y=Y_GLOB_SOURCE_DRAIN,
                fillcolor="limegreen",
                linecolor="black",
                text=str(self.id),
                fontsize=15,
                textcolor="black",
                parent=self,
                screen_coordinates=True,
            )

    def process(self):

        # enter job buffer
        self.enter(self.model.job_buffer)
        if self.model.env.animate() == True:
            self.model.job_buffer.set_job_pos()
        yield self.passivate()

        # select and enter SMD buffer
        self.model.job_buffer.remove(self)
        if self.model.env.animate() == True:
            self.model.job_buffer.set_job_pos()
        self.selected_smd = self.model.smd_allocation_method(job=self)
        self.model.unseized_smd_workload -= self.t_smd
        self.selected_smd.buffer.workload += self.t_smd
        if (
            len(self.selected_smd.buffer) == 0
            and self.family != self.selected_smd.setuptype
        ) or self.family != self.selected_smd.buffer[-1].family:
            self.selected_smd.buffer.workload += 65
        else:
            self.selected_smd.buffer.workload += 20
        if self.model.post_sequencing_function:
            self.model.post_sequencing_method(job=self, smd=self.selected_smd)
        else:
            self.enter(self.selected_smd.buffer)
        if self.model.env.animate() == True:
            self.selected_smd.buffer.set_job_pos()
        if self.selected_smd.ispassive() and self.selected_smd.state != "waiting":
            self.selected_smd.activate()
        yield self.passivate()

        # select and enter AOI buffer
        self.selected_aoi = self.model.aoi_allocation_method(job=self)
        self.enter(self.selected_aoi.buffer)
        self.selected_aoi.buffer.workload += 25 + self.t_aoi
        if self.model.env.animate() == True:
            self.selected_aoi.buffer.set_job_pos()
        if self.selected_aoi.ispassive():
            self.selected_aoi.activate()
        yield self.passivate()

        # calculate objectives and destroy job
        self.model.jobs_processed += 1
        if self.model.env.now() > self.duedate:
            self.model.total_tardiness += self.model.env.now() - self.duedate
            if self.model.env.animate() == True:
                self.model.info_tardiness.text = "Total Tardiness: " + str(
                    self.model.total_tardiness
                )
        if self.model.jobs_processed == self.model.num_jobs:
            self.model.makespan = self.model.env.now()
            # Workaround for a bug in NEAT:
            # * NEAT outputs an assertion error if the fitness is not of type float or integer
            # * Due to the use of pandas, some KPIs (e.g. the Total Tardiness) have a numpy data type
            # * Therefore, we cast all KPIs that could be relevant to measure the fitness to float or integer
            self.model.makespan = float(self.model.makespan)
            self.model.total_tardiness = float(self.model.total_tardiness)
            self.model.num_major_setups = int(self.model.num_major_setups)
            if self.model.env.animate() == True:
                self.model.info_makespan.text = "Makespan: " + str(self.model.makespan)
                if self.model.freeze_window_at_endsim:
                    self.model.env.an_menu()
        if self.model.env.animate() == True:
            self.img.x = self.model.drain.img[0].x
            self.img.y = self.model.drain.img[0].y
            yield self.hold(0)
        del self


class Source(sim.Component):
    def setup(
        self,
        model,
        img_w=GLOB_SOURCE_DRAIN_RADIUS,
        img_h=GLOB_SOURCE_DRAIN_RADIUS,
        img_x=X_0,
        img_y=Y_GLOB_SOURCE_DRAIN,
    ):

        # model
        self.model = model

        # visualization
        if self.model.env.animate() == True:
            self.img_w = img_w
            self.img_h = img_h
            self.img_x = img_x
            self.img_y = img_y
            self.img = [
                sim.AnimateCircle(
                    radius=img_w,
                    x=img_x,
                    y=img_y,
                    fillcolor="white",
                    linecolor="black",
                    linewidth=2,
                    layer=2,
                    arg=(img_x + img_w, img_y + img_h),
                    screen_coordinates=True,
                ),
                sim.AnimateCircle(
                    radius=0.3 * img_w,
                    x=img_x,
                    y=img_y,
                    fillcolor="black",
                    linecolor="black",
                    layer=1,
                    screen_coordinates=True,
                ),
            ]

    def process(self):

        # generate jobs
        for job in self.model.sequence:
            Job(
                model=self.model,
                job_id=job["id"],
                duedate=job["due date"],
                family=job["family"],
                t_smd=job["t_smd"],
                t_aoi=job["t_aoi"],
                duedate_scaled=job["scaled due date"],
                family_scaled=job["scaled family"],
                t_smd_scaled=job["scaled t_smd"],
                t_aoi_scaled=job["scaled t_aoi"],
                alloc_to_smd=job["alloc_to_smd"],
            )
            yield self.hold(0)

        # step-by-step mode
        if self.model.step_by_step_execution:
            self.model.env.an_menu()

        # activate jobs
        for job in self.model.job_buffer:
            job.activate()
            yield self.hold(0)


class Queue(sim.Queue):
    def setup(
        self,
        model,
        predecessors,
        img_w=None,
        img_h=None,
        img_x=None,
        img_y=None,
        img_slots=None,
    ):

        # model
        self.model = model

        # initial attributes
        self.predecessors = predecessors
        self.img_w = img_w
        self.img_h = img_h
        self.img_x = img_x
        self.img_y = img_y
        self.img_slots = img_slots

        # flags
        self.workload = 0

        # visualization
        if self.model.env.animate() == True:
            self.img = [
                [
                    sim.AnimateRectangle(
                        spec=(0, 0, (self.img_w / self.img_slots), self.img_h),
                        x=self.img_x + i * (self.img_w / self.img_slots),
                        y=self.img_y,
                        fillcolor="white",
                        linecolor="white",
                        linewidth=1,
                        layer=2,
                        arg=(
                            self.img_x
                            + (i * (self.img_w / self.img_slots))
                            + (self.img_w / (self.img_slots * 2)),
                            self.img_y + (self.img_h / 2),
                        ),
                        screen_coordinates=True,
                    )
                    for i in range(self.img_slots)
                ],
                sim.AnimateRectangle(
                    spec=(0, 0, self.img_w, self.img_h),
                    x=self.img_x,
                    y=self.img_y,
                    fillcolor="white",
                    linecolor="black",
                    linewidth=2,
                    layer=1,
                    screen_coordinates=True,
                ),
            ]
            self.predecessor_connections = [
                sim.AnimateLine(
                    spec=(
                        predecessor.img_x + predecessor.img_w,
                        predecessor.img_y
                        if predecessor.__class__.__name__ == "Source"
                        else predecessor.img_y + predecessor.img_h / 2,
                        self.img_x,
                        self.img_y + self.img_h / 2,
                    ),
                    linecolor="black",
                    linewidth=2,
                    layer=2,
                    screen_coordinates=True,
                )
                for predecessor in self.predecessors
            ]
            self.info = sim.AnimateText(
                text="# products: 0",
                x=self.img[0][0].x,
                y=self.img[0][0].y - 20,
                fontsize=18,
                textcolor="black",
                screen_coordinates=True,
            )

    def set_job_pos(self):

        if len(self) == 0:
            self.info.text = "# products: 0"
        else:
            for job, spot in zip(self, reversed(self.img[0])):
                job.img.visible = True
                job.img.x = spot.arg[0]
                job.img.y = spot.arg[1]
                self.info.text = "# products: " + str(len(self))
                if len(self) >= len(self.img[0]):
                    for i in range(len(self.img[0]), len(self)):
                        self[i].img.visible = False
                        self[i].img.x = self.img[0][0].arg[0]
                        self[i].img.y = self.img[0][0].arg[1]


class SMD(sim.Component):
    def setup(
        self,
        model,
        buffer,
        img_x=None,
        img_y=None,
        img_w=GLOB_PROCESS_WIDTH,
        img_h=GLOB_PROCESS_HEIGHT,
        info_x=None,
        info_y=None,
    ):

        # model
        self.model = model

        # initial attributes
        self.buffer = buffer
        self.img_x = img_x
        self.img_y = img_y
        self.img_w = img_w
        self.img_h = img_h
        self.info_x = info_x
        self.info_y = info_y

        # flags
        self.job = None
        self.setuptype = 0
        self.setuptype_scaled = 0
        self.setup_to = 0
        self.state = "idle"

        # visualization
        if model.env.animate() == True:
            self.img = sim.AnimatePolygon(
                spec=(
                    0,
                    0,
                    self.img_w - (self.img_w / 300) * 50,
                    0,
                    self.img_w,
                    self.img_h / 2,
                    self.img_w - (self.img_w / 300) * 50,
                    self.img_h,
                    0,
                    self.img_h,
                    (self.img_w / 300) * 50,
                    self.img_h / 2,
                    0,
                    0,
                ),
                x=self.img_x,
                y=self.img_y,
                fillcolor="white",
                linecolor="black",
                linewidth=2,
                text=self.name() + "\n\nsetuptype = 0\nidle",
                fontsize=GLOB_FONTSIZE,
                textcolor="black",
                layer=1,
                screen_coordinates=True,
            )
            self.buffer_connection = sim.AnimateLine(
                spec=(
                    self.buffer.img_x + self.buffer.img_w,
                    self.buffer.img_y + self.buffer.img_h / 2,
                    self.img_x + 50,
                    self.img_y + self.img_h / 2,
                ),
                linecolor="black",
                linewidth=2,
                layer=2,
                screen_coordinates=True,
            )

    def process(self):

        while True:

            # idle state
            if len(self.buffer) == 0:
                self.state = "idle"
                if (
                    self.setuptype != 0
                    and self.model.waiting_for_setuptype[self.setuptype]
                ):
                    released_setuptype = self.setuptype
                    self.setuptype = 0
                    self.model.waiting_for_setuptype[released_setuptype].pop(
                        0
                    ).activate()
                if self.model.env.animate() == True:
                    self.set_status(status=self.state)
                yield self.passivate()

            # pick next job from SMD buffer
            self.job = self.buffer.pop()
            self.buffer.workload -= self.job.t_smd
            if self.model.env.animate() == True:
                self.buffer.set_job_pos()
                self.job.img.x = self.img_x + (self.img_w / 2)
                self.job.img.y = self.img_y + (self.img_h / 2)

            # setup state
            if self.setuptype == self.job.family:
                self.state = "minor setup"
                self.buffer.workload -= 20
                if self.model.env.animate() == True:
                    self.set_status(status=self.state)
                yield self.hold(20)
            else:
                #####################################################################################################
                # the following routine ensures that a specific setup kit is only mounted on SMD at the time
                # (1) release current setup kit
                # (2) reassign current setup kit, if other machine waits on it
                # (3) check if other machine is right now using the demanding setup kit
                #     (3.1) if true: put SMD to waiting line for the demanded setup kit and passivate SMD
                #     (3.2) if false: eventually release setup kit from other SMD in idle state
                released_setuptype = self.setuptype  # (1)
                self.setuptype = 0  # (1)
                if (
                    released_setuptype != 0
                    and self.model.waiting_for_setuptype[released_setuptype]
                ):  # (2)
                    self.model.waiting_for_setuptype[released_setuptype].pop(
                        0
                    ).activate()  # (2)
                for smd in self.model.smds:  # (3)
                    if (
                        smd.setuptype == self.job.family
                        or smd.setup_to == self.job.family
                    ):  # (3)
                        if smd.job is not None or len(smd.buffer) > 0:  # (3.1)
                            self.model.waiting_for_setuptype[self.job.family].append(
                                self
                            )  # (3.1)
                            self.state = "waiting"  # (3.1)
                            if self.model.env.animate() == True:  # (3.1)
                                self.set_status(status=self.state)  # (3.1)
                            yield self.passivate()  # (3.1)
                            break  # (3.1)
                        else:  # (3.2)
                            smd.setuptype = 0  # (3.2)
                            break  # (3.2)
                #####################################################################################################
                self.state = "major setup"
                self.buffer.workload -= 65
                self.model.num_major_setups += 1
                if self.model.env.animate() == True:
                    self.set_status(status=self.state)
                    self.model.info_setups.text = "Major Setups: " + str(
                        self.model.num_major_setups
                    )
                self.setup_to = self.job.family
                # for-loop to check setup violations (more than one machine is mounted with the same setup kit) due setup process
                for smd in self.model.smds:
                    if self.name != smd.name:
                        if (
                            self.setup_to == smd.setuptype
                            or self.setup_to == smd.setup_to
                        ):
                            self.model.n_setup_violations += 1
                yield self.hold(65)
                self.setuptype = self.job.family
                self.setup_to = 0
                # for-loop to check setup violations (more than one machine is mounted with the same setup kit) due setup process
                for smd in self.model.smds:
                    if self.name != smd.name:
                        if (
                            self.setuptype == smd.setuptype
                            or self.setuptype == smd.setup_to
                        ):
                            self.model.n_setup_violations += 1

            # active state
            self.state = "active"
            if self.model.env.animate() == True:
                self.set_status(status=self.state)
            yield self.hold(self.job.t_smd)
            self.job.activate()
            self.job = None

    def set_status(self, status):

        dict_status = {
            "idle": "white",
            "active": "lime",
            "minor setup": "yellow",
            "major setup": "tomato",
            "waiting": "violet",
        }

        self.img.fillcolor = dict_status.get(status)
        self.img.text = (
            self.name() + "\n\nsetuptype = " + str(self.setuptype) + "\n" + status
        )

    def calc_workload(self, called_from=None):

        if self.state == "idle":
            return self.buffer.workload
        elif self.state == "active":
            return self.remaining_duration() + self.buffer.workload
        else:
            return self.remaining_duration() + self.job.t_smd + self.buffer.workload


class AOI(sim.Component):
    def setup(
        self,
        model,
        buffer,
        img_x=None,
        img_y=None,
        img_w=GLOB_PROCESS_WIDTH,
        img_h=GLOB_PROCESS_HEIGHT,
    ):

        # model
        self.model = model

        # initial attributes
        self.buffer = buffer
        self.img_x = img_x
        self.img_y = img_y
        self.img_w = img_w
        self.img_h = img_h

        # flags
        self.job = None
        self.state = "idle"

        # visualization
        if self.model.env.animate() == True:
            self.img = sim.AnimatePolygon(
                spec=(
                    0,
                    0,
                    self.img_w - (self.img_w / 300) * 50,
                    0,
                    self.img_w,
                    self.img_h / 2,
                    self.img_w - (self.img_w / 300) * 50,
                    self.img_h,
                    0,
                    self.img_h,
                    (self.img_w / 300) * 50,
                    self.img_h / 2,
                    0,
                    0,
                ),
                x=self.img_x,
                y=self.img_y,
                fillcolor="white",
                linecolor="black",
                linewidth=2,
                text=self.name() + "\n\nidle",
                fontsize=GLOB_FONTSIZE,
                textcolor="black",
                layer=1,
                screen_coordinates=True,
            )
            self.buffer_connection = sim.AnimateLine(
                spec=(
                    self.buffer.img_x + self.buffer.img_w,
                    self.buffer.img_y + self.buffer.img_h / 2,
                    self.img_x + 50,
                    self.img_y + self.img_h / 2,
                ),
                linecolor="black",
                linewidth=2,
                layer=2,
                screen_coordinates=True,
            )

    def process(self):

        while True:

            # idle state
            if len(self.buffer) == 0:
                self.state = "idle"
                if self.model.env.animate() == True:
                    self.set_status(status=self.state)
                yield self.passivate()

            # pick next job from AOIbuffer
            self.job = self.buffer.pop()
            self.buffer.workload -= self.job.t_aoi + 25
            if self.model.env.animate() == True:
                self.buffer.set_job_pos()
                self.job.img.x = self.img_x + (self.img_w / 2)
                self.job.img.y = self.img_y + (self.img_h / 2)

            # setup state
            self.state = "setting-up"
            if self.model.env.animate() == True:
                self.set_status(status=self.state)
            yield self.hold(25)

            # active state
            self.state = "active"
            if self.model.env.animate() == True:
                self.set_status(status=self.state)
            yield self.hold(self.job.t_aoi)
            self.job.activate()
            self.job = None

    def set_status(self, status):

        dict_status = {"idle": "white", "active": "lime", "setting-up": "yellow"}

        self.img.fillcolor = dict_status.get(status)
        self.img.text = self.name() + "\n\n" + status

    def calc_workload(self):

        if self.state == "idle":
            return self.buffer.workload
        elif self.state == "active":
            return self.remaining_duration() + self.buffer.workload
        else:
            return self.remaining_duration() + self.job.t_aoi + self.buffer.workload


class Drain:
    def __init__(
        self,
        model,
        predecessors,
        img_x=None,
        img_y=None,
        img_w=GLOB_SOURCE_DRAIN_RADIUS,
        img_h=GLOB_SOURCE_DRAIN_RADIUS,
    ):

        # initial attributes
        self.model = model
        self.predecessors = predecessors
        self.img_x = img_x
        self.img_y = img_y
        self.img_w = img_w
        self.img_h = img_h

        # visualization
        self.img = [
            sim.AnimateCircle(
                radius=img_w,
                x=self.img_x,
                y=self.img_y,
                fillcolor="white",
                linecolor="black",
                linewidth=2,
                layer=1,
                screen_coordinates=True,
            ),
            sim.AnimateLine(
                spec=(
                    img_w * math.cos(math.radians(45)) * (-1),
                    img_w * math.sin(math.radians(45)) * (-1),
                    img_w * math.cos(math.radians(45)),
                    img_w * math.sin(math.radians(45)),
                ),
                x=self.img_x,
                y=self.img_y,
                linecolor="black",
                linewidth=2,
                layer=1,
                arg=(self.img_x + img_w, self.img_y + img_w),
                screen_coordinates=True,
            ),
            sim.AnimateLine(
                spec=(
                    img_w * math.cos(math.radians(45)) * (-1),
                    img_w * math.sin(math.radians(45)),
                    img_w * math.cos(math.radians(45)),
                    img_w * math.sin(math.radians(45)) * (-1),
                ),
                x=self.img_x,
                y=self.img_y,
                linecolor="black",
                linewidth=2,
                layer=1,
                screen_coordinates=True,
            ),
        ]
        self.predecessor_connections = [
            sim.AnimateLine(
                spec=(
                    predecessor.img_x + predecessor.img_w,
                    predecessor.img_y + predecessor.img_h / 2,
                    self.img_x - self.img_w,
                    self.img_y,
                ),
                linecolor="black",
                linewidth=2,
                layer=2,
                screen_coordinates=True,
            )
            for predecessor in self.predecessors
        ]


#############################################################################################################################################################################################################

### Simulation Model ########################################################################################################################################################################################


class Model:
    def __init__(
        self,
        sequence,
        dataset,
        smd_allocation_function,
        aoi_allocation_function,
        post_sequencing_function=None,
        smd_allocation_ann=None,
        post_sequencing_ann=None,
        animation=False,
        step_by_step_execution=False,
        freeze_window_at_endsim=False,
        tracing=False,
    ):

        # input data
        self.sequence = sequence
        self.dataset = dataset

        # allocation functions
        if smd_allocation_function is not None:
            setattr(
                self,
                "smd_allocation_method",
                smd_allocation_function.__get__(self, self.__class__),
            )
            self.smd_allocation_function = True
        else:
            self.smd_allocation_function = False

        if aoi_allocation_function is not None:
            setattr(
                self,
                "aoi_allocation_method",
                aoi_allocation_function.__get__(self, self.__class__),
            )
            self.aoi_allocation_function = True
        else:
            self.aoi_allocation_function = False

        # sequencing function
        if post_sequencing_function is not None:
            setattr(
                self,
                "post_sequencing_method",
                post_sequencing_function.__get__(self, self.__class__),
            )
            self.post_sequencing_function = True
        else:
            self.post_sequencing_function = False

        # artificial neural networks
        self.smd_allocation_ann = smd_allocation_ann
        self.post_sequencing_ann = post_sequencing_ann

        # visualization
        self.animation = animation
        self.step_by_step_execution = step_by_step_execution
        self.freeze_window_at_endsim = freeze_window_at_endsim
        self.tracing = tracing

        # component lists
        self.smds = []
        self.aois = []

        # create queue dictionary for smds waiting for setuptype
        setuptypes = (
            pd.DataFrame(dataset)
            .transpose()["family"]
            .sort_values()
            .drop_duplicates()
            .tolist()
        )
        self.waiting_for_setuptype = {setuptype: [] for setuptype in setuptypes}

        # evaluation
        self.num_jobs = len(self.sequence)
        self.jobs_processed = 0
        self.total_workload = sum(job["t_smd"] for job in sequence)
        self.unseized_smd_workload = self.total_workload
        self.makespan = 0
        self.total_tardiness = 0
        self.num_major_setups = 0
        self.n_setup_violations = 0

        # model creation and execution
        self.define_environment()
        self.create_model()
        self.run_model()

    def smd_allocation_method(self):
        pass

    def aoi_allocation_method(self):
        pass

    def post_sequencing_method(self):
        pass

    def define_environment(self):

        self.env = sim.Environment(trace=self.tracing, time_unit="minutes")
        self.env.modelname(
            "Circuit Board Production: A Two-Stage Hybrid Flow Shop Scheduling Problem"
        )
        self.env.animation_parameters(
            animate=self.animation,
            synced=False,
            width=ENV_W,
            height=ENV_H,
            background_color="60%gray",
        )
        if self.animation == True:
            self.info_makespan = sim.AnimateText(
                text="Makespan: ", x=10, y=420, fontsize=GLOB_FONTSIZE
            )
            self.info_tardiness = sim.AnimateText(
                text="Total Tardiness: 0", x=10, y=390, fontsize=GLOB_FONTSIZE
            )
            self.info_setups = sim.AnimateText(
                text="Major Setups: 0", x=10, y=360, fontsize=GLOB_FONTSIZE
            )

    def create_model(self):

        # source
        self.source = Source(name="Source", model=self,)

        # job_buffer
        self.job_buffer = Queue(
            name="Job Buffer",
            model=self,
            predecessors=[self.source],
            img_w=200,
            img_h=GLOB_BUFFER_HEIGHT,
            img_x=X_0 + GLOB_SOURCE_DRAIN_RADIUS + 50,
            img_y=Y_GLOB_BUFFER,
            img_slots=5,
        )

        # smd_lines
        self.smd_buffer_0 = Queue(
            name="Buffer SMD 01",
            model=self,
            predecessors=[self.job_buffer],
            img_w=GLOB_BUFFER_WIDTH,
            img_h=GLOB_BUFFER_HEIGHT,
            img_x=self.job_buffer.img_x + self.job_buffer.img_w + 50,
            img_y=Y_GLOB_BUFFER + 100,
            img_slots=5,
        )

        self.smd_0 = SMD(
            name="SMD 01",
            model=self,
            buffer=self.smd_buffer_0,
            img_x=self.smd_buffer_0.img_x + self.smd_buffer_0.img_w + 50,
            img_y=self.smd_buffer_0.img_y - 10,
            info_x=10,
            info_y=100,
        )
        self.smds.append(self.smd_0)

        self.smd_buffer_1 = Queue(
            name="Buffer SMD 02",
            model=self,
            predecessors=[self.job_buffer],
            img_w=GLOB_BUFFER_WIDTH,
            img_h=GLOB_BUFFER_HEIGHT,
            img_x=self.job_buffer.img_x + self.job_buffer.img_w + 50,
            img_y=Y_GLOB_BUFFER + 35,
            img_slots=5,
        )

        self.smd_1 = SMD(
            name="SMD 02",
            model=self,
            buffer=self.smd_buffer_1,
            img_x=self.smd_buffer_1.img_x + self.smd_buffer_1.img_w + 50,
            img_y=self.smd_buffer_1.img_y - 10,
            info_x=10,
            info_y=70,
        )
        self.smds.append(self.smd_1)

        self.smd_buffer_2 = Queue(
            name="Buffer SMD 03",
            model=self,
            predecessors=[self.job_buffer],
            img_w=GLOB_BUFFER_WIDTH,
            img_h=GLOB_BUFFER_HEIGHT,
            img_x=self.job_buffer.img_x + self.job_buffer.img_w + 50,
            img_y=Y_GLOB_BUFFER - 35,
            img_slots=5,
        )

        self.smd_2 = SMD(
            name="SMD 03",
            model=self,
            buffer=self.smd_buffer_2,
            img_x=self.smd_buffer_2.img_x + self.smd_buffer_2.img_w + 50,
            img_y=self.smd_buffer_2.img_y - 10,
            info_x=10,
            info_y=40,
        )
        self.smds.append(self.smd_2)

        self.smd_buffer_3 = Queue(
            name="Buffer SMD 04",
            model=self,
            predecessors=[self.job_buffer],
            img_w=GLOB_BUFFER_WIDTH,
            img_h=GLOB_BUFFER_HEIGHT,
            img_x=self.job_buffer.img_x + self.job_buffer.img_w + 50,
            img_y=Y_GLOB_BUFFER - 100,
            img_slots=5,
        )

        self.smd_3 = SMD(
            name="SMD 04",
            model=self,
            buffer=self.smd_buffer_3,
            img_x=self.smd_buffer_3.img_x + self.smd_buffer_3.img_w + 50,
            img_y=self.smd_buffer_3.img_y - 10,
            info_x=10,
            info_y=10,
        )
        self.smds.append(self.smd_3)

        # aoi_lines
        self.aoi_buffer_0 = Queue(
            name="Buffer AOI 01",
            model=self,
            predecessors=[self.smd_0, self.smd_1, self.smd_2, self.smd_3],
            img_w=GLOB_BUFFER_WIDTH,
            img_h=GLOB_BUFFER_HEIGHT,
            img_x=self.smd_0.img_x + GLOB_PROCESS_WIDTH + 50,
            img_y=Y_GLOB_BUFFER + (2 * GLOB_BUFFER_HEIGHT) + 60,
            img_slots=5,
        )

        self.aoi_0 = AOI(
            name="AOI 01",
            model=self,
            buffer=self.aoi_buffer_0,
            img_x=self.aoi_buffer_0.img_x + self.aoi_buffer_0.img_w + 50,
            img_y=self.aoi_buffer_0.img_y - 10,
        )
        self.aois.append(self.aoi_0)

        self.aoi_buffer_1 = Queue(
            name="Buffer AOI 02",
            model=self,
            predecessors=[self.smd_0, self.smd_1, self.smd_2, self.smd_3],
            img_w=GLOB_BUFFER_WIDTH,
            img_h=GLOB_BUFFER_HEIGHT,
            img_x=self.smd_0.img_x + GLOB_PROCESS_WIDTH + 50,
            img_y=Y_GLOB_BUFFER + GLOB_BUFFER_HEIGHT + 30,
            img_slots=5,
        )

        self.aoi_1 = AOI(
            name="AOI 02",
            model=self,
            buffer=self.aoi_buffer_1,
            img_x=self.aoi_buffer_1.img_x + self.aoi_buffer_1.img_w + 50,
            img_y=self.aoi_buffer_1.img_y - 10,
        )
        self.aois.append(self.aoi_1)

        self.aoi_buffer_2 = Queue(
            name="Buffer AOI 03",
            model=self,
            predecessors=[self.smd_0, self.smd_1, self.smd_2, self.smd_3],
            img_w=GLOB_BUFFER_WIDTH,
            img_h=GLOB_BUFFER_HEIGHT,
            img_x=self.smd_0.img_x + GLOB_PROCESS_WIDTH + 50,
            img_y=Y_GLOB_BUFFER,
            img_slots=5,
        )

        self.aoi_2 = AOI(
            name="AOI 03",
            model=self,
            buffer=self.aoi_buffer_2,
            img_x=self.aoi_buffer_2.img_x + self.aoi_buffer_2.img_w + 50,
            img_y=self.aoi_buffer_2.img_y - 10,
        )
        self.aois.append(self.aoi_2)

        self.aoi_buffer_3 = Queue(
            name="Buffer AOI 04",
            model=self,
            predecessors=[self.smd_0, self.smd_1, self.smd_2, self.smd_3],
            img_w=GLOB_BUFFER_WIDTH,
            img_h=GLOB_BUFFER_HEIGHT,
            img_x=self.smd_0.img_x + GLOB_PROCESS_WIDTH + 50,
            img_y=Y_GLOB_BUFFER - GLOB_BUFFER_HEIGHT - 30,
            img_slots=5,
        )

        self.aoi_3 = AOI(
            name="AOI 04",
            model=self,
            buffer=self.aoi_buffer_3,
            img_x=self.aoi_buffer_3.img_x + self.aoi_buffer_3.img_w + 50,
            img_y=self.aoi_buffer_3.img_y - 10,
        )
        self.aois.append(self.aoi_3)

        self.aoi_buffer_4 = Queue(
            name="Buffer AOI 05",
            model=self,
            predecessors=[self.smd_0, self.smd_1, self.smd_2, self.smd_3],
            img_w=GLOB_BUFFER_WIDTH,
            img_h=GLOB_BUFFER_HEIGHT,
            img_x=self.smd_0.img_x + GLOB_PROCESS_WIDTH + 50,
            img_y=Y_GLOB_BUFFER - (2 * GLOB_BUFFER_HEIGHT) - 60,
            img_slots=5,
        )

        self.aoi_4 = AOI(
            name="AOI 05",
            model=self,
            buffer=self.aoi_buffer_4,
            img_x=self.aoi_buffer_4.img_x + self.aoi_buffer_4.img_w + 50,
            img_y=self.aoi_buffer_4.img_y - 10,
        )
        self.aois.append(self.aoi_4)

        if self.env.animate() == True:
            self.drain = Drain(
                model=self,
                predecessors=[
                    self.aoi_0,
                    self.aoi_1,
                    self.aoi_2,
                    self.aoi_3,
                    self.aoi_4,
                ],
                img_x=self.aoi_0.img_x + GLOB_PROCESS_WIDTH + 75,
                img_y=Y_GLOB_SOURCE_DRAIN,
            )

    def run_model(self):
        self.env.run()


#############################################################################################################################################################################################################
