"""
    This file to run trajectory, hence handling OpenFOAM files and executing them in machine

    called in : replay_buffer.py
"""

import _thread
import os
import queue
import subprocess
import time
from glob import glob

import torch

import numpy as np


class env:
    """
        This Class is to run trajectory, hence handling OpenFOAM files and executing them in machine
    """
    def __init__(self, n_worker, buffer_size, control_between):#, simulation_re):
        """

        Args:
            n_worker: no of trajectories at the same time (worker)
            buffer_size: total number of trajectories
            contol_between: random starting point range of action in trajectory
        """
        self.n_worker = n_worker
        self.buffer_size = buffer_size
        self.control_between = control_between
        #self.simulation_re = simulation_re

    def write_jobfile(self, core_count, job_name, file, job_dir):
        with open(f'{job_dir}/jobscript.sh', 'w') as rsh:
            rsh.write(f"""#!/bin/bash -l        
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --job-name={job_name}
#SBATCH --ntasks-per-node={core_count}

module load singularity/3.6.0rc2
module load mpi/openmpi/4.0.1/cuda_aware_gcc_6.3.0

cd {job_dir}

./Allrun.singularity

touch finished.txt""")

        os.system(f"chmod +x {job_dir}/jobscript.sh")

    def rand_n_to_contol(self, n):
        """
        To get the random number from the range -> .2f%

        Args:
            n: number of random sampled number (in this case n=1)

        Returns: random number

        """
        np.random.seed()
        n_rand = np.random.uniform(self.control_between[0], self.control_between[1], n)
        n_rand = np.round(n_rand, 2)
        return n_rand

    def remove_tensor_element(self, tensor, indices):
        mask = torch.ones(tensor.numel(), dtype=torch.bool)
        mask[indices] = False
        return tensor[mask]

    def get_snapshot_List(self, simulation_re=100):
        """
        Returns a list of snapshots for a simulation with a certain reynoldsnumber

        Args:
            simulation_re: The reynoldsnumber (Re) of th simulated flow. Default set to Re=100
            start with, since the baseline data is organized by Re (e.g. Re=100)

        Returns:
            snapshotList: List of snapshots
        """
        # Make sure simultionRe is a string
        if not isinstance(simulation_re, str):
            simulation_re = str(simulation_re)

        # Get a list of available baseline data snapshots belonging to a certain reynolds number
        snapshotList = glob(f'*/env/base_case/baseline_data/Re_{simulation_re}/processor0/*.*/')
        # Keep only the string with the time 
        snapshotList = [float(path.split('/')[-2]) for path in snapshotList]
        snapshotList.sort()
        return torch.Tensor(snapshotList)

    def get_random_control_start_time(self, lowerControlThreshold=None, upperControlThreshold=None, simulation_re=100):
        """
        Returns a random start time which is drawn from the base line data snapshots. Time boundaries can
        be set if necessary. The boundaries are not inclusive.

        Args:
            simulation_re: Contains the reynolds number (Re) of th simulated flow. Default set to Re=100
            lowerControlThreshold: Contains the lower time threshold for when to start control
            upperControlThreshold: Contains the upper time threshold for when to start control

        Returns:
            startTime: Returns a start time corresponding to the randomly selected index
            index: Returns the index of the randomly chosen point in time
        """
        # Get baseline data snapshots for given reynolds number
        snapshotList = self.get_snapshot_List(simulation_re).tolist()

        # Remove snapshots from list if thresholds apply
        if (lowerControlThreshold is not None) and (upperControlThreshold is None):
            new_snapshotList = [snapshot for snapshot in snapshotList if snapshot > lowerControlThreshold]
            print(new_snapshotList)

        elif (lowerControlThreshold is None) and (upperControlThreshold is not None):
            new_snapshotList = [snapshot for snapshot in snapshotList if snapshot < upperControlThreshold]
            print(new_snapshotList)

        elif (lowerControlThreshold is not None) and (upperControlThreshold is not None):
            new_snapshotList = [snapshot for snapshot in snapshotList if (snapshot > lowerControlThreshold) and (snapshot < upperControlThreshold)]
            print(new_snapshotList)

        index = torch.multinomial(torch.Tensor(new_snapshotList), 1)

        startTime = new_snapshotList[index]

        return startTime, index

    def process_waiter(self, proc, job_name, que):
        """
             This method is to wait for the executed process till it is completed
        """
        try:
            proc.wait()
        finally:
            que.put((job_name, proc.returncode))

    def run_trajectory(self, buffer_counter, proc, results, sample, action_bounds):
        """
        To run the trajectories

        Args:
            buffer_counter: which trajectory to run (n -> traj_0, traj_1, ... traj_n)
            proc: array to hold process waiting flag
            results: array to hold process finish flag
            sample: number of iteration of main ppo
            action_bounds: min and max omega value

        Returns: execution of OpenFOAM Allrun file in machine

        """
        # number of cores
        core_count = 4

        # get the random start
#        rand_control_traj = self.rand_n_to_contol(1)
        rand_control_traj = self.get_random_control_start_time(100, self.control_between[0], self.control_between[1])

        # changing of end time to keep trajectory length equal
        endtime = round(float(rand_control_traj[0] + 2), 2)

        # make dir for new trajectory
        traj_path = f"./env/sample_{sample}/trajectory_{buffer_counter}"

        print(f"\n starting trajectory : {buffer_counter} \n")
        os.makedirs(traj_path, exist_ok=True)

        zeros = '.2f'
        time_string = f"{rand_control_traj[0]:,{zeros}}"
        # copy files from base_case
        # change starting time of control -> 0.org/U && system/controlDict
        # change of ending time -> system/controlDict
        os.popen(f'cp -r ./env/base_case/agentRotatingWallVelocity/* {traj_path}/ && '
                 f'sed -i "s/startTime.*/startTime       {rand_control_traj[0]};/g" {traj_path}/0/U &&'
                 f'sed -i "s/absOmegaMax.*/absOmegaMax       {action_bounds[1]};/g" {traj_path}/0/U &&'
                 f'sed -i "s/startTime.*/startTime       {rand_control_traj[0]};/g" {traj_path}/0.org/U &&'
                 f'sed -i "s/absOmegaMax.*/absOmegaMax       {action_bounds[1]};/g" {traj_path}/0.org/U &&'
                 f'sed -i "/^endTime/ s/endTime.*/endTime         {endtime};/g" {traj_path}/system/controlDict &&'
                 f'sed -i "s/timeStart.*/timeStart       {rand_control_traj[0]};/g" {traj_path}/system/controlDict')
        
        for i in range(core_count):
            while not os.path.exists(f'{traj_path}/processor{i}//0.00025'):
                time.sleep(1)
            os.popen(f'cp -r ./env/base_case/baseline_data/Re_100/processor{i}/{time_string}025 {traj_path}/processor{i}/{time_string}025 &&'
                 f'sed -i "s/startTime.*/startTime       {rand_control_traj[0]};/g" {traj_path}/processor{i}/0.00025/U &&'
                 f'sed -i "s/startTime.*/startTime       {rand_control_traj[0]};/g" {traj_path}/processor{i}/{time_string}025/U &&'
                 f'sed -i "s/absOmegaMax.*/absOmegaMax       {action_bounds[1]};/g" {traj_path}/processor{i}/0.00025/U &&'
                 f'sed -i "s/absOmegaMax.*/absOmegaMax       {action_bounds[1]};/g" {traj_path}/processor{i}/{time_string}025/U')

        self.write_jobfile(core_count, job_name=f'traj_{buffer_counter}', file='./Allrun', job_dir=traj_path+'/')
        jobfile_path = f'{traj_path}' + '/jobscript.sh'

        proc[buffer_counter] = subprocess.Popen(['sh', 'submit_job.sh', jobfile_path])
        _thread.start_new_thread(self.process_waiter,
                                 (proc[buffer_counter], f"trajectory_{buffer_counter}", results))

    def sample_trajectories(self, sample, action_bounds):
        """

        Args:
            sample: main ppo iteration counter
            action_bounds: min and max omega value

        Returns: execution of n number of trajectory (n = buffer_size)

        """
        # set the counter to count the number of trajectory
        buffer_counter = 0

        # list for the status of trajectory running or finished
        proc = []

        # set the n_workers
        for t in range(int(max(self.buffer_size, self.n_worker))):
            item = "proc_" + str(t)
            proc.append(item)

        # get status of trajectory
        results = queue.Queue()
        process_count = 0

        # execute the n = n_workers trajectory simultaneously
        for n in np.arange(self.n_worker):
            self.run_trajectory(buffer_counter, proc, results, sample, action_bounds)
            process_count += 1
            # increase the counter of trajectory number
            buffer_counter += 1

        # check for any worker is done. if so give next trajectory to that worker
        while process_count > 0:
            job_name, rc = results.get()
            print("job : ", job_name, "finished with rc =", rc)
            if self.buffer_size > buffer_counter:
                self.run_trajectory(buffer_counter, proc, results, sample, action_bounds)
                process_count += 1
                buffer_counter += 1
            process_count -= 1


if __name__ == "__main__":
    n_worker = 2
    buffer_size = 4
    control_between = [0.1, 4]
    sample = 0
    env = env(n_worker, buffer_size, control_between)
    env.sample_trajectories(sample)
