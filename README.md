# Active control of the flow past a cylinder at varying Reynolds number

This research project is a direct continuation of the work by [Darshan Thummar](https://github.com/darshan315/flow_past_cylinder_by_DRL) and [Fabian Gabriel](https://github.com/FabianGabriel/Active_flow_control_past_cylinder_using_DRL). The repository is structured as follows:
- *test_cases*: OpenFOAM simulation setups

## Getting started

To run a test case, create a *run* folder (ignored by version control), copy the case from *test_cases* to *run*, and execute the *Allrun* script. To run with Singularity, the image has to be built fist; see *Singularity and SLURM*.

```
mkdir -p run
cp -r test_cases/cylinder2D_base run/
cd run/cylinder2D_base
# for execution with singularity
./Allrun.singularity
# for execution for local OpenFOAM installation
./Allrun
```

## Singularity and SLURM

[Singularity]() is a container tool that allows making results reproducible and performing simulations, to a large extent, platform independent. The only remaining dependencies are Singularity itself and Open-MPI (see next section for further comments). To build the image, run:

```
sudo singularity build of_v2012.sif docker://andreweiner/of_pytorch:of2012-py1.7.1-cpu
```
To run a simulation with Singularity, use the dedicated *Allrun.singularity* scripts. TU Braunschweig's HPC uses the SLURM scheduler. The repository contains an annotated example *jobscript*. The script expects the Singularity image in the top level directory of this repository and the simulation folder in *run*. To submit a job, run:

```
sbatch jobscript name_of_simulation
```
To show all running jobs of a user, use `squeue -u $USER`. Another helpful command is `quota -s` to check the available disk space.

## Starting a Training

Choose a setup: 

`cd DRL_py_beta`

Start the Training:

`sbatch python_job.sh`

## Evaluate a specific Episode
Inside a Setup Folder:

If you choose the sample 52 for example:

`cp ./env/base_case/agentRotatingWallVelocity_start_without_training ./env/run/sample_52`

`cp ./results/models/policy_51.pt ./env/run/sample_52/policy.pt`

Now edit the jobscript file in that newly created folder in line 11:

From:

`cd ./env/run/sample_*/`

To:

`cd ./env/run/sample_52/`

Now to start the evaluation:

`sbatch ./env/run/sample_52/jobscript.sh`

## Resetting the Setup
Make sure you have downloaded and saved all needed data

Choose a setup:

`cd DRL_py_beta`

Reset:

`sbatch cleanup.sh`
