# Model-based Reinforcement Learning for Accelerated Learning From CFD Simulations
## Abstract
This thesis/project presents and evaluates an approach for model-based deep reinforcement learning used for active flow control of a 2D flow past a cylinder to accelerate the learning process of the DRL agent. In the wake of a 2D flow past a cylinder, a von Kármán vortex street can be observed. By rotating the cylinder a DRL agent tries to find a proper control law for mitigating the vorticity and therefore the drag and the oscillations of the drag and the lift at the cylinder. Since the DRL agent only has access to 400 fixed pressure sensors on the cylinder's surface a feed-forward neural network is developed for predicting the pressure values of the next state using the pressure from the previous states and the action taken by the agent. The environment model is used autoregressively to predict whole trajectories from only one start state. The presented approach shows the general feasibility of model-based trajectory sampling for active flow control using DRL. Furthermore, the influence of the number of subsequent previous states used for the prediction of the next state is investigated, showing that more subsequent states yield a better prediction accuracy. Also, the reduction of the number of pressure sensors used for the environment model input is investigated considering the memory consumption and prediction accuracy. The resulting model predicts the next state containing 400 pressure values, as well as the drag and lift coefficient from 30 subsequent time steps containing only 16 pressure values plus the action. The influence of the number of neurons per hidden layer has also been examined, revealing that the prediction accuracy rises with a rising number of neurons per hidden layer but the models have not been able to provide a stable and promising DRL training. Although the tested neural network architectures are not sufficient enough for conducting a working model-based DRL training run, this thesis reveals several pitfalls and challenges of environment modeling for this flow problem and proposes the next steps to take from here.

The link to the pulbication of the related thesis can be found at the [bottom](#report) of the page.

This research project is a direct continuation of the work by [Darshan Thummar](https://github.com/darshan315/flow_past_cylinder_by_DRL) and [Fabian Gabriel](https://github.com/FabianGabriel/Active_flow_control_past_cylinder_using_DRL).

## Troubleshooting
Running into trouble? Use the [wiki](https://github.com/ErikSchulze1796/Active_flow_control_past_cylinder_using_DRL/wiki) or open an issue on the [issues page](https://github.com/ErikSchulze1796/Active_flow_control_past_cylinder_using_DRL/issues).

## Setting up the Cluster for Training (SLURM workload manager)
### Setup of Cluster Environment:

Deposit this repository on the cluster using an SFTP tool. I would reccomend [rsync](https://explainshell.com/explain?cmd=+rsync+-chavzP+--stats++%2Fpath%2Fto%2Flocal%2Fstorage++user%40remote.host%3A%2Fpath%2Fto%2Fcopy):

```
rsync -chavzP --stats /path/to/local/storage user@remote.host:/path/to/copy
```

Any other SFTP programm (e.g. [filezilla](https://filezilla-project.org/)) will do too.
If your want to get your files from the cluster back to your local machine just switch the source and destination path and append the option ```--exclude_from='exclude_file.txt'```:

```
rsync -chavzP --stats --exclude-from='exclude_me.txt' user@remote.host:/path/to/copy /path/to/local/storage
```


Python libraries on cluster are installed by creating a virtual environment. The current implementation requires the ```venv``` folder to be located inside the home directory, when working on the cluster. Otherwise the required libraries might not be found by the scripts:

```
module load python/3.7 
python3 -m pip install --user virtualenv 
python3 -m virtualenv venv
```


### Setup of Virtual Environment on Cluster:
To activate the virtual environment:

```
source venv/bin/activate
```

To deactivate the virtual environment:

```
deactivate
```


To install the python libraries in venv virtual environment:

```
pip install -r ./DRL_py/docker/requirements.txt
```

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
## Setting up openFOAM environment
In order to start the simulations correctly you need to first build the openFOAM simulation environment. This can be done following the instructions in the [README.MD](https://github.com/ErikSchulze1796/Active_flow_control_past_cylinder_using_DRL/blob/main/DRL_py_beta/agentRotatingWallVelocity/README.md) in the ```./DRL_py_beta/agentRotatingWallVelocity/``` directory or you could just use the following commands.
The starting point is the parent directory of ```./DRL_py_beta``` where the ```of_v2012.sif``` file should be located:

```
module load singularity/3.6.0rc2
# top level folder of repository
singularity shell of_v2012.sif
# now we are operating from inside the container
source /usr/lib/openfoam/openfoam2012/etc/bashrc
cd /DRL_py_beta/agentRotatingWallVelocity/
wmake
```

After following the steps the generated ```libAgentRotatingWallVelocity.so``` file hast to be copied/moved to the ```./DRL_py_beta/``` directory. The starting point is the ```./DRL_py_beta/agentRotatingWallVelocity/``` directory:
```
mv ./libAgentRotatingWallVelocity.so ./..
```

## Singularity and SLURM

[Singularity]() is a container tool that allows making results reproducible and performing simulations, to a large extent, platform independent. The only remaining dependencies are Singularity itself and Open-MPI (see next section for further comments). To build the image, run:

```
sudo singularity build of_v2012.sif docker://andreweiner/of_pytorch:of2012-py1.7.1-cpu
```
To run a simulation with Singularity, use the dedicated *Allrun.singularity* scripts. TU Braunschweig's HPC uses the SLURM scheduler. The repository contains an annotated example *jobscript* file. The script expects the Singularity image in the top level directory of this repository and the simulation folder in *run*. To submit a job, run and replace *name_of_simulation* with the folder that contains it:

```
sbatch jobscript name_of_simulation
// Example:
sbatch jobscript cylinder2D_base
```
To show all running jobs of a user, use `squeue -u $USER`. Another helpful command is `quota -s` to check the available disk space.
If you want to cancel a job just use the command `scancel JOBID`.

## Starting an Environement Model Training
In order to start a training of the environment model the hyperparameters of the training and the trained model have to be set first. This can be done in the `train_pressure_model_cluster.py`. The trained model is a feed-forward neural network. Therefore the number of hidden layers and the number of neurons per layer can be set by changing the `hidden_layer` and `neurons` variables. Furthermore the number of time steps can be set by changing the `n_steps_history` variable in order to adjust the subsequent state to be used for the prediciton. By changing the `every_nth_element` variable the number of input sensors for the environment model can be set to reduce memory consumption. The ouput of 400 sensors stays untouched. The training hyperparameters can be adjusted by changing the `batch_size`, `epochs` and `lr` variables for the batch size, the number of epochs to be trained for and the learning rate. The training data is loaded from the locations for the features and lables specified by `location_train_features` and `location_train_labels` variables.

The training can be started by simply executing the `train_pressure_model_cluster.py` script using:
```
python3 train_pressure_model_cluster.py
```
or by using the slurm scheduler on the cluster using the slurm script:
```
sbatch model_train_job.sh
```
The model is then saved to the path specified by `save_model_in` variable.

## Starting a DRL Training

Choose a setup: 

`cd DRL_py_beta`

Before you can start you have to download the baseline_data from here:
[baseline_data](https://cloudstorage.tu-braunschweig.de/getlink/fiM1FGVmAfb8ACriCFRs74wM/baseline_data.zip)(400MB)

Then you have to copy the content of that folder into the ./env/baseline_case/baseline_data folder

Start the Training:

`sbatch python_job.sh`

The environment model used for the DRL training is specified in `reply_buffer.py` by `environment_model_path` variable in the `fill_buffer_from_environment_model_total(...)` function.

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

# Report

The report for this study : https://doi.org/10.5281/zenodo.6375575

BibTex citation : 
```
@misc{schulze_erik_2022_6375575,
  author       = {Schulze, Erik},
  title        = {{Model-based Reinforcement Learning for Accelerated 
                   Learning From CFD Simulations}},
  month        = mar,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.6375575},
  url          = {https://doi.org/10.5281/zenodo.6375575}
}
```
