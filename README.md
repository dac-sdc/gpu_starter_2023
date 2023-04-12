# DAC 2023 Design Contest

For full contest details, please see the [2023 DAC System Design Contest](https://dac-sdc.github.io/2023/) page.

For general questions regarding this contest, please use the Slack workspace: <https://join.slack.com/t/dac-sdc/shared_invite/zt-1rrtmgjad-NCYE2leBfOw8xTOp52KR7w>

## Usage
The get started, users have to run the following command on the gpu board:

```shell
cd /home/root/jupyter_notebooks
git clone https://github.com/dac-sdc/gpu_starter_2023.git
```
Remember the user name and password are both `xilinx`.

After the above step is completed successfully, you will see a folder `gpu_starter_2023` under your 
jupyter notebook dashboard.  Open the `sample_team/dac_sdc.ipynb` notebook for directions on where to begin.

## Folder Structure

1. sample_team: This folder contains files for a sample team.  This includes a <teamname>.bit and <teamname>.tcl file that defines the hardware, and a `.ipynb` jupyter notebook, and a `hw` folder that is used to create a Vivado project.  You should create a new folder for your team, where you will keep all of your files.

2. images: All the test images are stored in this folder.  Replace the example images in this directory with the full training set.

3. result: The results contain the output xml produced when execution is complete, and contains the runtime, energy usage, and predicted location of each object in each image.


