# escr-yaltai

This is a repository to share code used for a workflow that combines segmentation data results both from eScriptorium (via [kraken](https://kraken.re/main/index.html)) and [YALTAi](https://github.com/ponteineptique/YALTAi). This is done because using YALTAi alone sometimes produces regions that are too tightly wrapped around the text, which then trims the polygons used for line masks and interferes with subsequent text recognition results. The code expects a SLURM-based environment and was created for use on Princeton HPCs.

## Setup

Clone the repository, then set up the conda environment. `cd` into the base `escr-yaltai` directory and then run the following commands:

```
cd /scratch/network/$USER
git clone https://github.com/cmroughan/escr-yaltai.git
cd escr-yaltai
module purge
module load anaconda3/2024.2
conda env create -f environment.yml
```

## Running the code

Download an **ALTO XML** export from eScriptorium **with images** included. Upload this .zip file to the `1_UPLOAD/` directory. Then run the below commands.

(If you have not yet loaded anaconda:)

```
module purge
module load anaconda3/2024.2
```

Then:

```
cd /scratch/network/$USER/escr-yaltai/src
conda activate yaltai
tmux
```

Then, inside the `tmux` session, run:

```
python run.py
```

Running the python code in `tmux` ensures that the code keeps running even if you close the command line.

## Results

When the code completes, it will save a new .zip file in `2_DOWNLOAD/`, along with a .txt file report of the results. Check this report to make sure no errors occurred. You can then download the .zip file and upload it to the corresponding document in eScriptorium using the "Import" "Transcriptions (XML)" option.
