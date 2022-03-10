# platelet-analysis
`plateletanalysis` is a Python package for the downstream analysis of platelet movement in forming blood clots. 


## Installation 
The package can be installed on your machine as follows:

```bash
git clone https://github.com/AbigailMcGovern/platelet-analysis.git
cd platelet-analysis
pip install -e . # installs an editable version of the package
```

## Background information
Prior to analysis, microscopy videos of growing clots, in which a small subset of platelets are labeled, are segmented to find the location of platelets. On the basis of segmentations, information is collected about the segmented objects (e.g., fluorescence intensity, volume, shape). The segmentations are then tracked using the Python package TrackPy and assigned an id (a column labeled `"particle"` in our data frames). 


## Data Configuration
Currently, the package is specialised to a particular configuration of data:

### Main tracking output
The main output (format: csv) from the segmentation and tracking analysis should have the following variables:
- `pid` – the platelet unique identifyer for that point in time
- `file` – the name of the column with the file name (from the original ND2 video file)
- `xs` – platelet x coordinate
- `ys` – platelet y coordinate
- `zs` – platelet z coordinate
- ... TBC


### Main tracking output metadata
For each video for which tracks are found there should be an accompanying metadata file (format: csv)
- path
- ... TBC


### Collated data files
These are the data files that should be used for generating analyses using the menu interface (format: parquet). Using functions supplied in `plateletanalysis` you can generate one of these files (currently there are an incomplete examples of this in the `examples` directory - the files ending `_df.py`). Generally we keep one data frame per experimental condition. That is, there is one data frame per inhibitor or control. The names of the data frames that will appear as options in the analysis are specified in the `plateletanalysis.config` module (with will eventually be reconfigured and renamed). The path at which to find the data frame is located in the `data` directory in `file_paths.csv`. 
Variables include:
- `pid` –
- `path` – 
- `x_s` – 
- `ys` – 
- `zs` – 
- ... TBC


## Running the analysis
The user interface-based analysis can be run by calling `plateletanalysis.run_analysis`. Before running the analysis Please ensure `file_paths.csv` is properly to include the locations of (1) the directory containing collated data files, (2) the max calcium quadratic analysis regression, (3) the min calcium quadratic analysis regression, (4) the paraview files if necessary, and (5) the data containing any outliers to be excluded. 

```Python
from plateletanalysis import run_analysis

run_analysis.show(run=True)
```


## Roadmap
- Ensure analyses run on a variety of machines with varying access to data
- Alter config file to add functions that help the user configure to their machine (i.e., help them edit the `file_paths.csv` file)
- Maybe move lists of treatment long and short names into a CSV (might be easier to edit)
- add the `t_traj_move` and any related functions for mapping platelets moving in different ways at different points in time


## Guidelines for version control
Here we will use git for version control. 

When you want to get the latest version of the main branch of the code, ensure you are on the `main` branch of the git repo by entering `git switch main` into the command line. You can then enter `git pull origin main`, which obtain the main branch from the repo on GitHub (which git knows as `origin`). It is recommended you do this before you change anything incase the copy of `main` on your computer is out of date. 

When you want to change the code, please checkout a new branch for the repo. Branches allow us to keep many versions of the code. When you switch to another branch, git records any changes from the main branch and allows you to switch between versions of the code. If you  You can do this by entering the following command `git checkout -b <new-branch-name>`. If you want to find out which branch you are on and which branches exist, use `git branch`. If you wish to switch to another branch use `git switch <branch-name>`. Note that you won't be able to switch between branches unless you have staged and committed your changes (see below). 

When you make changes to the code, in order to upload the changes to GitHub, you will have to first stage and commit. Staging the changes prepares them for commit. You can see which files have changes that are staged or un-staged using `git status`. Stage the changes by using `git add <filename>` or `git add <filename> <filename> ...` (if you need to stage multiple files). To commit the changes use `git commit -m "please type a message so that we can see what the changes do"`. Once the changes are staged and committed, they still need to be uploaded to GitHub. To do this, you can use `git push origin <branch-name>`. This will provide a link to the location of the changes on the repository. In order to add the changes to the main branch you will need to use GitHub itself. Go to the link provided and click "make pull request". 


