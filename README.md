_your zenodo badge here_

# Pollack-etal_2024_inprep

**Funding rules that promote equity in climate adaptation outcomes**

Adam Pollack<sup>1\*</sup>, Sara Santamaria-Aguilar<sup>2</sup>, Casey Helgeson<sup>3,4</sup>, Pravin Maduwantha<sup>2</sup>, Thomas Wahl<sup>2</sup>, Klaus Keller<sup>1</sup>

<sup>1 </sup> Thayer School of Engineering, Dartmouth College; Hanover, 03755, USA.
<sup>2 </sup> Department of Civil, Environmental and Construction Engineering, University of Central Florida; Orlando, 32816, USA.
<sup>3 </sup> Earth and Environmental Systems Institute, Penn State University; State College, 16801, USA.
<sup>4 </sup> Department of Philosophy, Penn State University; State College, 16801, USA.

\* corresponding author:  adam.b.pollack@dartmouth.edu

## Abstract
In this paper the authors develop a long-term global energy-economic model which is capable of assessing alternative energy evolutions over periods of up to 100 years. The authors have sought to construct the model so that it can perform its assigned task with as simple a modelling system as possible. The model structure is fully documented and a brief summary of results is given.

## Journal reference
Under construction

## Data reference 

### Input data
Under construction.
urls/dois for other input data. 
Need to add doi'd inundation model outputs. This section should be input data that is processed in this workflow. 

### Output data
Under construction
Reference for each minted data source for your output data.  For example:

Human, I.M. (2021). My output dataset name [Data set]. DataHub. https://doi.org/some-doi-number

Should do this in a way where it's all raw, interim, final data - even the raw data that gets downloaded by following the workflow. 

## Contributing modeling software
| Model | Version | Repository Link | DOI |
|-------|---------|-----------------|-----|
| model 1 | version | link to code repository | link to DOI dataset |
| model 2 | version | link to code repository | link to DOI dataset |
| component 1 | version | link to code repository | link to DOI dataset |

SFINCS, for one. Others?

## Reproduce my experiment
You can follow the instructions below to reproduce all results reported in the manuscript and supplementary materials. For this experiment, reproduction does not imply bit-wise reproducibility because there are stochastic processes. You should obtain similar quantitative results and figures. 

If you would like to check for internal bit-wise correctness, you can do so by assessing the output data we link to above. In addition to figures and outputs, we also include all downloaded, raw, and interim data. You can test whether the interim data, which includes realizations of the simulations we used for the published study, is consistent with the processed final data.

Note to self: how should we deal with the hazard model chain? It's not part of this repository, but it's part of the project. Maybe we just say that we are holding off on sharing the configuration for this, but have tested it for internal reproducibility. All users can reproduce the analysis for the boundary conditions, and all users can run SFINCS and have our outputs. We will share our exact configuaration after a separate study is published. We also can share them before that with our peer reviewers. 


1. Clone this repository into a local project directory.
2. [Run some line of code to get the conda environment installed]
3. Download and install the supporting input data required to conduct the experiement from [Input data](#input-data). This is preconfigured into the correct directory structure.
4. Run the following scripts in the `workflow` directory to re-create this experiment. The "How to Run" column is based on running python3 scripts from a Unix shell. 

| Script Name | Description | How to Run |
| --- | --- | --- |
| `download_data.py` | Script to download data | `python download_data.py` |
| `unzip_data.py` | Script to unzip and move data around | `python unzip_data.py` |
| `process_ev.py` | Process exposure and vulnerability data | `python process_ev.py` |
| `process_haz.py` | Process inundation model output | `python process_haz.py` |
| `ensemble.py` | Generate ensembles for risk estimation | `python ensemble.py` |
| `optimal_elev.py` | Use ensembles to find optimal heightening for each structure | `python optimal_elev.py` |
| `allocate_funds.py` | Apply funding rules | `python allocate_funds.py` |
| `results.py` | Generate figures and summary statistics | `python results.py` |
| `sup_results.py` | Generate supplementary figures and summary statistics  | `python sup_results.py` |