# Passivity-Based Gain-Scheduled Control with Scheduling Matrices

Accompanying code for the control of the planar rigid two-link robotic manipulator used in Section V of Passivity-Based Gain-Scheduled Control with Scheduling Matrices [[arXiv]](https://www.arxiv.org/abs/2408.06476) [[Slides]](https://drive.google.com/file/d/1-crOgLCHxBla6MH-zPWYoelzzhZyE5lR/view?usp=sharingP).

## Installation

To clone the repository, run
```sh
$ git clone git@github.com:decargroup/matrix_scheduling_vsp_controllers.git
```

To install all the required dependencies for this project, run
```sh
$ cd matrix_scheduling_vsp_controllers
$ pip install -r ./requirements.txt
```

## Usage
To generate Figure 3 and Figures 5-7 in the paper, run
```sh
$ python main.py
```

The plots can be saved to `./Figures` by running 
```sg
$ python main.py --savefig
```
