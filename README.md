# DLAI Standard Course Template and Best Practices

This repo contains the standard template for creating our course repos. Aside from the layout, it also contains assignment templates, READMEs, and other files that will help organize our course materials. 

This top-level README provides an overview of the files included and discusses some of the best practices when working with the course notebooks. More specific topics are discussed in the subdirectory notebooks and READMEs.


## Table of Contents

- [1. Directory Structure](#1)
  - [1.1 Single assignment per week](#1-1)
  - [1.2 Multiple assignments per week](#1-2)
- [2. Naming Convention](#2)
  - [2.1 Github Repo](#2-1)
  - [2.2 Course Folders](#2-2)
  - [2.3 Assignments](#2-3)
  - [2.4 Ungraded Labs](#2-4)
  - [2.5 Lecture Animations](#2-5)
  - [2.6 Quizzes](#2-6)
- [3. Development Notes](#3)
  - [3.1 Coursera Labs](#3-1)
  - [3.2 Colab](#3-2)
  - [3.3 Github](#3-2)
- [4. Miscellaneous](#4)
  - [4.1 Using the LFVG](#4-1)
  - [4.2 nbgrader notebooks](#4-2)


<a name='1'></a>
## 1. Directory Structure

This section illustrates how the directories are laid out in a course repo.

<a name='1-1'></a>
### 1.1 One assignment per week

Most of our courses so far only have one assignment per week so this layout will most likely be the one you will work with. The directory structure with descriptions of the main files are shown below:



```
.
+-- README                                                   # this README
+-- README_                                                  # placeholder README for the specialization
+-- standard-repo                                            # template directory
|   +-- C1                                                   # Course 1 materials
|       +-- lfvg                                             # learner facing version generator
|       +-- README                                           # contains at least the full course name
|       +-- W1                                               # Week 1 materials
|           +-- lecture_animation(s)                         # Week 1 lecture animation file(s), if applicable only then this folder will be created  
|           +-- autograder                                   # autograder files
|           +-- assignment                                   # Week 1 assignment materials 
|               +-- images                                   # images to put in the markdown
|               +-- data                                     # datasets used in the assignment
|               +-- models                                   # models used in the assignment
|               +-- previous-versions                        # optional folder to contain any extra files you want to be available in the next commits
|               +-- C1_W1_Assignment_Solution.ipynb          # assignment solution template
|               +-- C1_W1_Coursera_Assignment_Solution.ipynb    # assignment solution template with a Colab badge
|               +-- C1_W1_Colab_Assignment.ipynb             # placeholder for the Colab version of the assignment
|           +-- ungraded_labs                                # Week 1 ungraded lab materials
|               +-- C1_W1_Lab_1_name_of_the_lab.ipynb        # ungraded lab template      
|               +-- C1_W1_Lab_2_name_of_the_lab.ipynb
|           +-- quiz                                         # Week 1 quiz materials 
|       +-- W2
|       +-- W3
|       +-- W4
|   +-- C2
|   +-- C3
```

<a name='1-2'></a>
### 1.2 Multiple assignments per week

You can use this layout for courses that have more than one assignment. The contents are basically the same as the previous one but you will have to create additional `A` subdirectories to separate the different assignments (i.e. `A1` for assignment 1, `A2` for assignment 2...)

```
.
+-- README                                                  # this README
+-- standard-repo-multiple-assignments-per-week
|   +-- C1
|       +-- README
|       +-- W1
|           +-- lecture_animation(s)                        # Week 1 lecture animation file(s), if applicable only then this folder will be created  
|           +-- Assignment
|               +-- A1                                      # assignments are separated with these 'A'-prefixed folders (e.g. A1, A2)
|                   +-- autograder
|                   +-- images                                   
|                   +-- data                                     
|                   +-- models                                   
|                   +-- previous-versions                        
|                   +-- C1_W1_A1_Assignment.ipynb
|                   +-- C1_W1_A1_Assignment_Solution.ipynb
|               +-- A2
|                   +-- autograder
|                   +-- images                                   
|                   +-- data                                     
|                   +-- models 
|                   +-- previous-versions
|                   +-- C1_W1_A2_Assignment.ipynb
|                   +-- C1_W1_A2_Assignment_Solution.ipynb
|           +-- UngradedLabs
|               +-- C1_W1_Lab_1_name_of_the_lab.ipynb
|               +-- C1_W1_Lab_2_name_of_the_lab.ipynb
|           +-- Quizzes
|       +-- W2
|       +-- W3
|       +-- W4
|   +-- C2
|   +-- C3
```

<a name='2'></a>
## 2. Naming Convention

<a name='2-1'></a>
### 2.1 Github Repo

We have two options when naming the Github repo:

1. lower case with hyphens (e.g. `tensorflow-3`, `ai-4-medicine`)
2. capitalized acronyms (e.g.`MLEP`, `NLP`)

<a name='2-2'></a>
### 2.2 Course Folders

We will just use `C` followed by the course number (e.g. `C1`). We avoid appending course names (e.g. `C1 - Generative Adversarial Networks`) because these can change during development and renaming the folders will break the repo history and can cause merge conflicts.

<a name='2-3'></a>
### 2.3 Assignments

Assignments will follow the format:
* *CourseNumber_WeekNumber_Assignment_Solution.ipynb* for solutions 
  * e.g. `C1_W1_Assignment_Solution.ipynb`
* *CourseNumber_WeekNumber_Assignment.ipynb* for learner versions
  * e.g. `C1_W1_Assignment.ipynb`

If a course uses different environments for the assignment (e.g. Colab and Coursera), it is good to add the platform name in the filename to avoid confusion.
  * e.g. `C1_W1_Colab_Assignment_Solution.ipynb`


<a name='2-4'></a>
### 2.4 Ungraded Labs

Ungraded labs will follow the format:

* CourseNumber_WeekNumber_Lab_LabNumber_Name_of_lab.ipynb
  * e.g. `C3_W1_Lab_2_Transfer_Learning_CIFAR_10.ipynb`

<a name='2-5'></a>
### 2.5 Lecture Animations

Lecture animation files will follow the format:

* CourseNumber_WeekNumber_LAN_Lecture-name_name_of_the_file.ipynb
  * e.g. `C1_W2_LAN_Building-Graphs_Graphs.ipynb`

### 2.6 Quizzes

Quiz file(s) will follow the format:

In case of only 1 quiz per week
* CourseNumber_WeekNumber_Quiz
  * e.g. `C1_W1_Quiz`

In case of more than 1 quiz per week
* CourseNumber_WeekNumber_Quiz_QuizNumber_quiz-name
  * e.g. `C1_W1_Quiz_2_gradient-descent`


<a name='3'></a>
## 3. Development Notes

Below are some rules and best practices to follow during the development phase of the course and assignments.

<a name='3-1'></a>
### 3.1 Coursera Labs

* Avoid developing assignments in the instructor workspace. One reason is some labs generate extra files and this can be inadvertently published to the students. Another is you may want to reset your progress at some point and it's hard to do that when the files in the instructor workspace is already cluttered. You should work in your own workspace first, then upload only the important files in the instructor lab once you're satisfied with a set of revisions.

* Make sure to clear the output of the cells before publishing the lab.



<a name='3-2'></a>
### 3.2 Colab

* You can skip composing a table of contents because Colab automatically does that for you based on the markdown.
* Generate a lookup table of all Colabs for easy lookup. This will be very useful when maintaining the course post launch. [Here](https://docs.google.com/spreadsheets/d/1rftzL0Rea6YFqa6TEgK7r7jZXPmB8rRubGJH7KDyMA8/edit#gid=0) is an example.

<a name='3-3'></a>
### 3.3 GitHub

* Be regular with your pushes into your branch with your edits and revisions
* Be vocal about your pushes and progress with your team
* Make sure to mention the Course number and Week number (e.g. C2 W1) in your commit messages

<a name='4'></a>
## 4. Generating Learner and Solution Notebooks

<a name='4-1'></a>
### 4.1 Using the LFVG

This is a tool developed by Andres Zarta to automatically generate learner notebooks from solution notebooks. This is particularly useful for courses that use a custom autograder. Instructions on how to prepare the notebooks are in the assignment template notebooks while instructions on how to build the executable can be found [here](https://github.com/https-deeplearning-ai/dlai_tools/tree/master/learner_generator).

*TODO (Chris) - include instructions on how to use the new version of LFVG

<a name='4-2'></a>
### 4.2 nbgrader notebooks

For courses that use nbgrader, the learner notebook is automatically generated from the nbgrader interface. You can follow this process when pushing the notebooks to the repo:

1. From the nbgrader interface, click `Generate` to produce the learner version.
2. Click `Preview` to open the Jupyter workspace.
3. Download the learner version from the `release` folder.
4. Download the solution from the `source` folder and append `_Solution` to the filename (e.g. `C1_W1_Assignment_Solution.ipynb`)
5. Copy the files to the corresponding subdirectory in the repo (e.g. `C1 -> W1 -> Assignment`)
6. Add, Commit, and Push.

*TODO (Chris): add screenshots
