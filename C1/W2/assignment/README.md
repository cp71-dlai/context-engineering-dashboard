- READMEs which end with _ (README_) are the ones which should be created in the exact places during specialization/course development as well.
- Simple README are just for explanation purposes. 


# Assignment README

- The assignments folder where we have more than one assignment in a week should contain distinct assignment folders following the format __A(followed by the assignment#)__.
      
      - A1
      - A2
      .
      .
      .
- If the week has only 1 assignment we’ll skip making these folders.
- Also within this folder we’ll have our assignment files, solution & student facing versions. 
- If there are more than one version of the same assignment, we’ll name the assignment files appropriately by adding __v(version #)__ (shown below) and move the older versions into the folder __previous-versions__.
- __previous-versions__ will only be created when we’ll have more than one version. Otherwise this folder won’t exist.
- Within this folder we also need to create folders, whenever apply __images__ for saving all of the images that are being used in the assignment, __data__ for storing the data files being used and __models__ for storing all of the pre-trained models we'll be using in the assignment.
- In the name of the assignment file we’ll also mention the platform for which it is used for. 
      
      E.g 
      - Coursera
      - Colab
  * If there is no mention of the platform then by default it will be assumed that the assignment is for Coursera.

- For the name of the assignment file we’ll use **C(course #)_W(week #)_A(assignment #)_Platform-name_Assignment**.
- If it is the sole/only assignment of the week then we’ll _not_ add __A(assignment #)__.
- For the solution version we’ll use the word __solution__ as suffix to the word **Assignment**.
- Below are examples for assignment file names with varying conditions
      
      - Assignment with A# and platform-name:
          * C1_W2_A3_Colab_Assignment.ipynb
      - Solution assignment without a platform mention:
          * C1_W2_A3_Assignment_Solution.ipynb
      - If it is the only assignment of the week:
          * C1_W2_Assignment.ipynb
      - Assignment which has more than one version of it:
          * C1_W2_A3_Assignment_v2.ipynb
          * C1_W2_A3_Assignment_Solution_v2.ipynb
