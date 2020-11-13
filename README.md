# Savta Depth - Monocular Depth Estimation OSDS Project
Savta Depth is a collaborative *O*pen *S*ource *D*ata *S*cience project for monocular depth estimation.

Here you will find the code for the project, but also the data, models, pipelines and experiments. This means that the project is easily reproducible on any machine, but also that you can contribute to it as a data scientist.

Have a great idea for how to improve the model? Want to add data and metrics to make it more explainable/fair? We'd love to get your help.

## Contributing Guide
Here we'll list things we want to work on in the project as well as ways to start contributing.
If you'd like to take part, please follow the guide.

### Setting up your environment to contribute
* To get started, fork the repository on DAGsHub
* Now, you have 3 way to set up your environment: Google Colab, local or docker. If you're not sure which one to go with, we recommend using Colab.
#### Google Colab
Google Colab can be looked at as your web connected, GPU powered IDE. Below is a link to a well-documented Colab notebook, that will load the code and data from this repository, enabling you to modify the code and re-run training. Notice that you still need to modify the code within the `src/code/` folder, adding cells should be used only for testing things out.

In order to edit code files, you must save the notebook to your drive. You can do this by typing `ctrl+s` or `cmd+s` on mac.

\>\> **[SavtaDepth Colab Environment](https://colab.research.google.com/drive/19027P09jiiN1C99-YGk4FPj0Ol9iXUIU?usp=sharing)** \<\<

**_NOTE: The downside of this method (if you are not familiar with Colab) is that Google Colab will limit the amount of time an instance can be live, so you might be limited in your ability to train models for longer periods of time._**

This notebook is also a part of this project, in case it needs modification, in the `Notebooks` folder. You should not commit your version unless your contribution is an improvement to the environment.

#### Local
* Create a virtual environment or Conda environment and activate it
    ```bash
    # Create the virtual environment
    $ make env
  
    # Activate the virtual environment
    # VENV
    $ source env/bin/activate .
  
    # or Conda
    $ source activate savta_depth
    ```
* Install the required libraries
    ```bash
    $ make load_requirements
    ```
  **_NOTE: Here I assume a setup without GPU. Otherwise, you might need to modify requirements, which is outside the scope of this readme (feel free to contribute to this)._**
* Pull the dvc files to your workspace by typing:

    ```bash
    $ dvc pull -r dvc-remote
    $ dvc checkout #use this to get the data, models etc
    ```

    **Note**: You might need to install and setup the tools to pull from a remote. Read more in [this guide](https://dagshub.com/docs/getting-started/set-up-remote-storage-for-data-and-models/) on how to setup Google Storage or AWS S3 access.
* After you are finished your modification, make sure to do the following:
    * If you modified packages, make sure to freeze your virtualenv by typing in the terminal:

        ```bash
        $ make save_requirements
        ```

    * Push your code to DAGsHub, and your dvc managed files to your dvc remote. To setup a dvc remote please refer to [this guide](https://dagshub.com/docs/getting-started/set-up-remote-storage-for-data-and-models/).
    
#### Docker
* Next, clone the repository you just forked by typing the following command in your terminal:
  ```bash
  $ git clone https://dagshub.com/<your-dagshub-username>/SavtaDepth.git
  ```
* To get your environment up and running docker is the best way to go. We use an instance of [MLWorkspace](https://github.com/ml-tooling/ml-workspace). 
    * You can Just run the following commands to get it started.

        ```bash
        $ chmod +x run_dev_env.sh
        $ ./run_dev_env.sh
        ```

    * Open localhost:8080 to see the workspace you have created. You will be asked for a token – enter `dagshub_savta`
    * In the top right you have a menu called `Open Tool`. Click that button and choose terminal (alternatively open VSCode and open terminal there) and type in the following commands to install a virtualenv and dependencies:

        ```bash
        $ make env
        $ source activate savta_depth
        ```
        
        Now when we have an environment, let's install all of the required libraries.
        
        **Note**: If you don't have a GPU you will need to install pytorch separately and then run make requirements. You can install pytorch for computers without a gpu with the following command:

        ```bash
        $ conda install pytorch torchvision cpuonly -c pytorch
        ```
        
        To install the required libraries run the following command:
        
        ```bash
        $ make load_requirements
        ```


* Pull the dvc files to your workspace by typing:

    ```bash
    $ dvc pull -r dvc-remote
    $ dvc checkout #use this to get the data, models etc
    ```

    **Note**: You might need to install and setup the tools to pull from a remote. Read more in [this guide](https://dagshub.com/docs/getting-started/set-up-remote-storage-for-data-and-models/) on how to setup Google Storage or AWS S3 access.
* After you are finished your modification, make sure to do the following:
    * Freeze your virtualenv by typing in the terminal:

        ```bash
        $ make save_requirements
        ```

    * Push your code to DAGsHub, and your dvc managed files to your dvc remote. In order to setup a dvc remote please refer to [this guide](https://dagshub.com/docs/getting-started/set-up-remote-storage-for-data-and-models/).
    
---
### After pushing code and data to DAGsHub
* Create a Pull Request on DAGsHub!
* 🐶

### TODO:
[ ] Web UI
[ ] Testing various datasets as basis for training
[ ] Testing various models for the data
[ ] Adding qualitative tests for model performance (visually comparing 3d image outputs)
