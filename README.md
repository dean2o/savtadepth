# Savta Depth - Monocular Depth Estimation OSDS Project
Savta Depth is a collaborative *O*pen *S*ource *D*ata *S*cience project for monocular depth estimation.

Here you will find the code for the project, but also the data, models, pipelines and experiments. This means that the project is easily reproducible on any machine, but also that you can contribute to it as a data scientist.

Have a great idea for how to improve the model? Want to add data and metrics to make it more explainable/fair? We'd love to get your help.

## Contributing Guide
Here we'll list things we want to work on in the project as well as ways to start contributing.
If you'd like to take part, please follow the guide.

### Setting up your environment to contribute
* To get started, fork the repository on DAGsHub
* Next, clone the repository you just forked by typing the following command in your terminal:
  ```bash
  $ git clone https://dagshub.com/<your-dagshub-username>/SavtaDepth.git
  $ dvc checkout #use this to get the data, models etc
  ```
* To get your environment up and running docker is the best way to go.
  We created a dockerfile that has all you need in it and will install all requirements in the 'requirements.txt' file as well as run a jupyter lab instance.
    * Just open the terminal in your project directory and type `docker build "savta_depth_dev" ."
    * After the docker image is created run the following commands:
    ```bash
    $ chmod +x run_dev_env.sh
    $ ./run_dev_env.sh
    ```
    * Open localhost:8888 and you are good to go
* After you are finished your modification, don't forget to push your code to DAGsHub, and your dvc managed files to your dvc remote. In order to setup a dvc remote please refer to [this guide](https://dagshub.com/docs/getting-started/set-up-remote-storage-for-data-and-models/).
* Create a Pull Request on DAGsHub!
* 🐶
### TODO:
[] Web UI
[] Testing various datasets as basis for training
[] Testing various models for the data
[] Adding qualitative tests for model performance (visually comparing 3d image outputs)
