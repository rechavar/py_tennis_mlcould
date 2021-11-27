from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig, datastore, experiment, Dataset

if __name__ == "__main__":

    ws = Workspace.from_config(path='./.azureml', _file_name= 'config.json')

    datastore = ws.get_default_datastore()

    dataset = Dataset.File.from_files(path=(datastore, 'datasets/pytennis'))

    experiment = Experiment(workspace= ws, name= 'day1-experiment-train-pytennis')
    config = ScriptRunConfig(source_directory='./src',
                             script= 'train.py',
                             compute_target='pytennis-gpu',
                             arguments=[
                                 '--datasetpath', dataset.as_named_input('input').as_mount()
                             ])
    
    env = Environment.from_pip_requirements(
        name = 'pytennis_aml_tf',
        file_path = '.azureml/requirements.txt'
    )

    config.run_config.environment = env
    run = experiment.submit(config)
    aml_url = run.get_portal_url()
    print(aml_url)

