from os import environ
from azureml.core import webservice
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig, Model
from azureml.core.environment import Environment
from azureml.core import Workspace, Workspace
from azureml.core.compute import ComputeTarget, AksCompute
from azureml.exceptions import ComputeTargetException

ws = Workspace.from_config(path='./.azureml', _file_name='config.json')


aks_name = "pytennis-gpu"


try:
    aks_target = ComputeTarget(workspace=ws, name=aks_name)
    print('Found existing compute target')
except ComputeTargetException:
    print('Creating a new compute target...')
    prov_config = AksCompute.provisioning_configuration(vm_size="Standard_NC6")

    aks_target = ComputeTarget.create(
        workspace=ws, name=aks_name, provisioning_configuration=prov_config
    )

    aks_target.wait_for_completion(show_output=True)



model = Model(ws,name ='pyTennis', version=2)

env = Environment.from_pip_requirements(
    name='pytennis-aml-env',
    file_path='./.azureml/requirements.txt'
)

inferece_config = InferenceConfig(entry_script='./src/score.py', environment=env)

gpu_aks_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)


web_service = Model.deploy(workspace=ws,
                        name='pytennis-ml-17',
                        models=[model],
                        inference_config=inferece_config,
                        deployment_target=aks_target,
                        deployment_config=gpu_aks_config)

web_service.wait_for_deployment(show_output=True)
print(web_service.scoring_uri)