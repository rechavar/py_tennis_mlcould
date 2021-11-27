from azureml.core.webservice import Webservice
from azureml.core.model import InferenceConfig, Model
from azureml.core.environment import Environment
from azureml.core import Workspace, Workspace

ws = Workspace.from_config(path='./.azureml', _file_name='config.json')

model = Model(ws,name ='pyTennis', version=2)

env = Environment.from_pip_requirements(
    name='pytennis-aml-env',
    file_path='./.azureml/requirements.txt'
)

