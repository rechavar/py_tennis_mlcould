from azureml.core import Workspace
from azureml.core.model import Model

if __name__ == "__main__":

    ws = Workspace.from_config(path='./.azureml', _file_name= 'config.json')
    
    model = Model.register(ws, model_name="pyTennis",
                            model_path="./model-dsbowl2018-1.h5")
