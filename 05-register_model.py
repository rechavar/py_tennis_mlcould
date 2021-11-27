from azureml.core import Workspace
from azureml.core.model import Model

if __name__ == "__main__":

    ws = Workspace.from_config(path='./.azureml', _file_name= 'config.json')
    
    model = Model.register(ws, model_name="pyTennis",
                            tags = {'area' : 'monografia especializacion',
                            'img_size':'512',
                            'number_chanels':'1'},
                            model_path="./outputs/model-dsbowl2018-1.h5")
