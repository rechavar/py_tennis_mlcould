from azureml.core import Workspace, datastore

ws = Workspace.from_config()

datastore = ws.get_default_datastore()
datastore.upload(src_dir='./data',
                target_path='datasets/pytennis',
                overwrite= True
                
                )

