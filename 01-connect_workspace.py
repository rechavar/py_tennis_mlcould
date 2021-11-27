from os import name
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core import Workspace

interactive_auth = InteractiveLoginAuthentication(
    tenant_id='99e1e721-7184-498e-8aff-b2ad4e53c1c2')

ws = Workspace.get(name='mlw-esp-udea-re-2',
                    subscription_id= '8e761af4-60f3-45a5-b326-4f2abac4f0a0',
                    resource_group = 'pytennis-great-gpu',
                    location = 'eastus',
                    auth = interactive_auth
                    )

ws.write_config(path = '.azureml')
