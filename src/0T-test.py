import requests
import os

def download_h5(file_name = 'model-dsbowl2018-1.h5', 
                url = 'https://s3.amazonaws.com/rlx/model-dsbowl2018-1.h5'):

    if os.path.exists(file_name):
        print('File already on dir, skipping download')
        return
    
    r = requests.get(url, allow_redirects=True)

    open(file_name, 'wb').write(r.content)

if __name__ == '__main__':
    download_h5()
    