import requests
from bs4 import BeautifulSoup
import os

def download_files(url, folder_name):
    # Create folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    links = soup.find_all('a', text='Download')

    for link in links:
        partial_url = link.get('href')
        download_url = f'https://whoicf2.whoi.edu/{partial_url}'
        file_name = os.path.join(folder_name, download_url.split('/')[-1])

        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(file_name, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f'Downloaded {file_name}')

# URL of the webpage and folder name
url = 'https://whoicf2.whoi.edu/science/B/whalesounds/fullCuts.cfm?SP=BE7A&YR=97'
folder_name = 'orca'

download_files(url, folder_name)
