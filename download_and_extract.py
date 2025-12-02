# download_and_extract.py
import os
import tarfile
import requests

def download_and_extract():
    # Если папки data нет
    if not os.path.exists("data"):
        print("Скачиваю индексы...")
        
        # Скачайте data.tar.gz с любого облачного хранилища
        # Или используйте этот временный URL
        url = "ВАШ_URL_НА_DATA.TAR.GZ"
        
        response = requests.get(url)
        with open("data.tar.gz", "wb") as f:
            f.write(response.content)
        
        # Распаковка
        with tarfile.open("data.tar.gz", "r:gz") as tar:
            tar.extractall()
        
        os.remove("data.tar.gz")
        print("✅ Индексы загружены")