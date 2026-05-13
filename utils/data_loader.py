"""
Модуль для загрузки данных с GitHub
"""

import requests
import pandas as pd
from io import StringIO


class DataLoader:
    """Класс для загрузки данных"""
    
    @staticmethod
    def load_bus_data(url):
        """Загрузка данных Bus.csv с GitHub"""
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        return df