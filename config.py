"""
Конфигурационный файл
"""

# Ссылка на raw-файл Bus.csv в GitHub
GITHUB_CSV_URL = "https://raw.githubusercontent.com/Ju5TShur1k/Python_Kursovaya/refs/heads/main/Bus.csv"

# Данные для транспортной задачи
COST_MATRIX = [
    [450, 520, 380, 490],
    [480, 500, 420, 510],
    [430, 540, 390, 470],
    [500, 480, 440, 530],
    [460, 510, 410, 500]
]
SUPPLY = [8, 7, 6, 5, 8]
DEMAND = [10, 8, 7, 9]
ROUTE_NAMES = ['Маршрут №101', 'Маршрут №102', 'Маршрут №103', 'Маршрут №104']
BUS_NAMES = ['Автобус A', 'Автобус B', 'Автобус C', 'Автобус D', 'Автобус E']

# Образцы автобусов для оценки качества
SAMPLE_BUSES = {
    "Междугородние автобусы": [
        {"name": "Volvo 9700", "max_speed": 125, "capacity": 57, "resource": 850, 
         "comfort": 9.5, "fuel_consumption": 21.5},
        {"name": "Neoplan Tourliner", "max_speed": 130, "capacity": 55, "resource": 820, 
         "comfort": 9.0, "fuel_consumption": 22.0},
        {"name": "Yutong ZK6122", "max_speed": 115, "capacity": 52, "resource": 750, 
         "comfort": 8.5, "fuel_consumption": 23.5},
        {"name": "MAN Lion's Coach", "max_speed": 128, "capacity": 56, "resource": 840, 
         "comfort": 9.2, "fuel_consumption": 21.8}
    ],
    "Автобусы малой вместимости": [
        {"name": "Mercedes Sprinter", "maneuverability": 9.2, "eco_class": 5, 
         "resource": 550, "fuel_consumption": 12.5},
        {"name": "Ford Transit", "maneuverability": 8.8, "eco_class": 4, 
         "resource": 520, "fuel_consumption": 13.0},
        {"name": "GAZelle Next", "maneuverability": 8.5, "eco_class": 4, 
         "resource": 480, "fuel_consumption": 14.5}
    ],
    "Спецтранспорт (эвакуаторы)": [
        {"name": "MAN TGS", "load_capacity": 13, "evac_speed": 85, 
         "maneuverability": 7.5, "fuel_consumption": 28},
        {"name": "Scania R-Series", "load_capacity": 14, "evac_speed": 82, 
         "maneuverability": 7.2, "fuel_consumption": 29},
        {"name": "Volvo FH", "load_capacity": 12.5, "evac_speed": 80, 
         "maneuverability": 7.8, "fuel_consumption": 27.5}
    ]
}

# Стили для интерфейса
TAB_STYLE = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '12px',
    'fontWeight': 'bold',
    'backgroundColor': '#34495e',
    'color': 'white'
}

TAB_SELECTED_STYLE = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#3498db',
    'color': 'white',
    'padding': '12px',
    'fontWeight': 'bold'
}

APP_LAYOUT_STYLE = {
    'fontFamily': 'Arial, sans-serif',
    'margin': '0 auto',
    'maxWidth': '1400px'
}

CARD_STYLE = {
    'padding': '20px',
    'backgroundColor': '#ecf0f1',
    'borderRadius': '10px',
    'margin': '10px'
}

BUTTON_STYLES = {
    'primary': {
        'margin': '10px 0',
        'padding': '10px 20px',
        'backgroundColor': '#3498db',
        'color': 'white',
        'border': 'none',
        'borderRadius': '5px',
        'cursor': 'pointer'
    },
    'success': {
        'margin': '10px 0',
        'padding': '10px 20px',
        'backgroundColor': '#27ae60',
        'color': 'white',
        'border': 'none',
        'borderRadius': '5px',
        'cursor': 'pointer'
    }
}