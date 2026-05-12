"""
Информационно-аналитическая система для Автобусного парка
Вариант 3: Автобусный парк

Модули:
1. Оценка технического уровня автобусов
2. Прогнозирование дневной выручки (на основе Bus.csv)
3. Оптимизация распределения автобусов по маршрутам
"""

import pandas as pd
import numpy as np
import requests
from io import StringIO
import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objs as go
import plotly.express as px
from scipy.optimize import linprog
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# ЗАГРУЗКА ДАННЫХ ИЗ GITHUB
# ============================================================

# Прямая ссылка на raw-файл Bus.csv в GitHub
GITHUB_CSV_URL = "https://raw.githubusercontent.com/Ju5TShur1k/Python_Kursovaya/refs/heads/main/Bus.csv"

def load_bus_data():
    """Загрузка данных Bus.csv с GitHub"""
    response = requests.get(GITHUB_CSV_URL)
    response.raise_for_status()  # Проверка что всё ок
    df = pd.read_csv(StringIO(response.text))
    return df

# Загружаем данные
df_bus = load_bus_data()
print(f"✅ Данные загружены: {len(df_bus)} записей")
print(f"📊 Признаки: {list(df_bus.columns)}")

# ============================================================
# КЛАССЫ ДЛЯ РЕАЛИЗАЦИИ ФУНКЦИОНАЛА
# ============================================================

class QualityEvaluator:
    """Класс для оценки технического уровня автобусов"""
    
    # Характеристики для разных типов автобусов
    CHARACTERISTICS = {
        "Междугородние автобусы": {
            "stimulators": ["max_speed", "capacity", "resource", "comfort"],
            "destimulators": ["fuel_consumption"],
            "labels": {
                "max_speed": "Макс. скорость (км/ч)",
                "capacity": "Вместимость (чел)",
                "resource": "Ресурс (тыс. км)",
                "comfort": "Комфорт (баллы)",
                "fuel_consumption": "Расход топлива (л/100км)"
            },
            "etalon": {"max_speed": 120, "capacity": 55, "resource": 800, 
                      "comfort": 10, "fuel_consumption": 22}
        },
        "Автобусы малой вместимости": {
            "stimulators": ["maneuverability", "eco_class", "resource"],
            "destimulators": ["fuel_consumption"],
            "labels": {
                "maneuverability": "Маневренность (баллы)",
                "eco_class": "Эко-класс (баллы)",
                "resource": "Ресурс (тыс. км)",
                "fuel_consumption": "Расход топлива (л/100км)"
            },
            "etalon": {"maneuverability": 9, "eco_class": 5, "resource": 600, 
                      "fuel_consumption": 18}
        },
        "Спецтранспорт (эвакуаторы)": {
            "stimulators": ["load_capacity", "evac_speed", "maneuverability"],
            "destimulators": ["fuel_consumption"],
            "labels": {
                "load_capacity": "Грузоподъемность (т)",
                "evac_speed": "Скорость эвакуации (км/ч)",
                "maneuverability": "Маневренность (баллы)",
                "fuel_consumption": "Расход топлива (л/100км)"
            },
            "etalon": {"load_capacity": 12, "evac_speed": 80, 
                      "maneuverability": 8, "fuel_consumption": 25}
        }
    }
    
    @staticmethod
    def calculate_quality(sample, etalon, stimulators, destimulators):
        """Расчет комплексного показателя качества"""
        ratios = []
        for p in stimulators:
            if p in sample and p in etalon:
                ratios.append(sample[p] / etalon[p])
        for p in destimulators:
            if p in sample and p in etalon:
                ratios.append(etalon[p] / sample[p])
        
        if not ratios:
            return 0
        
        geom_mean = np.exp(np.mean(np.log(ratios)))
        return geom_mean * 100
    
    @staticmethod
    def get_normalized_values(sample, etalon, stimulators, destimulators):
        """Получение нормированных значений для радиальной диаграммы"""
        normalized = {}
        for p in stimulators:
            if p in sample and p in etalon:
                normalized[p] = sample[p] / etalon[p]
        for p in destimulators:
            if p in sample and p in etalon:
                normalized[p] = etalon[p] / sample[p]
        return normalized


class RevenueForecast:
    """Класс для прогнозирования дневной выручки"""
    
    def __init__(self, df):
        self.df = df
        self.model = None
        self.poly_features = None
        self.is_polynomial = False
        self.scaler = None
        
        # Определение признаков и целевой переменной
        self.feature_names = ['fuel_price', 'avg_route_length', 'is_holiday', 
                              'bus_count', 'weather_condition']
        self.target_name = 'daily_revenue'
        
        # Описания признаков
        self.feature_labels = {
            'fuel_price': 'Цена топлива (руб./л)',
            'avg_route_length': 'Средняя протяжённость маршрута (км)',
            'is_holiday': 'Праздник/выходной (1=да, 0=нет)',
            'bus_count': 'Количество работающих автобусов',
            'weather_condition': 'Погода (1=хорошая, 0=плохая)'
        }
        
    def train_linear(self):
        """Обучение линейной регрессии"""
        X = self.df[self.feature_names].values
        y = self.df[self.target_name].values
        
        self.model = LinearRegression()
        self.model.fit(X, y)
        self.is_polynomial = False
        self.poly_features = None
        
        # Расчет метрик
        y_pred = self.model.predict(X)
        self.r2 = r2_score(y, y_pred)
        self.mae = mean_absolute_error(y, y_pred)
        self.rmse = np.sqrt(mean_absolute_error(y, y_pred))
        
        # Коэффициенты модели
        self.coefficients = dict(zip(self.feature_names, self.model.coef_))
        self.intercept = self.model.intercept_
        
        return self.model, self.r2, self.mae
    
    def train_polynomial(self, degree=2):
        """Обучение полиномиальной регрессии"""
        X = self.df[self.feature_names].values
        y = self.df[self.target_name].values
        
        self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = self.poly_features.fit_transform(X)
        
        self.model = LinearRegression()
        self.model.fit(X_poly, y)
        self.is_polynomial = True
        
        y_pred = self.model.predict(X_poly)
        self.r2 = r2_score(y, y_pred)
        self.mae = mean_absolute_error(y, y_pred)
        self.rmse = np.sqrt(mean_absolute_error(y, y_pred))
        
        return self.model, self.r2, self.mae
    
    def predict(self, features_dict):
        """Прогноз для заданных признаков"""
        features_list = [features_dict[f] for f in self.feature_names]
        
        if self.is_polynomial and self.poly_features:
            X_pred = self.poly_features.transform([features_list])
        else:
            X_pred = [features_list]
        
        return self.model.predict(X_pred)[0]
    
    def get_dependence(self, fixed_features, varying_feature, range_values):
        """Получение зависимости целевой переменной от изменяемого признака"""
        predictions = []
        for val in range_values:
            features = fixed_features.copy()
            features[varying_feature] = val
            predictions.append(self.predict(features))
        return predictions
    
    def get_feature_importance(self):
        """Получение важности признаков (только для линейной модели)"""
        if not self.is_polynomial and self.model:
            importance = np.abs(self.model.coef_)
            importance = importance / importance.sum()
            return dict(zip(self.feature_names, importance))
        return None


class TransportOptimizer:
    """Класс для оптимизации распределения автобусов по маршрутам"""
    
    @staticmethod
    def solve_transport_problem(cost_matrix, supply, demand):
        """
        Решение транспортной задачи методом линейного программирования
        cost_matrix: матрица затрат (автобусы x маршруты)
        supply: запасы автобусов
        demand: потребность в рейсах
        """
        n_supply = len(supply)
        n_demand = len(demand)
        
        total_supply = sum(supply)
        total_demand = sum(demand)
        
        # Приведение к сбалансированной задаче
        if total_supply != total_demand:
            if total_supply > total_demand:
                new_demand = demand + [total_supply - total_demand]
                new_cost = np.zeros((n_supply, n_demand + 1))
                new_cost[:, :n_demand] = cost_matrix
                cost_matrix = new_cost
                demand = new_demand
                n_demand += 1
            else:
                new_supply = supply + [total_demand - total_supply]
                new_cost = np.zeros((n_supply + 1, n_demand))
                new_cost[:n_supply, :] = cost_matrix
                cost_matrix = new_cost
                supply = new_supply
                n_supply += 1
        
        n_vars = n_supply * n_demand
        c = cost_matrix.flatten()
        
        # Ограничения по поставщикам
        A_eq_supply = []
        b_eq_supply = []
        for i in range(n_supply):
            row = [0] * n_vars
            for j in range(n_demand):
                row[i * n_demand + j] = 1
            A_eq_supply.append(row)
            b_eq_supply.append(supply[i])
        
        # Ограничения по потребителям
        A_eq_demand = []
        b_eq_demand = []
        for j in range(n_demand):
            row = [0] * n_vars
            for i in range(n_supply):
                row[i * n_demand + j] = 1
            A_eq_demand.append(row)
            b_eq_demand.append(demand[j])
        
        A_eq = np.array(A_eq_supply + A_eq_demand)
        b_eq = np.array(b_eq_supply + b_eq_demand)
        
        bounds = [(0, None) for _ in range(n_vars)]
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if result.success:
            solution = result.x.reshape(n_supply, n_demand)
            return {
                'success': True,
                'solution': solution,
                'total_cost': result.fun,
                'n_supply': n_supply,
                'n_demand': n_demand
            }
        else:
            return {'success': False, 'message': 'Решение не найдено'}


# ============================================================
# ДАННЫЕ ДЛЯ ПРИМЕРОВ (ОЦЕНКА КАЧЕСТВА И ОПТИМИЗАЦИЯ)
# ============================================================

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

# Данные для транспортной задачи
COST_MATRIX = np.array([
    [450, 520, 380, 490],
    [480, 500, 420, 510],
    [430, 540, 390, 470],
    [500, 480, 440, 530],
    [460, 510, 410, 500]
])
SUPPLY = [8, 7, 6, 5, 8]
DEMAND = [10, 8, 7, 9]
ROUTE_NAMES = ['Маршрут №101', 'Маршрут №102', 'Маршрут №103', 'Маршрут №104']
BUS_NAMES = ['Автобус A', 'Автобус B', 'Автобус C', 'Автобус D', 'Автобус E']


# ============================================================
# ИНИЦИАЛИЗАЦИЯ МОДЕЛИ ПРОГНОЗИРОВАНИЯ
# ============================================================

forecast_model = RevenueForecast(df_bus)
forecast_model.train_linear()  # Обучаем модель по умолчанию

print(f"📈 Модель обучена: R² = {forecast_model.r2:.4f}, MAE = {forecast_model.mae:.2f}")

# ============================================================
# СОЗДАНИЕ DASH ПРИЛОЖЕНИЯ
# ============================================================

app = dash.Dash(__name__, title="ИАС Автобусный парк")
app.config.suppress_callback_exceptions = True

# Стили
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

app.layout = html.Div([
    html.H1("🚍 Информационно-аналитическая система", 
            style={'textAlign': 'center', 'color': '#2c3e50', 'padding': '20px'}),
    html.H3("Автобусный парк — управление качеством, прогнозирование выручки и оптимизация",
            style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '30px'}),
    
    dcc.Tabs(id="main-tabs", value="tab-quality", children=[
        dcc.Tab(label="📊 Оценка технического уровня", value="tab-quality",
                style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
        dcc.Tab(label="📈 Прогноз дневной выручки", value="tab-forecast",
                style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
        dcc.Tab(label="🚌 Оптимизация перевозок", value="tab-optimization",
                style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
    ]),
    
    html.Div(id="tabs-content"),
], style={'fontFamily': 'Arial, sans-serif', 'margin': '0 auto', 'maxWidth': '1400px'})


# ============================================================
# РЕНДЕР ВКЛАДОК
# ============================================================

@app.callback(Output("tabs-content", "children"), Input("main-tabs", "value"))
def render_content(tab):
    if tab == "tab-quality":
        return render_quality_tab()
    elif tab == "tab-forecast":
        return render_forecast_tab()
    elif tab == "tab-optimization":
        return render_optimization_tab()
    return html.Div()


def render_quality_tab():
    return html.Div([
        html.Div([
            html.Div([
                html.H4("Выбор типа автобусов", style={'color': '#2c3e50'}),
                dcc.Dropdown(
                    id="bus-type-select",
                    options=[
                        {"label": "🚌 Междугородние автобусы", "value": "Междугородние автобусы"},
                        {"label": "🚐 Автобусы малой вместимости", "value": "Автобусы малой вместимости"},
                        {"label": "🛻 Спецтранспорт (эвакуаторы)", "value": "Спецтранспорт (эвакуаторы)"}
                    ],
                    value="Междугородние автобусы",
                    style={'marginBottom': '20px'}
                ),
                html.Button("📊 Рассчитать технический уровень", id="btn-calc-quality",
                           style={'margin': '10px 0', 'padding': '10px 20px',
                                  'backgroundColor': '#27ae60', 'color': 'white', 'border': 'none',
                                  'borderRadius': '5px', 'cursor': 'pointer'}),
            ], className="six columns", style={'padding': '20px', 'backgroundColor': '#ecf0f1', 
                                               'borderRadius': '10px', 'margin': '10px'}),
            
            html.Div([
                html.H4("Результаты оценки", style={'color': '#2c3e50'}),
                html.Div(id="quality-results-text"),
            ], className="six columns", style={'padding': '20px', 'backgroundColor': '#ecf0f1',
                                               'borderRadius': '10px', 'margin': '10px'}),
        ], className="row"),
        
        html.Div([
            html.Div([
                html.H4("📡 Радиальная диаграмма", style={'textAlign': 'center'}),
                dcc.Graph(id="radar-chart")
            ], className="six columns", style={'padding': '10px'}),
            
            html.Div([
                html.H4("📊 Технический уровень", style={'textAlign': 'center'}),
                dcc.Graph(id="bar-chart")
            ], className="six columns", style={'padding': '10px'}),
        ], className="row"),
    ])


def render_forecast_tab():
    return html.Div([
        html.Div([
            html.Div([
                html.H4("Настройки модели", style={'color': '#2c3e50'}),
                dcc.RadioItems(
                    id="model-type",
                    options=[
                        {"label": "Линейная регрессия", "value": "linear"},
                        {"label": "Полиномиальная регрессия (степень 2)", "value": "polynomial"}
                    ],
                    value="linear",
                    style={'marginBottom': '20px'}
                ),
                
                html.Button("🔄 Обучить модель", id="btn-train-model",
                           style={'margin': '10px 0', 'padding': '10px 20px',
                                  'backgroundColor': '#3498db', 'color': 'white', 'border': 'none',
                                  'borderRadius': '5px', 'cursor': 'pointer'}),
                
                html.Div(id="model-metrics", style={'marginTop': '20px', 'padding': '15px',
                                                    'backgroundColor': '#d5f5e3', 'borderRadius': '5px'}),
                
                html.H4("Ручной прогноз выручки", style={'marginTop': '30px', 'color': '#2c3e50'}),
                html.Div([
                    html.Label("💰 Цена топлива (руб./л):"),
                    dcc.Input(id="pred-fuel", type="number", value=50, step=0.5,
                             style={'width': '100%', 'marginBottom': '10px'}),
                    html.Label("🛣️ Средняя протяжённость маршрута (км):"),
                    dcc.Input(id="pred-route", type="number", value=25, step=1,
                             style={'width': '100%', 'marginBottom': '10px'}),
                    html.Label("📅 Праздник/выходной (1=да, 0=нет):"),
                    dcc.Input(id="pred-holiday", type="number", value=0, step=1,
                             style={'width': '100%', 'marginBottom': '10px'}),
                    html.Label("🚌 Количество автобусов:"),
                    dcc.Input(id="pred-buses", type="number", value=30, step=1,
                             style={'width': '100%', 'marginBottom': '10px'}),
                    html.Label("☁️ Погода (1=хорошая, 0=плохая):"),
                    dcc.Input(id="pred-weather", type="number", value=1, step=1,
                             style={'width': '100%', 'marginBottom': '10px'}),
                    html.Button("📈 Получить прогноз", id="btn-predict",
                               style={'margin': '10px 0', 'padding': '10px 20px',
                                      'backgroundColor': '#27ae60', 'color': 'white', 
                                      'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                    html.Div(id="prediction-result", style={'marginTop': '15px', 'fontSize': '20px',
                                                            'fontWeight': 'bold', 'color': '#2980b9'}),
                ], style={'padding': '15px', 'backgroundColor': '#f9e79f', 'borderRadius': '10px'}),
            ], className="six columns", style={'padding': '20px', 'backgroundColor': '#ecf0f1',
                                               'borderRadius': '10px', 'margin': '10px'}),
            
            html.Div([
                html.H4("Анализ зависимости выручки", style={'color': '#2c3e50'}),
                html.Label("Выберите признак для анализа:"),
                dcc.Dropdown(
                    id="varying-feature",
                    options=[
                        {"label": "Цена топлива", "value": "fuel_price"},
                        {"label": "Протяжённость маршрута", "value": "avg_route_length"},
                        {"label": "Количество автобусов", "value": "bus_count"}
                    ],
                    value="fuel_price",
                    style={'marginBottom': '10px'}
                ),
                html.Label("Диапазон значений:"),
                html.Div([
                    dcc.Input(id="range-start", type="number", value=40,
                             style={'width': '45%', 'marginRight': '5%'}),
                    dcc.Input(id="range-end", type="number", value=60,
                             style={'width': '45%'})
                ], style={'marginBottom': '10px'}),
                html.Button("📊 Построить график зависимости", id="btn-plot-dependence",
                           style={'margin': '10px 0', 'padding': '10px 20px',
                                  'backgroundColor': '#8e44ad', 'color': 'white', 
                                  'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                dcc.Graph(id="dependence-graph", style={'marginTop': '20px'})
            ], className="six columns", style={'padding': '20px', 'backgroundColor': '#ecf0f1',
                                               'borderRadius': '10px', 'margin': '10px'}),
        ], className="row"),
        
        html.Div([
            html.H4("📋 Исходные данные (первые 20 записей)", style={'marginLeft': '10px'}),
            dash_table.DataTable(
                id="forecast-data-table",
                columns=[{"name": i, "id": i} for i in df_bus.columns],
                data=df_bus.head(20).to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'center'},
                page_size=10
            )
        ], style={'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '10px', 'margin': '10px'})
    ])


def render_optimization_tab():
    return html.Div([
        html.Div([
            html.Div([
                html.H4("Параметры транспортной задачи", style={'color': '#2c3e50'}),
                html.H5("Матрица затрат (руб./рейс)"),
                dash_table.DataTable(
                    id="cost-matrix-table",
                    columns=[{"name": f"Маршрут {i+1}", "id": f"col_{i}"} for i in range(4)],
                    data=[{f"col_{j}": COST_MATRIX[i][j] for j in range(4)} 
                          for i in range(5)],
                    editable=True,
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'center', 'minWidth': '80px'}
                ),
                
                html.H5("Запасы автобусов (рейсов)", style={'marginTop': '20px'}),
                html.Div([
                    html.Div([html.Label(f"{BUS_NAMES[i]}:"), 
                             dcc.Input(id=f"supply_{i}", type="number", value=SUPPLY[i],
                                      style={'width': '80px', 'marginLeft': '10px'})])
                    for i in range(5)
                ]),
                
                html.H5("Потребность маршрутов (рейсов)", style={'marginTop': '20px'}),
                html.Div([
                    html.Div([html.Label(f"{ROUTE_NAMES[i]}:"), 
                             dcc.Input(id=f"demand_{i}", type="number", value=DEMAND[i],
                                      style={'width': '80px', 'marginLeft': '10px'})])
                    for i in range(4)
                ]),
                
                html.Button("🚀 Оптимизировать распределение", id="btn-optimize",
                           style={'margin': '20px 0', 'padding': '12px 25px',
                                  'backgroundColor': '#27ae60', 'color': 'white', 'border': 'none',
                                  'borderRadius': '5px', 'cursor': 'pointer', 'fontSize': '16px'}),
            ], className="six columns", style={'padding': '20px', 'backgroundColor': '#ecf0f1',
                                               'borderRadius': '10px', 'margin': '10px'}),
            
            html.Div([
                html.H4("Результаты оптимизации", style={'color': '#2c3e50'}),
                html.Div(id="optimization-results"),
            ], className="six columns", style={'padding': '20px', 'backgroundColor': '#ecf0f1',
                                               'borderRadius': '10px', 'margin': '10px'}),
        ], className="row"),
    ])


# ============================================================
# CALLBACK ДЛЯ ОЦЕНКИ КАЧЕСТВА
# ============================================================

@app.callback(
    [Output("radar-chart", "figure"),
     Output("bar-chart", "figure"),
     Output("quality-results-text", "children")],
    [Input("btn-calc-quality", "n_clicks")],
    [State("bus-type-select", "value")]
)
def update_quality(n_clicks, bus_type):
    if bus_type is None:
        bus_type = "Междугородние автобусы"
    
    buses = SAMPLE_BUSES.get(bus_type, [])
    characteristics = QualityEvaluator.CHARACTERISTICS.get(bus_type, {})
    etalon = characteristics.get("etalon", {})
    stimulators = characteristics.get("stimulators", [])
    destimulators = characteristics.get("destimulators", [])
    labels = characteristics.get("labels", {})
    
    results = []
    for bus in buses:
        quality = QualityEvaluator.calculate_quality(bus, etalon, stimulators, destimulators)
        results.append({"name": bus["name"], "quality": quality, "data": bus})
    
    results.sort(key=lambda x: x["quality"], reverse=True)
    
    # Радиальная диаграмма
    if results:
        best_bus = results[0]
        normalized = QualityEvaluator.get_normalized_values(
            best_bus["data"], etalon, stimulators, destimulators
        )
        
        radar_fig = go.Figure()
        radar_fig.add_trace(go.Scatterpolar(
            r=[normalized.get(p, 0) for p in stimulators + destimulators],
            theta=[labels.get(p, p) for p in stimulators + destimulators],
            fill='toself',
            name=best_bus["name"]
        ))
        radar_fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1.5])),
            title=f"Нормированные показатели: {best_bus['name']}"
        )
    else:
        radar_fig = go.Figure().update_layout(title="Нет данных")
    
    # Столбчатая диаграмма
    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(
        x=[r["name"] for r in results],
        y=[r["quality"] for r in results],
        marker_color=['#27ae60' if i == 0 else '#3498db' for i in range(len(results))],
        text=[f"{r['quality']:.1f}%" for r in results],
        textposition='auto'
    ))
    bar_fig.update_layout(
        title="Технический уровень автобусов",
        xaxis_title="Модель автобуса",
        yaxis_title="Технический уровень (%)",
        yaxis_range=[0, 120]
    )
    
    results_text = html.Div([
        html.H5(f"Результаты оценки для: {bus_type}"),
        html.H6("Эталонный образец:"),
        html.P(", ".join([f"{labels.get(k,k)}: {v}" for k,v in etalon.items()])),
        html.H6("Рейтинг образцов:"),
        html.Ol([html.Li(f"{r['name']}: {r['quality']:.1f}%") for r in results])
    ])
    
    return radar_fig, bar_fig, results_text


# ============================================================
# CALLBACK ДЛЯ ПРОГНОЗИРОВАНИЯ
# ============================================================

@app.callback(
    [Output("model-metrics", "children"),
     Output("prediction-result", "children")],
    [Input("btn-train-model", "n_clicks"),
     Input("btn-predict", "n_clicks")],
    [State("model-type", "value"),
     State("pred-fuel", "value"),
     State("pred-route", "value"),
     State("pred-holiday", "value"),
     State("pred-buses", "value"),
     State("pred-weather", "value")]
)
def update_forecast(train_clicks, pred_clicks, model_type, fuel, route, holiday, buses, weather):
    global forecast_model
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
    
    if "btn-train-model" in trigger_id:
        if model_type == "linear":
            forecast_model.train_linear()
            model_name = "Линейная регрессия"
        else:
            forecast_model.train_polynomial(degree=2)
            model_name = "Полиномиальная регрессия"
        
        metrics = html.Div([
            html.H5(f"📈 Модель: {model_name}"),
            html.P(f"R² (качество модели): {forecast_model.r2:.4f}"),
            html.P(f"MAE (средняя ошибка): {forecast_model.mae:.2f} тыс. руб."),
            html.P("✅ Модель успешно обучена")
        ])
        
        if "btn-predict" not in trigger_id:
            return metrics, html.Div()
    
    if "btn-predict" in trigger_id:
        try:
            prediction = forecast_model.predict({
                'fuel_price': fuel or 50,
                'avg_route_length': route or 25,
                'is_holiday': holiday or 0,
                'bus_count': buses or 30,
                'weather_condition': weather or 1
            })
            pred_result = html.Div([
                f"💰 Прогноз дневной выручки: {prediction:.0f} тыс. руб."
            ])
        except Exception as e:
            pred_result = html.Div(f"Ошибка: {str(e)}", style={'color': 'red'})
        
        metrics = html.Div([
            html.H5("📈 Модель готова"),
            html.P(f"R²: {forecast_model.r2:.4f}"),
            html.P(f"MAE: {forecast_model.mae:.2f} тыс. руб.")
        ])
        return metrics, pred_result
    
    return html.Div("Нажмите 'Обучить модель'"), html.Div()


@app.callback(
    Output("dependence-graph", "figure"),
    Input("btn-plot-dependence", "n_clicks"),
    [State("varying-feature", "value"),
     State("range-start", "value"),
     State("range-end", "value")]
)
def plot_dependence(n_clicks, varying_feature, range_start, range_end):
    if not n_clicks:
        return go.Figure().update_layout(title="Нажмите 'Построить график'")
    
    if range_start is None or range_end is None or range_start >= range_end:
        return go.Figure().update_layout(title="Укажите корректный диапазон")
    
    # Фиксированные значения (средние по данным)
    fixed = {
        'fuel_price': 50,
        'avg_route_length': 25,
        'is_holiday': 0,
        'bus_count': 30,
        'weather_condition': 1
    }
    
    range_values = np.linspace(range_start, range_end, 50)
    predictions = forecast_model.get_dependence(fixed, varying_feature, range_values)
    
    feature_names = {
        'fuel_price': 'Цена топлива (руб./л)',
        'avg_route_length': 'Протяжённость маршрута (км)',
        'bus_count': 'Количество автобусов'
    }
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=range_values,
        y=predictions,
        mode='lines+markers',
        name='Дневная выручка',
        line=dict(color='#3498db', width=3)
    ))
    
    fig.update_layout(
        title=f"Зависимость выручки от {feature_names.get(varying_feature, varying_feature)}",
        xaxis_title=feature_names.get(varying_feature, varying_feature),
        yaxis_title="Дневная выручка (тыс. руб.)",
        template="plotly_white"
    )
    
    return fig


# ============================================================
# CALLBACK ДЛЯ ОПТИМИЗАЦИИ
# ============================================================

@app.callback(
    Output("optimization-results", "children"),
    Input("btn-optimize", "n_clicks"),
    [State(f"supply_{i}", "value") for i in range(5)] +
    [State(f"demand_{i}", "value") for i in range(4)] +
    [State(f"cost-matrix-table", "data")]
)
def solve_optimization(n_clicks, s0, s1, s2, s3, s4, d0, d1, d2, d3, cost_data):
    if not n_clicks:
        return html.Div("Нажмите 'Оптимизировать'")
    
    supply = [s0 or 0, s1 or 0, s2 or 0, s3 or 0, s4 or 0]
    demand = [d0 or 0, d1 or 0, d2 or 0, d3 or 0]
    
    if sum(supply) == 0 or sum(demand) == 0:
        return html.Div("Ошибка: суммы должны быть > 0", style={'color': 'red'})
    
    if cost_data:
        cost_matrix = np.array([[row[f"col_{j}"] for j in range(4)] for row in cost_data])
    else:
        cost_matrix = COST_MATRIX
    
    result = TransportOptimizer.solve_transport_problem(cost_matrix, supply, demand)
    
    if result['success']:
        solution = result['solution']
        total_cost = result['total_cost']
        n_buses = result['n_supply']
        n_routes = result['n_demand']
        
        bus_names = BUS_NAMES + ["Фиктивный"] if n_buses > 5 else BUS_NAMES[:n_buses]
        route_names = ROUTE_NAMES + ["Фиктивный"] if n_routes > 4 else ROUTE_NAMES[:n_routes]
        
        table_data = []
        for i in range(n_buses):
            row = {"Автобус": bus_names[i]}
            for j in range(n_routes):
                row[route_names[j]] = f"{solution[i][j]:.1f}"
            table_data.append(row)
        
        table = dash_table.DataTable(
            columns=[{"name": "Автобус", "id": "Автобус"}] + 
                    [{"name": r, "id": r} for r in route_names],
            data=table_data,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center'},
            style_header={'backgroundColor': '#3498db', 'color': 'white'}
        )
        
        return html.Div([
            html.H5(f"✅ Минимальные затраты: {total_cost:,.0f} руб."),
            html.H6("Оптимальное распределение рейсов:"),
            table
        ])
    else:
        return html.Div(f"❌ Ошибка: {result.get('message', 'Решение не найдено')}", style={'color': 'red'})


# ============================================================
# ЗАПУСК ПРИЛОЖЕНИЯ
# ============================================================

if __name__ == "__main__":
    app.run(debug=True, port=8050)