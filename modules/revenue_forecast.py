import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

class RevenueForecast:
    """Класс для прогнозирования дневной выручки"""
    
    def __init__(self, df):
        self.df = df
        self.model = None
        self.poly_features = None
        self.is_polynomial = False
        self.scaler = None
        self.test_size = 0.2  # 20% данных на тест
        self.random_state = 42
        
        # Определение признаков и целевой переменной
        self.feature_names = ['fuel_price', 'avg_route_length', 'is_holiday', 
                              'bus_count', 'weather_condition']
        self.target_name = 'daily_revenue'
        
        # Разделение на обучающую и тестовую выборки
        self.X_train, self.X_test, self.y_train, self.y_test = self._split_data()
        
        # Описания признаков
        self.feature_labels = {
            'fuel_price': 'Цена топлива (руб./л)',
            'avg_route_length': 'Средняя протяжённость маршрута (км)',
            'is_holiday': 'Праздник/выходной (1=да, 0=нет)',
            'bus_count': 'Количество работающих автобусов',
            'weather_condition': 'Погода (1=хорошая, 0=плохая)'
        }
    
    def _split_data(self):
        """Разделение данных на обучающую и тестовую выборки"""
        X = self.df[self.feature_names].values
        y = self.df[self.target_name].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            shuffle=True  # перемешиваем данные
        )
        
        print(f"📊 Разделение данных: обучение {len(X_train)} записей, тест {len(X_test)} записей")
        return X_train, X_test, y_train, y_test
    
    def train_linear(self):
        """Обучение линейной регрессии на ТРЕНИРОВОЧНЫХ данных"""
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        self.is_polynomial = False
        self.poly_features = None
        
        # Оценка на ТЕСТОВЫХ данных (честная оценка)
        y_pred = self.model.predict(self.X_test)
        self.r2 = r2_score(self.y_test, y_pred)
        self.mae = mean_absolute_error(self.y_test, y_pred)
        
        # Сохраняем коэффициенты
        self.coefficients = dict(zip(self.feature_names, self.model.coef_))
        self.intercept = self.model.intercept_
        
        return self.model, self.r2, self.mae
    
    def train_polynomial(self, degree=2):
        """Обучение полиномиальной регрессии на ТРЕНИРОВОЧНЫХ данных"""
        # Нормализация
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # Полиномиальные признаки
        self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = self.poly_features.fit_transform(X_train_scaled)
        X_test_poly = self.poly_features.transform(X_test_scaled)
        
        # Обучение
        self.model = LinearRegression()
        self.model.fit(X_train_poly, self.y_train)
        self.is_polynomial = True
        
        # Оценка на ТЕСТОВЫХ данных
        y_pred = self.model.predict(X_test_poly)
        self.r2 = r2_score(self.y_test, y_pred)
        self.mae = mean_absolute_error(self.y_test, y_pred)
        
        return self.model, self.r2, self.mae
    
    def predict(self, features_dict):
        """Прогноз для заданных признаков"""
        features_list = [features_dict[f] for f in self.feature_names]
        
        if self.is_polynomial and self.poly_features:
            # Нормализуем и преобразуем
            features_scaled = self.scaler.transform([features_list])
            X_pred = self.poly_features.transform(features_scaled)
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