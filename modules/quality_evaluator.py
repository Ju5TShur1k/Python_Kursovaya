"""
Модуль оценки технического уровня автобусов
"""

import numpy as np


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