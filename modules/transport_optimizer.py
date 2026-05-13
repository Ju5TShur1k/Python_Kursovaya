"""
Модуль оптимизации распределения автобусов по маршрутам
"""

import numpy as np
from scipy.optimize import linprog


class TransportOptimizer:
    """Класс для оптимизации распределения автобусов по маршрутам"""
    
    @staticmethod
    def solve_transport_problem(cost_matrix, supply, demand):
        """
        Решение транспортной задачи методом линейного программирования
        cost_matrix: матрица затрат (автобусы x маршруты)
        supply: запасы автобусов (количество рейсов)
        demand: потребность в рейсах на маршрутах
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
        
        # Ограничения по поставщикам (автобусам)
        A_eq_supply = []
        b_eq_supply = []
        for i in range(n_supply):
            row = [0] * n_vars
            for j in range(n_demand):
                row[i * n_demand + j] = 1
            A_eq_supply.append(row)
            b_eq_supply.append(supply[i])
        
        # Ограничения по потребителям (маршрутам)
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