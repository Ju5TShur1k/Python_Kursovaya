"""
Обработчики событий (callbacks)
"""

import dash
import numpy as np
from dash import dcc, html, dash_table, Input, Output, State
import plotly.graph_objs as go

from modules import QualityEvaluator, RevenueForecast, TransportOptimizer
from config import SAMPLE_BUSES, BUS_NAMES, ROUTE_NAMES


def register_callbacks(app, forecast_model):
    """Регистрация всех callbacks"""
    
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
    
    @app.callback(
        Output("optimization-results", "children"),
        Input("btn-optimize", "n_clicks"),
        [State(f"supply_{i}", "value") for i in range(5)] +
        [State(f"demand_{i}", "value") for i in range(4)] +
        [State("cost-matrix-table", "data")]
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
            from config import COST_MATRIX
            cost_matrix = np.array(COST_MATRIX)
        
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