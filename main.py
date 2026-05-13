"""
Информационно-аналитическая система для Автобусного парка
"""

import warnings
import dash
from dash import html, dcc

from config import GITHUB_CSV_URL
from utils.data_loader import DataLoader
from modules import RevenueForecast
from ui.layout import get_layout, render_quality_tab, render_forecast_tab, render_optimization_tab
from ui.callbacks import register_callbacks

warnings.filterwarnings('ignore')

# Загрузка данных
print("🔄 Загрузка данных с GitHub...")
df_bus = DataLoader.load_bus_data(GITHUB_CSV_URL)
print(f"✅ Данные загружены: {len(df_bus)} записей")

# Модель прогнозирования
forecast_model = RevenueForecast(df_bus)
forecast_model.train_linear()
print(f"📈 Модель обучена: R² = {forecast_model.r2:.4f}, MAE = {forecast_model.mae:.2f}")

# Приложение
app = dash.Dash(__name__, title="ИАС Автобусный парк")
app.config.suppress_callback_exceptions = True

app.layout = get_layout()
register_callbacks(app, forecast_model)

# Callback для переключения вкладок
@app.callback(
    dash.dependencies.Output("tabs-content", "children"),
    dash.dependencies.Input("main-tabs", "value")
)
def render_content(tab):
    if tab == "tab-quality":
        return render_quality_tab()
    elif tab == "tab-forecast":
        return render_forecast_tab(df_bus)
    elif tab == "tab-optimization":
        return render_optimization_tab()
    return html.Div()

if __name__ == "__main__":
    app.run(debug=True, port=8050)