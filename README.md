# Streamlit Dashboard de Tickets TI

Este proyecto ahora esta orientado al analisis operativo de tickets del area de tecnologia a partir de un CSV real.

## Stack

- Streamlit
- Pandas
- Plotly
- CSV real de tickets

## Run

```powershell
cd 04-dashboard-streamlit
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- La app busca automaticamente un archivo `*Tickets*.csv` dentro de la carpeta [datasets](../datasets).
- El dashboard se organiza por:
  - `Resumen Ejecutivo`
  - `Operacion y SLA`
  - `Agentes`
  - `Detalle`
- Para personalizar branding, coloca tus archivos en [branding](./branding):
  - `logo.png`
  - `fondo-app.jpg`
  - `fondo-sidebar.jpg`
  - `fondo-hero.jpg`
- To customize branding, place your assets in [branding](./branding):
  - `logo.png`
  - `fondo-app.jpg`
  - `fondo-sidebar.jpg`
  - `fondo-hero.jpg`
