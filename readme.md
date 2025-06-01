# 📊 ScenViz – Scenario Visualization Dashboard

**ScenViz** is a modular, themeable dashboard built with Streamlit and Plotly for visualizing scenario-based data such as GHG emissions, land use, energy demand, and agricultural costs.

---

## 🚀 Features

- 📁 MVC architecture: clear separation of models, views, and config
- 🧭 Sidebar with scenario and year filters
- 📊 Multiple tabs for: Costs, Emissions, Land Use, Energy Demand
- 📦 Excel-based data import (by 5-year intervals)
- 🖼️ Logo branding in sidebar
- 🎨 Custom theming and styling with centralized config
- 🧪 Git-ready structure with versioning and changelog

---

## 📁 Project Structure

ScenViz_Template/
├── app.py # Main controller
├── config/
│ ├── theme.py # Styling and layout settings
│ └── version.py # Version constant
├── models/
│ └── loaders.py # Excel file loading functions
├── views/
│ └── charts.py # Reusable chart builders
├── static/
│ ├── logo.png # Sidebar logo image
│ └── style.css # Custom CSS
├── data/
│ └── (your Excel files here)
├── requirements.txt # Dependencies
├── .gitignore
└── CHANGELOG.md

## 🧪 Version Info

**Current version:** `1.0.0`  
See [CHANGELOG.md](CHANGELOG.md) for release history.

## 🙌 Credits

Built for **SDSN GCH** to support comparative scenario modeling across climate, agriculture, land use, and energy domains.  
Developed with Streamlit, Plotly, and Python.
