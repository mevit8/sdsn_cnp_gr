# Changelog

All notable changes to this project will be documented here using [Semantic Versioning](https://semver.org/).

---

## [1.0.0] - 2025-05-15

### Added
- Modular folder structure following MVC architecture:
  - `models/` for data loaders
  - `views/` for chart rendering
  - `config/` for themes and settings
  - `static/` for assets (logo, CSS)
- Streamlit dashboard layout with:
  - Tabbed interface for: 💰 Costs, 🌱 Emissions, 🌍 Land Use, ⚡ Energy Demand
  - Sidebar with scenario filtering and logo
- Responsive Plotly charts (stacked bar + line)
- Theming via centralized `theme.py`
- Custom styling via `static/style.css`
- Git setup with `.gitignore`
- Virtual environment support

### Changed
- Normalized scenario strings and aligned column casing across all Excel inputs
- Chart rendering refactored into reusable functions

---

## [0.1.0] - 2025-05-14

### Added
- Initial single-page Streamlit prototype