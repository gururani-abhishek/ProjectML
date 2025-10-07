# Project ML Dashboard

This project predicts SDLC control gaps using machine learning and visualizes results in a Streamlit dashboard.

## Approach

- **Model Choice:**
  - Tree-based model (Multi Random Forests) is used for 24 binary controls, as it captures complex rules better than linear models (which may underfit).
  - Neural networks are not used—they would be overkill for this tabular data.
- **Data:**
  - JSON dummy data: `0` = control pass, `1` = control gap (fail).
- **Tech Stack:**
  - Model: Scikit-Learn (Random Forest)
  - Dashboard: Streamlit

## How to Run

1. **Activate your virtual environment:**
   ```sh
   source app/data/.venv/bin/activate
   ```
2. **Generate the model:**

```sh
    python app/model/main.py
```

This script will train the Multi-Random Forest model, generate a `.cvv` file containing per-control predictions (saved as `per_control_predictions_current_202510.cvv`), and save the model for use in the dashboard. The dashboard will use this `.cvv` file to display results. 3. **Start the dashboard:**

```sh
streamlit run app/dashboard.py
```

That's it! The dashboard will open in your browser.
