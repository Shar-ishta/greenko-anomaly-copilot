# Demo Notes

**Flow:** open the dashboard → pick an asset → read the flagged events → ask the copilot → review the evaluation table.

| Screenshot | What it shows |
|---|---|
| 01_dashboard_solar.png | Summary tiles (2 assets, 240 readings, 5 flagged, 2 high-severity events) and SOLAR_INV_01 power, vibration and temperature charts. Red markers and bands are anomalies. |
| 02_dashboard_wind.png | WTG_07: vibration and temperature spike with output ≈ 0.72 MW vs ≈ 1.30 MW expected. |
| 03_event_card_solar.png | Operator-language explanation: vibration 3.98 mm/s (normal ≈ 1.38), temperature +19 °C, ranked causes, next checks and a safety note. |
| 04_copilot.png | Copilot answers "Summerize high severity" with playbook-based checks and shows its reasoning steps. |
| 05_evaluation.png | Synthetic fault-injection results: ensemble F1 0.99 (obvious) to 0.85 (subtle). |


**Key message:** the two fault windows in the bundled data are detected with no false alarms. Subtle faults are the main weakness, as documented in the README.