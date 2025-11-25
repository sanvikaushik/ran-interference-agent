 ran-interference-agent

A lab for simulating RAN interference scenarios, diagnosing them with a rule/DTW-based agent, and visualizing mitigations.

## Quickstart
1. Install dependencies (use a virtualenv if you prefer):
   ```bash
   pip install -r requirements.txt
   ```
2. Run the console demo to see one sample sequence and metric summary:
   ```bash
   python main.py
   ```
3. Launch the Streamlit dashboard for interactive exploration:
   ```bash
   streamlit run ui/dashboard.py
   ```
4. (Optional) Recompute evaluation numbers for both the rule engine and ML classifier:
   ```bash
   python run_metrics.py
   ```
5. (Optional) Run the unit tests:
   ```bash
   pytest
   ```
