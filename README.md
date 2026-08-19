# CBA Import Candidate Ranking

A ranking tool for generating a candidate list of CBA foreign players, with a locally run interactive Streamlit dashboard.

Its purpose is scouting decision support: narrowing a large overseas player pool to a shortlist for further review. It does not predict that a player will be signed; final decisions remain with human scouts and club staff.

## Run locally

From this directory:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The dashboard is then available at `http://localhost:8501`.

## Data and scoring

The CSV files under `data/` provide the demonstration inputs and generated recommendation tables used by the dashboard. The main screening rule is:

```text
S_i = F_i + 0.25 U_i + 0.02 P_i
```

Here, `F_i` is the CBA import-fit signal, `U_i` is the usage proxy, and `P_i` is points per 36 minutes. The coefficients are fixed scale adjustments rather than learned percentages. Dashboard role scores provide additional scouting views; all rankings should be checked against video, availability, roster fit, and other information not contained in public season statistics.

