# Model evaluation summary

Test set: 2006-10-25 to 31 Oct 2006, 40 sites, 24900 predictions per model (next 15-minute flow).

## Overall metrics

|            |   RMSE |    MAE |   NRMSE |   WAPE_% |    R2 |
|:-----------|-------:|-------:|--------:|---------:|------:|
| lstm       | 47.857 | 31.628 |   0.135 |    8.943 | 0.97  |
| naive_last | 52.45  | 33.778 |   0.148 |    9.551 | 0.964 |
| hist_avg   | 66.755 | 37.127 |   0.189 |   10.498 | 0.942 |

## RMSE: weekday vs weekend

| daytype   |   lstm |   naive_last |   hist_avg |
|:----------|-------:|-------------:|-----------:|
| weekday   |  49.02 |        54.18 |      62.13 |
| weekend   |  44.82 |        47.87 |      77.07 |

## Number of sites where each model has the lowest RMSE

|      |   sites |
|:-----|--------:|
| lstm |      40 |

## Findings (check against the charts before using)

- Lowest overall RMSE among the ML models: **lstm** (47.86 vehicles/15 min); 0.0% lower than lstm.
- Baseline `naive_last` RMSE = 52.45; best ML model is 8.8% better.
- Baseline `hist_avg` RMSE = 66.75; best ML model is 28.3% better.
- Hardest site for lstm: 3001 (RMSE 113.75); easiest: 4335 (RMSE 18.44).
