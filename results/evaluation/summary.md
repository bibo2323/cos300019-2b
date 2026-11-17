# Model evaluation summary

Test set: 2006-10-25 to 31 Oct 2006, 40 sites, 24900 predictions per model (next 15-minute flow).

## Overall metrics

|               |   RMSE |    MAE |   NRMSE |   WAPE_% |    R2 |
|:--------------|-------:|-------:|--------:|---------:|------:|
| lstm          | 47.857 | 31.628 |   0.135 |    8.943 | 0.97  |
| gru           | 45.317 | 29.306 |   0.128 |    8.287 | 0.973 |
| random_forest | 37.983 | 23.911 |   0.107 |    6.761 | 0.981 |

## RMSE: weekday vs weekend

| daytype   |   lstm |   gru |   random_forest |
|:----------|-------:|------:|----------------:|
| weekday   |  49.02 | 46.45 |           37.29 |
| weekend   |  44.82 | 42.36 |           39.66 |

## Number of sites where each model has the lowest RMSE

|               |   sites |
|:--------------|--------:|
| lstm          |       1 |
| gru           |       0 |
| random_forest |      39 |

## Findings (check against the charts before using)

- Lowest overall RMSE among the ML models: **random_forest** (37.98 vehicles/15 min); 20.6% lower than lstm.
- Hardest site for random_forest: 3001 (RMSE 111.11); easiest: 4051 (RMSE 16.73).
