## Per-sequence results

| seq | frames | path [m] | loops | odom drift t[%] / r[deg/100m] | ATE odom [m] | ATE +PGO [m] | end err odom [m] | end err +PGO [m] |
|---|---|---|---|---|---|---|---|---|
| 00 | 933 | 5065 | 21 | 1.51 / 0.92 | 26.24 | 9.14 | 73.1 | 44.6 |
| 01 | 221 | 1232 | 10 | 0.46 / 0.30 | 0.69 | 0.51 | 5.7 | 3.5 |
| 02 | 221 | 694 | 2 | 0.74 / 0.86 | 1.33 | 0.66 | 5.6 | 0.1 |
| 03 | 815 | 3221 | 3 | 1.67 / 0.78 | 7.64 | 7.32 | 26.5 | 29.8 |

## Ablation

| variant (seq 00, odometry only) | drift t [%] | drift r [deg/100m] | ATE [m] | end err [m] |
|---|---|---|---|---|
| full system (scan-to-map, point-to-plane, Tukey, adaptive thr.) | 1.50 | 0.91 | 25.51 | 71.2 |
| point-to-point instead of point-to-plane | 4.70 | 2.11 | 93.59 | 299.5 |
| scan-to-scan instead of scan-to-local-map | 2.06 | 1.23 | 42.05 | 117.3 |
| no robust kernel | 2.20 | 1.33 | 43.73 | 115.6 |
| fixed 3 m threshold instead of adaptive | 1.92 | 1.15 | 30.53 | 84.2 |
