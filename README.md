# GNCProject

Originally developed in Python for a graduate survey course on guidance, navigation, and control,
this waypoint-following project uses a Kalman filter for position/velocity estimation, integrates
with MobileSim for robot simulation, and uses ARIA for robot hardware integration. It was first used
on a Pioneer3 robot, then translated to C++ to practice building a full C++ project. The program
follows a sequence of waypoints using a control law driven by a Kalman state estimate. It logs
telemetry to CSV, and a Python script generates plots and error distributions.

## Prerequisites
Install these vendor tools before building or running contained within 'RequiredSoftware\':
- ARIA (Adept MobileRobots ARIA SDK):https://github.com/srmq/ARIA/tree/master 
- MobileSim (robot simulator): https://github.com/srmq/MobileSim?tab=readme-ov-file 
- Mapper3 / Mapper3D (map authoring tools): http://vigir.missouri.edu/~gdesouza/Research/MobileRobotics/Software/Mapper3/ 

## Repository Structure
- `src/` C++ entrypoint and main loop
- `include/` headers (Kalman filter, Robot Actions, helpers)
- `data/` waypoint list and data logging files
- `scripts/` Python plotting and analysis
- `third_party/` Eigen (used for developing the Kalman filter helper)
- `build/` local build artifacts (not required)

## Dependencies
### C++ / Build
- CMake >= 3.20
- ARIA SDK (set `-DARIA_ROOT`)
- Eigen (vendored in `third_party/eigen-5.0.1`)

### Python
- Python 3.x
- `matplotlib`, `numpy`, `scipy`, `sympy`, `filterpy`

Install Python deps:
```cmd
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Build (Windows / MSVC)
```cmd
cd path\to\project\build
cmake .. -G "Visual Studio <version>" -A Win32 -DARIA_ROOT="C:\Program Files\MobileRobots\Aria"
cmake --build . --config Release
```


## Run
Run the C++ program from `..\build`:
```cmd
.\Release\follow_wp_kfposvel.exe ..\data\waypoint_list.txt ..\data\log.csv
```

## Plot / Analyze
Activate your virtual environment and run plotting from the repository root. Plots are saved to the
current working directory.
```cmd
.\.venv\Scripts\activate
python scripts\plot_wp_kfposvel.py data\waypoint_list.txt data\log.csv
```

## Quick Reference
- Build (Windows/MSVC): `cmake .. -G "Visual Studio <version>" -A Win32 -DARIA_ROOT="C:\Program Files\MobileRobots\Aria"`
- Run: `.\Release\follow_wp_kfposvel.exe ..\data\waypoint_list.txt ..\data\log.csv`
- Plot: `python scripts\plot_wp_kfposvel.py data\waypoint_list.txt data\log.csv`


## Data Inputs and Outputs
- Input: `data/waypoint_list.txt`
- Output: `data/log.csv` (pose, estimate, control, error terms)

## Kalman Filter (FilterPy -> C++ Adaptation)
This implementation is adapted from the FilterPy Python library into a C++/Eigen
implementation. Key mapping:
- State vector, `F`, `H`, `Q`, `R` follow the same structure as FilterPy
- Predict/update are implemented with Eigen matrix math
- Tuning and numerical stability adjustments are documented in code comments

If you modify the model, keep the state definition consistent between the C++
filter and the Python analysis scripts.

## Control Laws
The controller is split into a waypoint-following yaw controller and a speed
controller, both driven by the Kalman filter estimate.

- **Cross-track geometry and reference frame**: the local path frame uses a
  Parallel Transport (P-T) frame [1]. For a segment from `(x1, y1)`
  to `(x2, y2)`, define the tangent `t = [dx, dy] = [x2 - x1, y2 - y1]` and a
  90-degree clockwise normal `n_cw = [dy, -dx]`. Using the normal as the line
  coefficients, in [2], gives `a = n_x`, `b = n_y`, `c = -(a x1 + b y1)`, so the
  cross-track error is the signed distance to `ax + by + c = 0`, saturated to
  +/-1 m.
- **Heading error**: desired heading is the line direction
  `heading_ref = atan2(a, -b)`. Current heading is estimated from velocity,
  `heading_cur = atan2(vy, vx)`. The wrapped heading error is
  `e_psi = wrap(heading_ref - heading_cur)`.
- **Yaw control (rotational velocity)**: two PID loops act on lateral and
  heading error, summed as
  `u = u_lat + u_head`, then saturated to ±`maxRotVel`.
- **Waypoint switching**: when the distance to the next waypoint is below
  `wpThreshold_m`, the next segment becomes active.
- **Speed control**: translational speed is reduced as lateral/heading errors
  grow using multiplicative attenuation factors, then clamped to
  `[minSpeed, maxSpeed]`.

Key gains live in `include/Actions.h` and can be tuned via the PID controllers.

## Results / Plots
Example outputs:
- `kfposvel_analysis.png`
- `kfposvel_error_distributions.png`

## Troubleshooting
- ARIA not found: verify `-DARIA_ROOT` points to the ARIA install directory
- DLL issues: ensure `Aria.dll` is available in `ARIA_ROOT\BIN`
- Python import errors: activate `.venv` and install `requirements.txt`

## What I'd Do Differently Next Time
- Separate interfaces for hardware vs simulation implementation
- Write a more agnostic Kalman filter class that takes vector/matrix dimensions instead of hardcoding
  them
- Use an EKF to implement basic SLAM and obstacle avoidance logic
- Provide a parameter file for tuning rather than hardcoding PD gains

## References
[1] Nguyen T. Hung, Francisco Rego, Joao Quintas, Joao Cruz, Marcelo Jacinto,
  David Souto, Andre Potes, Luis Sebastiao, Antonio M. Pascoal. "A review of
  path following control strategies for autonomous robotic vehicles: theory,
  simulations, and experiments." arXiv:2204.07319 (2022).
  https://arxiv.org/abs/2204.07319 (DOI: https://doi.org/10.48550/arXiv.2204.07319)

[2] “Dot Product - Distance between Point and a Line | Brilliant Math & Science Wiki,” Brilliant.org, 2024. https://brilliant.org/wiki/dot-product-distance-between-point-and-a-line/

## License
MIT License

Copyright (c) 2026 Teddy Herrera

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
