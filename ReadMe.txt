Example configure and build commands
1. cd C:\Users\herre\RobotProject\kalmanfilter\build
2. cmake .. -G "Visual Studio 16 2019" -A x64 -DARIA_ROOT="C:\Program Files\MobileRobots\Aria"
3. cmake --build . --config Release

run cpp program: 
.\Release\follow_wp_kfposvel.exe ..\data\waypoint_list.txt ..\data\log.csv

python
navigate to root folder and activate python virtual environment
.\.venv\Scripts\activate
to run plotting
python scripts\plot_wp_kfposvel.py data\waypoint_list.txt data\log.csv
