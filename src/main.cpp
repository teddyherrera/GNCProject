#include <Aria.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include "Actions.h"

static bool loadWaypoints(const std::string& filename, std::vector<Waypoint>& wps) {
    std::fstream in(filename);
    if(!in) return false;

    std::string line;
    while(std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string xs, ys;
        if (!std::getline(ss, xs, ',')) continue;
        if (!std::getline(ss, ys, ',')) continue;

        Waypoint wp;
        wp.x_mm = std::stod(xs);
        wp.y_mm = std::stod(ys);
        wps.push_back(wp);
    }

    // close waypoint loop
    if (wps.size() >= 2) {
        if (wps.front().x_mm != wps.back().x_mm || wps.front().y_mm != wps.back().y_mm) {
            wps.push_back(wps.front());

        }
        return true;
    }
    return false;
}

int main(int argc, char** argv)
{
    Aria::init();

    ArArgumentParser parser(&argc, argv);
    parser.loadDefaultArguments();

    if (argc < 3 || parser.checkArgument("-help")) {
        ArLog::log(ArLog::Terse, "Usage: follow_wp_kfposvel <waypoints.txt> <log.csv> [Aria args]");
        Aria::shutdown();
        return 1;
    }
    if (!Aria::parseArgs()) 
    {
        Aria::logOptions();
        Aria::shutdown();
        return 1;
    }


    const std::string waypointFile = argv[1];
    const std::string logFile = argv[2];
    std::vector<Waypoint> waypoints;
    if (!loadWaypoints(waypointFile, waypoints)) {
        ArLog::log(ArLog::Terse, "Failed to load waypoints.");
        Aria::shutdown();
        return 1;
    }

    ArRobot robot;

    ArRobotConnector conn(&parser, &robot);
    if (!conn.connectRobot()) {
        ArLog::log(ArLog::Terse, "Could not connect to robot.");
        Aria::shutdown();
        return 1;
    }

    robot.lock();
    robot.enableMotors();
    robot.unlock();

    // Logger and header
    CSVLogger logger(logFile);
    logger.writeHeader(
        "SimTime,"
        "X_true,Y_true,Heading_true,Vx_true,Vy_true,RotVel_true,"
        "X_est,Y_est,Heading_est,Vx_est,Vy_est,Ax_est,Ay_est,"
        "x_meas,y_meas,vx_meas,vy_meas,"
        "LateralError,HeadingError,CurrentSegment,DistanceToNext,"
        "LateralControl,HeadingControl,u_psi,"
        "P_trace,K_x_gain,K_xdot_gain,K_y_gain,K_ydot_gain,"
        "Q_std,R_pos_std,R_vel_std,"
        "P_xx,P_xy,P_yx,P_yy"
    );

    // shared black board between waypointfollower and speed controller
    FollowState followState;

    // Kalman Filter Parameters
    const double R_pos = 0.25;
    const double R_vel = 0.05;
    const double Q_std = 0.2;

    // Actions
    EstimateRobotStateAction estimator(Q_std, R_pos, R_vel, /*simulateNoise=*/true);
    ActionSpeedControl speed(followState, /*max*/600, /*min*/250);
    ActionWaypointFollow follower(waypoints, followState, logger, estimator);

    // Prioritize Actions
    robot.addAction(&estimator, 30);
    robot.addAction(&follower, 20);
    robot.addAction(&speed, 10);

    robot.run(true);

    Aria::exit(0);
    return 0;
}
