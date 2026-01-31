#pragma once

#define NOMINMAX
#include <Aria.h>
#include <Eigen/Dense>
#include <random>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

#include "PIDController.h"
#include "KalmanFilter.h"
#include "CSVLogger.h"

// Helpers & Data types 
inline double wrapDeg180(double angle) 
{
    if (angle > 0.0)
    {
        if (angle > 180.0)
            return angle - 2.0 * 180.0;
    }
    else
    {
        if (angle < -180.0)
            return angle + 2.0 * 180.0;
    }
    return angle;
}

struct Waypoint {
    double x_mm{0};
    double y_mm{0};
};
// Blackboard shared betwen WaypointFollow (producer) and SpeedControl (consumer)
struct FollowState
{
    bool active{false};

    // Produced by WaypointFollow, consumed by SpeedControl
    double lateralError_m{0.0}; // signed cross-track error (m)
    double headingError_deg{0.0}; // heading error (deg)

    int segment{0};
    double distanceToNext_m{0.0};
};


// State Estimator Action
class EstimateRobotStateAction : public ArAction {
public:
    // constructor
    EstimateRobotStateAction(double Q_std, double R_pos, double R_vel, bool simulateNoise = true)
    : ArAction("estimateRobotState"), 
    Q_std(Q_std), R_pos(R_pos), R_vel(R_vel), 
    simulateNoise(simulateNoise), 
    rng(std::random_device{}()),
    npos(0.0, R_pos),
    nvel(0.0, R_vel)
    {
        myDesired.reset();
        kf.setR(R_pos, R_vel);

        KalmanFilter::VecX x0; x0.setZero();
        KalmanFilter::MatX P0 = KalmanFilter::MatX::Identity() * 0.25;
        kf.setInitial(x0, P0);

        startMs = ArUtil::getTime();
        lastMs = startMs;
    }
    
    // Destructor, doesn't need to do anything
    virtual ~EstimateRobotStateAction() {}

    // called by action resolver to obtain actions requested behavior
    // This action does not command robot
    virtual ArActionDesired *fire(ArActionDesired currentDesired) 
    {
        myDesired.reset();

        if(myRobot == NULL) return NULL;

        const long nowMs = ArUtil::getTime();
        double dt = (nowMs - lastMs) / 1000.0;
        if (dt <= 0.0) dt = 1e-3;
        lastMs = nowMs;

        lastDt = dt;
        simTime = (nowMs - startMs) / 1000.0;

        kf.updateModel(dt, Q_std);
        kf.predict();
        ArPose pose = myRobot->getPose();
        const double headingRad = pose.getTh() * M_PI / 180.0;

        const double v_mm_s = myRobot->getVel();
        const double vx_m_s = (v_mm_s * std::cos(headingRad)) / 1000.0;
        const double vy_m_s = (v_mm_s * std::sin(headingRad)) / 1000.0;

        const double x_m = pose.getX()/1000.0;
        const double y_m = pose.getY()/1000.0;

        double mx = x_m, my = y_m, mvx = vx_m_s, mvy = vy_m_s;
        if (simulateNoise) {
            mx += npos(rng); my += npos(rng);
            mvx += nvel(rng); mvy += nvel(rng);
        }

        KalmanFilter::VecZ z;
        z << mx, mvx, my, mvy;
        kf.update(z);

        lastPose = pose;
        lastMeas = z;

        return NULL;   
    }

    // Accessors
    const KalmanFilter& filter() const { return kf; }
    const ArPose& pose() const { return lastPose; }
    const KalmanFilter::VecZ meas() const { return lastMeas; }
    double time() const { return simTime; }
    double dt() const { return lastDt; }

    double getQ_std() const { return Q_std; }
    double getR_pos() const { return R_pos; }
    double getR_vel() const { return R_vel; }
private: 
    ArActionDesired myDesired;

    KalmanFilter kf;
    double Q_std{0.0}, R_pos{0.0}, R_vel{0.0};
    bool simulateNoise;
    long startMs{0}, lastMs{0};
    double simTime{0.0};
    double lastDt{0.05};
    ArPose lastPose;
    KalmanFilter::VecZ lastMeas;
    std::mt19937 rng;
    std::normal_distribution<double> npos;
    std::normal_distribution<double> nvel;
};

// Waypoint Follow Action
class ActionWaypointFollow : public ArAction
{
public:
    ActionWaypointFollow(const std::vector<Waypoint>& wps,
                         FollowState& st,
                         CSVLogger& logger,
                         EstimateRobotStateAction& estimator)   
    : ArAction("WaypointFollow"),
    waypoints(wps),
    state(st),
    logger(logger),
    estimator(estimator)
    {
        myDesired.reset();
        totalSegments = (waypoints.size() >= 2) ? (int)waypoints.size() -1 : 0;
        
        // PID gains
        lateralPID.setGains(7.2, 0.0, 0.72);
        lateralPID.setWindupGuard(5.0);
        headingPID.setGains(0.9, 0.0, 0.09);
        headingPID.setWindupGuard(10.0);

        if(totalSegments > 0) updateLineSegment();
    } 
    double wpThreshold_m = 0.5;
    double maxRotVel = 100.0; 
    double u = 0.0;
    
    virtual ~ActionWaypointFollow() {}

    virtual void setRobot(ArRobot *robot) override
    {
        ArAction::setRobot(robot);
        if(robot == NULL)
        {
            ArLog::log(ArLog::Terse, "WaypointFollow: no robot, deactivating.");
            deactivate();
        }
    }

    virtual ArActionDesired *fire(ArActionDesired currentDesired) 
    {
        myDesired.reset();

        if(myRobot == NULL)
        {
            deactivate();
            return NULL;
        }
        state.active = false;
        if(totalSegments <= 0) return NULL;

        if(currentSegment >= totalSegments)
        {
            state.active = false;
            myDesired.reset();
            myDesired.setVel(0);
            myDesired.setRotVel(0);
            logRow();
            myRobot->stopRunning();
            deactivate();
            return &myDesired;
        }

        state.active = true;

        // KF state: [x, xdot, ax, y, ydot, ay]
        const auto xhat = estimator.filter().state();
        const double x_m = xhat(0);
        const double y_m = xhat(3);
        const double vx = xhat(1);
        const double vy = xhat(4);

        computeDistanceToLine(x_m, y_m);
        computeHeadingError(vx, vy);
        computeDistanceToNext(x_m, y_m);

        // Publish to blackboard for speed control
        state.lateralError_m = lateral_error_m;
        state.headingError_deg = heading_error_deg;
        state.segment = currentSegment;
        state.distanceToNext_m = distance_to_next_m;

        // Print telemetry data every 5th report
        static int reportCount = 0;
        reportCount++;
        if (reportCount % 5 == 0)
        {
            // Print telemetry data to the log or viewer
            ArLog::log(ArLog::Normal, "Segment: %d, Lateral Error: %.2f m, Heading Error: %.2f°, Distance to Next: %.2f m",
                    currentSegment, lateral_error_m, heading_error_deg, distance_to_next_m);
        }

        // Advance waypoint if close enough
        if(distance_to_next_m < wpThreshold_m)
        {
            currentSegment++;
            if(currentSegment < totalSegments) updateLineSegment();
        }

        // PID update using estimator dt
        double dt = estimator.dt();
        if(dt <= 0.0 || dt >1.0) dt = 0.05;
        
        const double u_lat = lateralPID.updateError(lateral_error_m, dt);
        const double u_head = headingPID.updateError(heading_error_deg, dt);

        // control law: u = K_d * e_d + K_psi * e_psi
        u = u_lat + u_head;
        u = std::max<double>(-maxRotVel, std::min<double>(u, maxRotVel));

        myDesired.setRotVel(u);

        logRow();
        return &myDesired;
    }

private:
    void updateLineSegment()
    {
        // Line through wp1->wp2: ax + by + c = 0
        const Waypoint& wp1 = waypoints[currentSegment];
        const Waypoint& wp2 = waypoints[currentSegment + 1];

        const double x1 = wp1.x_mm / 1000.0;
        const double y1 = wp1.y_mm / 1000.0;
        const double x2 = wp2.x_mm / 1000.0;
        const double y2 = wp2.y_mm / 1000.0;

        double dx = (x2 - x1);
        double dy = (y2 - y1);
        a = dy;
        b = -dx;
        c = - (a * x1 + b * y1 );
    }

    void computeDistanceToLine(double x_m, double y_m)
    {
        const double num = a * x_m + b * y_m + c;
        const double den = std::sqrt(a * a + b * b);
        lateral_error_m = (den > 1e-9) ? (num / den) : 0.0;
        lateral_error_m = std::max<double>(-3.0, std::min<double>(lateral_error_m, 3.0));
    }

    void computeHeadingError(double vx, double vy)
    {
        // Desired direction along the line:
        const double heading_ref = std::atan2(a, -b); // radians
        double heading_cur = std::atan2(vy, vx);
        heading_error_deg = wrapDeg180((heading_ref - heading_cur) * 180.0 / M_PI );
    }

    void computeDistanceToNext(double x_m, double y_m)
    {
        const Waypoint& next = waypoints[currentSegment + 1];
        const double dx = (next.x_mm / 1000.0) - x_m;
        const double dy = (next.y_mm / 1000.0) - y_m;
        distance_to_next_m = std::sqrt(dx * dx + dy * dy);
    }

    void logRow()
    {
        const auto& kf = estimator.filter();
        const auto xhat = kf.state();
        const auto& P = kf.cov();
        const auto& K = kf.gain();
        const ArPose pose = estimator.pose();
        const double sim_t = estimator.time();

        // true kinematics from simulation
        const double th_deg = pose.getTh();
        const double th_rad = th_deg * M_PI/180.0;
        const double v_mm_s = getRobot()->getVel();
        const double vx_true = (v_mm_s * std::cos(th_rad)) / 1000.0;
        const double vy_true = (v_mm_s * std::sin(th_rad)) / 1000.0;
        const double rot_true = getRobot()->getRotVel();

        // Estiamted heading from velocity
        const double heading_est_deg = std::atan2(xhat(4), xhat(1)) * 180.0/M_PI;
        
        const double P_trace = P.trace();
        const auto z = estimator.meas();

        // Kalman Gains
        const double K_x = K(0,0); 
        const double K_xdot = K(1,1);
        const double K_y = K(3,2);
        const double K_ydot = K(4,3);

        // Save data
        std::vector<double> row = {
            sim_t,
            pose.getX()/1000.0, pose.getY()/1000.0, th_deg,
            vx_true, vy_true, rot_true,

            xhat(0), xhat(3), heading_est_deg,
            xhat(1), xhat(4),
            xhat(2), xhat(5),

            z(0), z(2), z(1), z(3),

            lateral_error_m, heading_error_deg,
            static_cast<double>(currentSegment),
            distance_to_next_m,

            lateralPID.getOutput(),
            headingPID.getOutput(),
            u,

            P_trace, K_x, K_xdot, K_y, K_ydot,

            estimator.getQ_std(), estimator.getR_pos(), estimator.getR_vel(),

            P(0,0), P(0,3), P(3,0), P(3,3)
        };
        
        logger.writeRow(row);
    }
private:
    ArActionDesired myDesired;

    const std::vector<Waypoint>& waypoints;
    FollowState& state;
    CSVLogger& logger;
    EstimateRobotStateAction& estimator;

    int currentSegment{0};
    int totalSegments{0};

    double a{0.0};
    double b{0.0};
    double c{0.0};

    double lateral_error_m{0.0};
    double heading_error_deg{0.0};
    double distance_to_next_m{0.0};

    PIDController lateralPID;
    PIDController headingPID;
};

// Speed Control Action
class ActionSpeedControl : public ArAction {
public:
    ActionSpeedControl( const FollowState& st, double maxSpeed_mm_s, double minSpeed_mm_s)
    : ArAction("SpeedControl"),
      state(st),
      maxSpeed(maxSpeed_mm_s), 
      minSpeed(minSpeed_mm_s)
    {
        myDesired.reset();
    }
    
    virtual ~ActionSpeedControl() {}
    
    virtual ArActionDesired *fire(ArActionDesired currentDesired) 
    {
        myDesired.reset();
        if(myRobot == NULL) return NULL;
        if(!state.active) return NULL;

        const double lateral_error = std::abs(state.lateralError_m);
        const double heading_error = std::abs(state.headingError_deg);

        // Error factors for attenuating speed when errors are large
        const double lateral_error_factor = std::max<double>(0.5, 1.0 - (lateral_error / 1.0));
        const double heading_error_factor = std::max<double>(0.6, 1.0 - (heading_error / 90.0));
        
        double cmd = maxSpeed * lateral_error_factor * heading_error_factor;
        cmd = std::max<double>(minSpeed, std::min<double>(cmd, maxSpeed));
        myDesired.setVel(cmd);
        return &myDesired;
    }
private:
    ArActionDesired myDesired;
    const FollowState& state;

    double maxSpeed{700.0};
    double minSpeed{250.0}; 
};