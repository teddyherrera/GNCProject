#pragma once
#include <Eigen/Dense>

class KalmanFilter {
public:
    static constexpr int NX = 6;
    static constexpr int NZ = 4;
    
    using VecX = Eigen::Matrix<double, NX, 1>;
    using VecZ = Eigen::Matrix<double, NZ, 1>;
    using MatX = Eigen::Matrix<double, NX, NX>;
    using MatZ = Eigen::Matrix<double, NZ,NZ>;
    using MatHX = Eigen::Matrix<double, NZ, NX>;
    using MatK = Eigen::Matrix<double, NX, NZ>;

    KalmanFilter() {
        x.setZero();
        P.setIdentity();
        Q.setIdentity();
        R.setIdentity();
        H.setZero();

        // z = [x, xdot, y, ydot]
        H(0,0) = 1.0;
        H(1,1) = 1.0;
        H(2,3) = 1.0;
        H(3,4) = 1.0;

        F.setIdentity();
    }
    void setInitial(const VecX& x0, const MatX& P0) {x = x0; P = P0; }
    void setR(double R_pos_std, double R_vel_std) {
        R.setZero();
        R(0,0) = R_pos_std * R_pos_std;
        R(1,1) = R_vel_std * R_vel_std;
        R(2,2) = R_pos_std * R_pos_std;
        R(3,3) = R_vel_std * R_vel_std;
    }

    // Constant-accel discrete time model
    static Eigen::Matrix3d F_axis(double dt) {
        Eigen::Matrix3d Fa;
        Fa << 1, dt, 0.5*dt*dt,
              0, 1,         dt,
              0, 0,          1;
        return Fa;
    }

    static Eigen::Matrix3d Q_axis(double dt, double accel_var) {
        // White noise on acceleration input 
        // Q = acel_var * [dt^4/4 dt^3/2 dt^2/2; dt^3/2 dt^2 dt; dt^2/2 dt 1]
        const double dt2 = dt*dt;
        const double dt3 = dt2*dt;
        const double dt4 = dt2*dt2;
        
        Eigen::Matrix3d Qa;
        Qa << dt4/4.0, dt3/2.0, dt2/2.0,
              dt3/2.0, dt2,      dt,
              dt2/2.0, dt,       1.0;
        return accel_var * Qa;
    }
    void updateModel(double dt, double Q_std) {
        // Build block-diagonal F and Q
        F.setZero();
        const Eigen::Matrix3d Fa = F_axis(dt);
        F.block<3,3>(0,0) = Fa;
        F.block<3,3>(3,3) = Fa;

        Q.setZero();
        const double accel_var = Q_std * Q_std;
        const Eigen::Matrix3d Qa = Q_axis(dt, accel_var);
        Q.block<3,3>(0,0) = Qa;
        Q.block<3,3>(3,3) = Qa;
    }
    
    void predict() {
        x = F * x;
        P = F * P * F.transpose() + Q;
    }

    void update(const VecZ& z) {
        // Innovation step
    const VecZ y = z - (H * x);
    const MatZ S = H * P * H.transpose() + R;

    const MatK K = P * H.transpose() * S.ldlt().solve(MatZ::Identity());

    x = x + K * y;

    const MatX I = MatX::Identity();
    const MatX KH = K * H;
    P = (I - KH) * P * (I - KH).transpose() + K * R * K.transpose();

    lastK = K;
    }

    const VecX& state() const { return x; }
    const MatX& cov() const {return P; }
    const MatK& gain() const {return lastK; }

private:
    VecX x; 
    MatX P;
    MatX F;
    MatX Q;
    MatHX H;
    MatZ R; 
    MatK lastK;
};