#pragma once
#define NOMINMAX
#include <algorithm>

class PIDController {
public:
    PIDController(double kp=0.2, double ki = 0.02, double kd=0.01, double windupGuard = 20.0) : Kp(kp), Ki(ki), Kd(kd), windup_guard(windupGuard) {}

    void reset() {
        setpoint = 0.0;
        p_term = 0.0;
        i_term = 0.0;
        d_term = 0.0;
        last_error = 0.0;
        output = 0.0;
        initialized = false;
    }

    void setSetpoint(double sp) {setpoint = sp; }
    
    // update using error directly 
    double updateError(double error, double dt) {
        if (dt <= 0.0) return output;

        if (!initialized) {
            last_error = error;
            initialized = true;
        }
        
        p_term = error;
        
        i_term += error * dt;
        i_term = std::max<double>(-windup_guard, std::min<double>(i_term, windup_guard));

        d_term = (error - last_error) / dt;
        last_error = error;

        output = (Kp * p_term) + (Ki * i_term) + (Kd * d_term);
        return output;
    }

    double getOutput() const {return output; }

    // Public setters for gains for tuning 
    void setGains(double kp, double ki, double kd) { Kp = kp; Ki = ki; Kd = kd; }
    void setWindupGuard(double wg) {windup_guard = wg; }

private:
    double Kp{0}, Ki{0}, Kd{0};
    double windup_guard{20.0};

    double setpoint{0.0};
    double p_term{0.0}, i_term{0.0}, d_term{0.0};
    double last_error{0.0};
    double output{0.0};
    bool initialized{false};
};