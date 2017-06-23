#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // initial state vector
    x_ = VectorXd(5);

    // initial covariance matrix
    P_ = MatrixXd(5, 5);

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = M_PI;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = M_PI;

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    long previous_timestamp = 0;
    if (!is_initialized_) {
        x_ << 1, 1, 1, 1, 1;
        if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
            const double &rho = meas_package.raw_measurements_[0];
            const double &phi = meas_package.raw_measurements_[1];
            const double &rho_dot = meas_package.raw_measurements_[2];
            x_(0) = rho * cos(phi);
            x_(1) = rho * sin(phi);
            x_(2) = rho_dot * cos(phi);
            x_(3) = rho_dot * sin(phi);
        } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
            x_(0) = meas_package.raw_measurements_(0);
            x_(1) = meas_package.raw_measurements_(1);
        }
        previous_timestamp = meas_package.timestamp_;
        is_initialized_ = true;
        return;
    }
    double delta_t = (meas_package.timestamp_ - previous_timestamp) / 1000000.0;
    Prediction(delta_t);
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
        UpdateLidar(meas_package);
    } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        UpdateRadar(meas_package);
    }
    previous_timestamp = meas_package.timestamp_;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
    /**
    Estimate the object's location. Modify the state
    vector, x_. Predict sigma points, the state, and the state covariance matrix.
    */
    VectorXd augmented_mean_state = VectorXd(7);
    int state_dimension = 5;
    x_ << 5.7441,
            1.3800,
            2.2049,
            0.5015,
            0.3528;
    augmented_mean_state.head(5) = x_;
    augmented_mean_state(5) = 0;
    augmented_mean_state(6) = 0;

    MatrixXd augmented_state_covariance = MatrixXd(7, 7);
    augmented_state_covariance.fill(0, 0);
    P_ << 0.0043, -0.0013, 0.0030, -0.0022, -0.0020,
            -0.0013, 0.0077, 0.0011, 0.0071, 0.0060,
            0.0030, 0.0011, 0.0054, 0.0007, 0.0008,
            -0.0022, 0.0071, 0.0007, 0.0098, 0.0100,
            -0.0020, 0.0060, 0.0008, 0.0100, 0.0123;
    augmented_state_covariance.topLeftCorner(5, 5) = P_;
    augmented_state_covariance(5, 5) = std_a_ * std_a_;
    augmented_state_covariance(6, 6) = std_yawdd_ * std_yawdd_;

    MatrixXd square_root_matrix = augmented_state_covariance.llt().matrixL();

    //create augmented sigma points
    int augmented_dimension = 7;
    MatrixXd sigma_points = MatrixXd(augmented_dimension, 2 * augmented_dimension + 1);
    sigma_points.col(0) = augmented_mean_state;
    int lambda = 3 - augmented_dimension;
    for (int i = 0; i < augmented_dimension; i++) {
        sigma_points.col(i + 1) = augmented_mean_state + sqrt(lambda + augmented_dimension) * square_root_matrix.col(i);
        sigma_points.col(i + 1 + augmented_dimension) =
                augmented_mean_state - sqrt(lambda + augmented_dimension) * square_root_matrix.col(i);
    }

    //predict sigma points
    MatrixXd predicted_sigma_points = MatrixXd(augmented_dimension, 2 * augmented_dimension + 1);
    for (int i = 0; i < 2 * augmented_dimension + 1; i++) {
        double p_x = sigma_points(0, i);
        double p_y = sigma_points(1, i);
        double v = sigma_points(2, i);
        double yaw = sigma_points(3, i);
        double yaw_dot = sigma_points(4, i);
        double nu_a = sigma_points(5, i);
        double nu_dot = sigma_points(6, i);

        double px_p, py_p;

        if (fabs(yaw_dot) > 0.001) {
            px_p = p_x + v / yaw_dot * (sin(yaw + yaw_dot * delta_t) - sin(yaw));
            py_p = p_y + v / yaw_dot * (cos(yaw) - cos(yaw + yaw_dot * delta_t));
        } else {
            px_p = p_x + v * delta_t * cos(yaw);
            py_p = p_y + v * delta_t * sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yaw_dot * delta_t;
        double yawd_p = yaw_dot;

        //add noise
        px_p += 0.5 * nu_a * delta_t * delta_t * cos(yaw);
        py_p += 0.5 * nu_a * delta_t * delta_t * sin(yaw);
        v_p += nu_a * delta_t;
        yaw_p += 0.5 * nu_dot * delta_t * delta_t;
        yawd_p += nu_dot * delta_t;


        predicted_sigma_points(0, i) = px_p;
        predicted_sigma_points(1, i) = py_p;
        predicted_sigma_points(2, i) = v_p;
        predicted_sigma_points(3, i) = yaw_p;
        predicted_sigma_points(4, i) = yawd_p;
    }

    // set weights
    VectorXd weights = VectorXd(2 * augmented_dimension + 1);
    weights(0) = lambda / (lambda + augmented_dimension);
    for (int i = 1; i < 2 * augmented_dimension + 1; i++) {  //2n+1 weights
        weights(i) = 0.5 / (augmented_dimension + lambda);
    }

    //predicted state mean
    x_.fill(0.0);
    for (int i = 0; i < 2 * augmented_dimension + 1; i++) {  //iterate over sigma points
        x_ += weights(i) * predicted_sigma_points.col(i);
    }

    //predicted state covariance matrix
    P_.fill(0.0);
    for (int i = 0; i < 2 * augmented_dimension + 1; i++) {  //iterate over sigma points
        VectorXd state_difference = predicted_sigma_points.col(i) - x_;
        //angle normalization
        while (state_difference(3) > M_PI) state_difference(3) -= 2. * M_PI;
        while (state_difference(3) < -M_PI) state_difference(3) += 2. * M_PI;
        P_ += weights(i) * state_difference * state_difference.transpose();
    }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
    /**
    TODO:

    Complete this function! Use lidar data to update the belief about the object's
    position. Modify the state vector, x_, and covariance, P_.

    You'll also need to calculate the lidar NIS.
    */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    /**
    TODO:

    Complete this function! Use radar data to update the belief about the object's
    position. Modify the state vector, x_, and covariance, P_.

    You'll also need to calculate the radar NIS.
    */
}
