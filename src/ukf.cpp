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
    std_a_ = 1;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 0.5;

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
    n_x_ = 5;
    n_aug_ = 7;
    lambda_ = 3 - n_x_;
    Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
    weights_ = VectorXd(2 * n_aug_ + 1);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    if (!is_initialized_) {
        x_ << 1, 1, 1, 1, 1;
        P_.fill(0.0);
        P_(0, 0) = 1;
        P_(1, 1) = 1;
        P_(2, 2) = 1;
        P_(3, 3) = 1;
        P_(4, 4) = 1;

        if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
            const double &rho = meas_package.raw_measurements_[0];
            const double &phi = meas_package.raw_measurements_[1];
            const double &rho_dot = meas_package.raw_measurements_[2];
            x_(0) = rho * cos(phi);
            x_(1) = rho * sin(phi);
            x_(2) = rho_dot * cos(phi);
            x_(3) = rho_dot * sin(phi);
        } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
            x_(0) = meas_package.raw_measurements_(0);
            x_(1) = meas_package.raw_measurements_(1);
        }
        time_us_ = meas_package.timestamp_;
        is_initialized_ = true;
        return;
    }
    double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
    time_us_ = meas_package.timestamp_;
    Prediction(delta_t);
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
        UpdateLidar(meas_package);
    } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        UpdateRadar(meas_package);
    }
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
//    AugmentedSigmaPoints
    VectorXd augmented_mean_vector = VectorXd(n_aug_);

    //create augmented mean state
    augmented_mean_vector.head(5) = x_;
    augmented_mean_vector(5) = 0;
    augmented_mean_vector(6) = 0;

    MatrixXd augmented_state_covariance = MatrixXd(n_aug_, n_aug_);

    //create augmented covariance matrix
    augmented_state_covariance.fill(0.0);
    augmented_state_covariance.topLeftCorner(5, 5) = P_;
    augmented_state_covariance(5, 5) = std_a_ * std_a_;
    augmented_state_covariance(6, 6) = std_yawdd_ * std_yawdd_;

    MatrixXd square_root_matrix = augmented_state_covariance.llt().matrixL();

    MatrixXd augmented_sigma_points = MatrixXd(n_aug_, 2 * n_aug_ + 1);

    //create augmented sigma points
    augmented_sigma_points.col(0) = augmented_mean_vector;
    for (int i = 0; i < n_aug_; i++) {
        augmented_sigma_points.col(i + 1) = augmented_mean_vector + sqrt(lambda_ + n_aug_) * square_root_matrix.col(i);
        augmented_sigma_points.col(i + 1 + n_aug_) =
                augmented_mean_vector - sqrt(lambda_ + n_aug_) * square_root_matrix.col(i);
    }
//    SigmaPointPrediction
    //predict sigma points
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        //extract values for better readability
        double p_x = augmented_sigma_points(0, i);
        double p_y = augmented_sigma_points(1, i);
        double v = augmented_sigma_points(2, i);
        double yaw = augmented_sigma_points(3, i);
        double yawd = augmented_sigma_points(4, i);
        double nu_a = augmented_sigma_points(5, i);
        double nu_yawdd = augmented_sigma_points(6, i);

        //predicted state values
        double px_p, py_p;

        //avoid division by zero
        if (fabs(yawd) > 0.001) {
            px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
            py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
        } else {
            px_p = p_x + v * delta_t * cos(yaw);
            py_p = p_y + v * delta_t * sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yawd * delta_t;
        double yawd_p = yawd;

        //add noise
        px_p += 0.5 * nu_a * delta_t * delta_t * cos(yaw);
        py_p += 0.5 * nu_a * delta_t * delta_t * sin(yaw);
        v_p += nu_a * delta_t;
        yaw_p += 0.5 * nu_yawdd * delta_t * delta_t;
        yawd_p += nu_yawdd * delta_t;

        //write predicted sigma point into right column
        Xsig_pred_(0, i) = px_p;
        Xsig_pred_(1, i) = py_p;
        Xsig_pred_(2, i) = v_p;
        Xsig_pred_(3, i) = yaw_p;
        Xsig_pred_(4, i) = yawd_p;
    }

//    PredictMeanAndCovariance
    // set weights
    weights_(0) = lambda_ / (lambda_ + n_aug_);
    for (int i = 1; i < 2 * n_aug_ + 1; i++) {  //2n+1 weights
        weights_(i) = 0.5 / (n_aug_ + lambda_);
    }

    //predicted state mean
    x_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
        x_ += weights_(i) * Xsig_pred_.col(i);
    }

    //predicted state covariance matrix
    P_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
        VectorXd state_difference = Xsig_pred_.col(i) - x_;
        //angle normalization
        while (state_difference(3) > M_PI) state_difference(3) -= 2. * M_PI;
        while (state_difference(3) < -M_PI) state_difference(3) += 2. * M_PI;

        P_ += weights_(i) * state_difference * state_difference.transpose();
    }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
    /**
    Use lidar data to update the belief about the object's position. Modify the state vector, x_, and covariance, P_.
    */
    MatrixXd measurement_matrix_h = MatrixXd(2, 5);
    measurement_matrix_h << 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0;

    VectorXd measurement_residual_y = meas_package.raw_measurements_ - measurement_matrix_h * x_;

    MatrixXd measurement_covariance_r = MatrixXd(2, 2);
    measurement_covariance_r << 0.0225, 0,
            0, 0.0225;

    MatrixXd h_transpose = measurement_matrix_h.transpose();
    MatrixXd residual_covariance_s = measurement_matrix_h * P_ * h_transpose + measurement_covariance_r;
    MatrixXd kalman_gain_k = P_ * h_transpose * residual_covariance_s.inverse();

    x_ += kalman_gain_k * measurement_residual_y;
    MatrixXd identity = MatrixXd::Identity(x_.size(), x_.size());
    P_ = (identity - kalman_gain_k * measurement_matrix_h) * P_;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    /**
    Use radar data to update the belief about the object's position. Modify the state vector, x_, and covariance, P_.
    */

//    PredictRadarMeasurement
    //set measurement dimension, radar can measure r, phi, and r_dot
    int measurement_dimension = 3;

    MatrixXd measurement_sigma_points = MatrixXd(measurement_dimension, 2 * n_aug_ + 1);

    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

        // extract values for better readibility
        double p_x = Xsig_pred_(0, i);
        double p_y = Xsig_pred_(1, i);
        double v = Xsig_pred_(2, i);
        double yaw = Xsig_pred_(3, i);

        double v1 = cos(yaw) * v;
        double v2 = sin(yaw) * v;

        // measurement model
        measurement_sigma_points(0, i) = sqrt(p_x * p_x + p_y * p_y);                        //r
        measurement_sigma_points(1, i) = atan2(p_y, p_x);                                 //phi
        measurement_sigma_points(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y);   //r_dot
    }

    VectorXd predicted_measurement = VectorXd(measurement_dimension);
    predicted_measurement.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        predicted_measurement += weights_(i) * measurement_sigma_points.col(i);
    }

    MatrixXd measurement_covariance_s = MatrixXd(measurement_dimension, measurement_dimension);
    measurement_covariance_s.fill(0.0);
    VectorXd residual_difference;
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        residual_difference = measurement_sigma_points.col(i) - predicted_measurement;

        //angle normalization
        while (residual_difference(1) > M_PI) residual_difference(1) -= 2. * M_PI;
        while (residual_difference(1) < -M_PI) residual_difference(1) += 2. * M_PI;

        measurement_covariance_s += weights_(i) * residual_difference * residual_difference.transpose();
    }

    MatrixXd measurement_noise_covariance_r = MatrixXd(measurement_dimension, measurement_dimension);
    measurement_noise_covariance_r << std_radr_ * std_radr_, 0, 0,
            0, std_radphi_ * std_radphi_, 0,
            0, 0, std_radrd_ * std_radrd_;
    measurement_covariance_s += measurement_noise_covariance_r;

//    UpdateState
    MatrixXd cross_correlation = MatrixXd(n_x_, measurement_dimension);

    //calculate cross correlation matrix
    cross_correlation.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        residual_difference = measurement_sigma_points.col(i) - predicted_measurement;
        //angle normalization
        while (residual_difference(1) > M_PI) residual_difference(1) -= 2. * M_PI;
        while (residual_difference(1) < -M_PI) residual_difference(1) += 2. * M_PI;

        VectorXd state_difference = Xsig_pred_.col(i) - x_;
        //angle normalization
        while (state_difference(3) > M_PI) state_difference(3) -= 2. * M_PI;
        while (state_difference(3) < -M_PI) state_difference(3) += 2. * M_PI;

        cross_correlation += weights_(i) * state_difference * residual_difference.transpose();
    }

    MatrixXd kalman_gain_k = cross_correlation * measurement_covariance_s.inverse();

    residual_difference = meas_package.raw_measurements_ - predicted_measurement;

    //angle normalization
    while (residual_difference(1) > M_PI) residual_difference(1) -= 2. * M_PI;
    while (residual_difference(1) < -M_PI) residual_difference(1) += 2. * M_PI;

    //update state mean and covariance matrix
    x_ += kalman_gain_k * residual_difference;
    P_ -= kalman_gain_k * measurement_covariance_s * kalman_gain_k.transpose();
}
