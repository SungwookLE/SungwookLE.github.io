#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;
  
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //output matrix - laser, radar with zero as Hj_
  H_laser_ << 1,0,0,0,
              0,1,0,0;

  Hj_ << 0,0,0,0,
         0,0,0,0,
         0,0,0,0;

  /**
   * @brief Initializing the FusionEKF.
   */
  ekf_.F_ = MatrixXd::Identity(4,4);

  ekf_.x_ = VectorXd::Zero(4);
  ekf_.P_ = MatrixXd::Zero(4,4);
  ekf_.P_ << 1,0,0,0,
             0,1,0,0,
             0,0,100,0,
             0,0,0,100;
  ekf_.Q_ = MatrixXd::Zero(4,4);
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  if (!is_initialized_) {
    /**
     * @brief Initialze the state ekf_.x_ with the first measurement and Create the covariance matrix.
     */

    // first measurement
    cout << "EKF initiated.." << endl;
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
       * @brief Convert radar from polar to cartesian coordinates and initialize state when receiving RADAR 
       */
      ekf_.x_(0) = measurement_pack.raw_measurements_(0) * cos(measurement_pack.raw_measurements_(1));
      ekf_.x_(1) = measurement_pack.raw_measurements_(0) * sin(measurement_pack.raw_measurements_(1));
      ekf_.x_(2) = measurement_pack.raw_measurements_(2) * cos(measurement_pack.raw_measurements_(1));
      ekf_.x_(3) = measurement_pack.raw_measurements_(2) * sin(measurement_pack.raw_measurements_(1));

    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
       * @brief Initialize state when receiving LASER
       */
      ekf_.x_(0) = measurement_pack.raw_measurements_(0);
      ekf_.x_(1) = measurement_pack.raw_measurements_(1);
      ekf_.x_(2) = 0;
      ekf_.x_(3) = 0;
    }

    // done initializing, no need to predict or update
    previous_timestamp_ = measurement_pack.timestamp_;
    is_initialized_ = true;
    return;
  }


  /**
   * @brief Prediction
   * Update the state transition matrix F according to the new elapsed time. Time is measured in seconds.
   * Update the process noise covariance matrix. variance_ax = 9 and variance_ay = 9 for your Q matrix.
   */

  double dt;
  dt = (measurement_pack.timestamp_ - previous_timestamp_)/1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  ekf_.F_(0,2) = dt;
  ekf_.F_(1,3) = dt;

  double dt_2 = dt* dt;
  double dt_3 = dt_2* dt;
  double dt_4 = dt_3* dt;

  double cov_ax =9 , cov_ay = 9;

  ekf_.Q_ << dt_4 /4 * cov_ax, 0, dt_3/2 * cov_ax, 0,
             0, dt_4 / 4 * cov_ay, 0, dt_3 /2 * cov_ay,
             dt_3/2*cov_ax, 0, dt_2 * cov_ax , 0,
             0, dt_3/2*cov_ay, 0, dt_2 * cov_ay;
  
  
  ekf_.Predict();

  /**
   * @brief Update
   * Per sensor type to perform the update step.
   * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    Hj_ = tools.CalculateJacobian(ekf_.x_); // Linearization
    ekf_.Init(ekf_.x_, ekf_.P_,ekf_.F_, Hj_, R_radar_, ekf_.Q_ );
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
    ekf_.Init(ekf_.x_, ekf_.P_, ekf_.F_, H_laser_, R_laser_, ekf_.Q_);
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
