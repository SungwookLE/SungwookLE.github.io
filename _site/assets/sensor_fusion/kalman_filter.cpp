#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/* 
 * Eigen library does not initialize 
 * VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
   * @brief: predict the state
   */

  VectorXd u = VectorXd::Zero(4);

  std::cout << "Predict: \n";
  x_ = F_ * x_ + u;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
   * @brief: update the state by using Kalman Filter equations when recieving LIDAR data
   */
  std::cout << "Lidar Update: \n";
  VectorXd y = z - H_ * x_;
  MatrixXd S = H_ * P_ * H_.transpose() + R_;
  MatrixXd K = P_ * H_.transpose() * S.inverse();

  x_ = x_ + K*y;
  P_ = P_ - K*H_*P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
   * @brief: update the state by using Extended Kalman Filter equations when receiving RADAR data
   */
  std::cout << "Radar Update: \n";

  double px = x_(0);
  double py = x_(1);
  double vx = x_(2);
  double vy = x_(3);

  double rho = sqrt(px*px + py*py);
  double phi = atan2(py,px);
  double rho_dot =0;

  if(fabs(rho) < 0.0001){
    std::cout << "Defense: devide by zero...\n";
    rho_dot =0;
  }
  else
    rho_dot = ( px * vx + py * vy) / rho;

  VectorXd z_pred = VectorXd::Zero(3);
  z_pred(0) = rho;
  z_pred(1) = phi;
  z_pred(2) = rho_dot;

  VectorXd y = z-z_pred;

  if (y(1) < -M_PI)
    y(1) += 2*M_PI;
  else if (y(1) > M_PI)
    y(1) -= 2*M_PI;
  
  MatrixXd S = H_ * P_ * H_.transpose() + R_;  // 3x4 * 4x4 * 4x3 + 3x3 = 3x3
  MatrixXd K = P_ * H_.transpose() * S.inverse(); // 4x4 * 4x3 + 3x3
  
  x_ = x_ +K*y;
  P_ = P_ - K*H_*P_;
}

