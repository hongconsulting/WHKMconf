// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#ifndef WH_NAN_REMOVE_H
#define WH_NAN_REMOVE_H

Eigen::VectorXi WH_NAN_indexcol(const Eigen::MatrixXd& input) {
  Eigen::VectorXi output(input.cols());
  int count = 0;
  for (int i = 0; i < input.cols(); ++i) {
    if (!input.col(i).array().isNaN().any()) {
      output(count) = i;
      count++;
    }
  }
  return output.head(count);
} // returns indices of columns with no NAN

Eigen::VectorXd WH_NAN_remove(const Eigen::VectorXd& input) {
  Eigen::VectorXd output(input.size());
  int count = 0;
  for (int i = 0; i < input.size(); ++i) {
    if (!std::isnan(input[i])) {
      output[count] = input[i];
      count++;
    }
  }
  return output.head(count);
}

Eigen::MatrixXd WH_NAN_removecol(const Eigen::MatrixXd& input) {
  Eigen::MatrixXd output(input.rows(), input.cols());
  int count = 0;
  for (int i = 0; i < input.cols(); ++i) {
    if (!input.col(i).array().isNaN().any()) {
      output.col(count) = input.col(i);
      count++;
    }
  }
  return output.leftCols(count);
}

#endif