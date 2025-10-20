// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#ifndef WH_SELECT_BOOL_H
#define WH_SELECT_BOOL_H

Eigen::VectorXd WH_select_bool(const Eigen::VectorXd& input, const Eigen::Array<bool, Eigen::Dynamic, 1>& mask) {
  int count = (mask == true).count();
  Eigen::VectorXd output(count);
  int index = 0;
  for (int i = 0; i < input.size(); ++i) {
    if (mask[i]) {
      output(index++) = input(i);
    }
  }
  return output;
}

Eigen::VectorXi WH_select_bool(const Eigen::VectorXi& input, const Eigen::Array<bool, Eigen::Dynamic, 1>& mask) {
  int count = (mask == true).count();
  Eigen::VectorXi output(count);
  int index = 0;
  for (int i = 0; i < input.size(); ++i) {
    if (mask[i]) {
      output(index++) = input(i);
    }
  }
  return output;
}

Eigen::MatrixXd WH_selectrow_bool(const Eigen::MatrixXd& input, const Eigen::Array<bool, Eigen::Dynamic, 1>& mask) {
  Eigen::MatrixXd output(mask.count(), input.cols());
  int index = 0;
  for (int i = 0; i < input.rows(); ++i) {
    if (mask[i]) {
      output.row(index++) = input.row(i);
    }
  }
  return output;
}

#endif