// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#ifndef WH_UNIQUE_H
#define WH_UNIQUE_H

#include "WH_NAN_remove.h"

// Eigen::VectorXd WH_unique(const Eigen::VectorXd& input) { //will also sort
//   std::vector<double> output(input.data(), input.data() + input.size());
//   std::sort(output.begin(), output.end());
//   std::vector<double>::iterator it = std::unique(output.begin(), output.end());
//   output.resize(std::distance(output.begin(), it));
//   return Eigen::Map<Eigen::VectorXd>(output.data(), output.size());
// }

Eigen::VectorXd WH_unique(const Eigen::VectorXd& input) {
  Eigen::VectorXd cleaned = WH_NAN_remove(input);
  std::vector<double> sorted(cleaned.data(), cleaned.data() + cleaned.size());
  std::sort(sorted.begin(), sorted.end());
  auto it = std::unique(sorted.begin(), sorted.end());
  int n_unique = std::distance(sorted.begin(), it);
  return Eigen::Map<Eigen::VectorXd>(sorted.data(), n_unique);
}

Eigen::VectorXi WH_unique(const Eigen::VectorXi& input) { //will also sort
  std::vector<int> output(input.data(), input.data() + input.size());
  std::sort(output.begin(), output.end());
  std::vector<int>::iterator it = std::unique(output.begin(), output.end());
  output.resize(std::distance(output.begin(), it));
  return Eigen::Map<Eigen::VectorXi>(output.data(), output.size());
}


std::vector<std::set<int>> WH_unique(const std::vector<std::set<int>>& input) {
  std::vector<std::set<int>> output = input;
  std::sort(output.begin(), output.end());
  std::vector<std::set<int>>::iterator it = std::unique(output.begin(), output.end());
  output.resize(std::distance(output.begin(), it));
  return output;
}

#endif