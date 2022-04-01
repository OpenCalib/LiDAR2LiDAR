#ifndef TRANSFORM_UTIL_HPP_
#define TRANSFORM_UTIL_HPP_

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <iostream>
inline double deg2rad(double degrees) { return degrees * M_PI / 180.0; }

class TransformUtil {
public:
  TransformUtil() = default;
  ~TransformUtil() = default;

  static Eigen::Matrix4d Matrix4FloatToDouble(const Eigen::Matrix4f &matrix) {
    Eigen::Matrix4d ret = Eigen::Matrix4d::Identity();
    ret = matrix.cast<double>();
    return ret;
  }
  static Eigen::Matrix4d GetDeltaT(const float var[6]) {
    Eigen::Matrix3d deltaR;
    deltaR = GetRotation(deg2rad(var[0]), deg2rad(var[1]), deg2rad(var[2]));
    Eigen::Matrix4d deltaT = Eigen::Matrix4d::Identity();
    deltaT.block<3, 3>(0, 0) = deltaR;
    deltaT(0, 3) = var[3];;
    deltaT(1, 3) = var[4];;
    deltaT(2, 3) = var[5];;
    return deltaT;
  }
  static Eigen::Matrix4d GetMatrix(double x, double y, double z, double roll,
                                   double pitch, double yaw) {
    Eigen::Matrix4d ret = Eigen::Matrix4d::Identity();
    Eigen::Vector3d T = GetTranslation(x, y, z);
    Eigen::Matrix3d R = GetRotation(roll, pitch, yaw);
    ret.block<3, 1>(0, 3) = T;
    ret.block<3, 3>(0, 0) = R;
    return ret;
  }
  static Eigen::Matrix4d GetMatrix(const Eigen::Vector3d &T,
                                   const Eigen::Matrix3d &R) {
    Eigen::Matrix4d ret = Eigen::Matrix4d::Identity();
    ret.block<3, 1>(0, 3) = T;
    ret.block<3, 3>(0, 0) = R;
    return ret;
  }
  static Eigen::Vector3d GetTranslation(const Eigen::Matrix4d &matrix) {
    Eigen::Vector3d T;
    T << matrix(0, 3), matrix(1, 3), matrix(2, 3);
    return T;
  }
  static Eigen::Vector3d GetTranslation(double x, double y, double z) {
    Eigen::Vector3d T;
    T << x, y, z;
    return T;
  }
  static Eigen::Matrix3d GetRotation(const Eigen::Matrix4d &matrix) {
    Eigen::Matrix3d R = matrix.block<3, 3>(0, 0);
    return R;
  }
  static Eigen::Matrix3d GetRotation(double roll, double pitch, double yaw) {
    Eigen::Matrix3d rotation;
    Eigen::AngleAxisd Rx(Eigen::AngleAxisd(roll,Eigen::Vector3d::UnitX()));
    Eigen::AngleAxisd Ry(Eigen::AngleAxisd(pitch,Eigen::Vector3d::UnitY()));
    Eigen::AngleAxisd Rz(Eigen::AngleAxisd(yaw,Eigen::Vector3d::UnitZ()));  
    rotation = Rz * Ry * Rx;
    return rotation;
  }
  static double GetX(const Eigen::Matrix4d &matrix) { return matrix(0, 3); }
  static double GetY(const Eigen::Matrix4d &matrix) { return matrix(1, 3); }
  static double GetZ(const Eigen::Matrix4d &matrix) { return matrix(2, 3); }
  /* avoid using it: matrix.block<3, 3>(0, 0).eulerAngles(0, 1, 2)[0];*/
  static double GetRoll(const Eigen::Matrix4d &matrix) {
    Eigen::Matrix3d R = matrix.block<3, 3>(0, 0);
    double y = atan2(R(1,0), R(0,0));
    double r =
        atan2(R(0,1) * sin(y) - R(1,1) * cos(y), -R(0,2) * sin(y) + R(1,2) * cos(y));
    return r;
  }
  static double GetPitch(const Eigen::Matrix4d &matrix) {
    Eigen::Matrix3d R = matrix.block<3, 3>(0, 0);
    double y = atan2(R(1,0), R(0,0));
    double p = atan2(-R(2,0), R(0,0) * cos(y) + R(1,0) * sin(y));

    return p;
  }
  static double GetYaw(const Eigen::Matrix4d &matrix) {
    Eigen::Matrix3d R = matrix.block<3, 3>(0, 0);
    double y = atan2(R(1,0), R(0,0));

    return y;
  }
};

#endif // TRANSFORM_UTIL_HPP_
