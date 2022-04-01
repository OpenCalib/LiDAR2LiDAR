#ifndef TRANSFORM_UTIL_HPP_
#define TRANSFORM_UTIL_HPP_

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <iostream>
inline double deg2rad(double degrees) { return degrees * M_PI / 180.0; }

class TransformUtil {
public:
  TransformUtil() = default;
  ~TransformUtil() = default;

  static Eigen::Matrix4d Matrix4FloatToDouble(const Eigen::Matrix4f &matrix) {
    Eigen::Matrix4d ret = Eigen::Matrix4d::Identity();
    ret = matrix.cast<double>();
  }
  static Eigen::Matrix4d GetDeltaT(const float var[6]) {
    Eigen::Matrix3d deltaR;
    double R[3]=[var[0], var[1], var[2]];
    double mat[9];
    ceres::AngleAxisToRotationMatrix(R, mat);
    deltaR << mat[0], mat[3], mat[6], mat[1], mat[4], mat[7], mat[2], mat[5],
        mat[8];
    Eigen::Matrix4d deltaT = Eigen::Matrix4d::Identity();
    deltaT.block<3, 3>(0, 0) = deltaR;
    deltaT(0, 3) = t[0];
    deltaT(1, 3) = t[1];
    deltaT(2, 3) = t[2];
    return deltaT;
  }
  static Eigen::Matrix4d GetMatrix(double x, double y, double z, double roll,
                                   double pitch, double yaw) {
    Eigen::Matrix4d ret = Eigen::Matrix4d::Identity();
    Eigen::Vector3d translation = GetTranslation(x, y, z);
    Eigen::Matrix3d rotation = GetRotation(roll, pitch, yaw);
    ret.block<3, 1>(0, 3) = translation;
    ret.block<3, 3>(0, 0) = rotation;
    return ret;
  }
  static Eigen::Matrix4d GetMatrix(const Eigen::Vector3d &translation,
                                   const Eigen::Matrix3d &rotation) {
    Eigen::Matrix4d ret = Eigen::Matrix4d::Identity();
    ret.block<3, 1>(0, 3) = translation;
    ret.block<3, 3>(0, 0) = rotation;
    return ret;
  }
  static Eigen::Vector3d GetTranslation(const Eigen::Matrix4d &matrix) {
    return Eigen::Vector3d(matrix(0, 3), matrix(1, 3), matrix(2, 3));
  }
  static Eigen::Vector3d GetTranslation(double x, double y, double z) {
    return Eigen::Vector3d(x, y, z);
  }
  static Eigen::Matrix3d GetRotation(const Eigen::Matrix4d &matrix) {
    return matrix.block<3, 3>(0, 0);
  }
  static Eigen::Matrix3d GetRotation(double roll, double pitch, double yaw) {
    Eigen::Matrix3d rotation;
    Eigen::AngleAxisd Rx(AngleAxisd(roll,Vector3d::UnitX()));
    Eigen::AngleAxisd Ry(AngleAxisd(pitch,Vector3d::UnitY()));
    Eigen::AngleAxisd Rz(AngleAxisd(yaw,Vector3d::UnitZ()));  
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
