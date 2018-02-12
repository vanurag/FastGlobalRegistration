/*
 * Fast Global registration (FGR) provides initial estimate and then libpointmatcher (LPM) is used to perform ICP based pose refinement
 */


#include <ros/ros.h>
#include <stdio.h>
#include "app.h"
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/TransformStamped.h>
#include <nabo/nabo.h>
#include <rviz_talking_view_controller/CLIEngine.h>

// PCL
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/common/common_headers.h>
#include <pcl/common/transforms.h>
#include <pcl/io/ply_io.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/features/fpfh_omp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/conversions.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

// Libpointmatcher
#include "pointmatcher/PointMatcher.h"

typedef PointMatcher<float> PM;
typedef PM::DataPoints DP;

class FGR
{
 private:
  ros::NodeHandle nh_, nh_private_;
  ros::Subscriber subPcl, subCLI;
  ros::Publisher pubPose_;
  geometry_msgs::TransformStamped pose_msg_;  // resulting TF
  rviz_talking_view_controller::CLIEngine cli_config_msg_;

  bool do_relocalize_;

  typedef enum {
    //! Using Libpointmatcher for ICP
    ICP_LPM,
    //! ICP_LPM but operates on image hierarchy
    ICP_LPM_HIERARCHY,
  } ICPType;

  ICPType icp_type_;

  // Libpointmatcher
  DP* parent_scene_dp_;
  PM::ICP icp_;

  // libnabo kd-tree
  boost::shared_ptr<Nabo::NNSearchF> nns_;

  // PCL Point cloud to LPM Point cloud
  DP PCLPointCloudToLPMPointCloud(const pcl::PointCloud<pcl::PointNormal>::ConstPtr& pcl_cloud);

  // FGR app
  CApp app_;

  // params
  float div_factor_;                 // Division factor used for graduated non-convexity
  bool use_abs_scale_;               // Measure distance in absolute scale (1) or in scale relative to the diameter of the model (0)
  float max_corr_dist_;	            // Maximum correspondence distance (also see comment of USE_ABSOLUTE_SCALE)
  int max_iter_;	                    // Maximum number of iteration
  float tuple_scale_; 		            // Similarity measure used for tuples of feature points.
  int tuple_max_cnt_;                // Maximum tuple numbers.
  std::string stored_scene_feat_;    // path to stored features file of the scene
  std::string stored_scene_mesh_;    // path to stored mesh file of the scene
  std::string lpm_config_file_;      // configuration file used by LPM
  bool do_geom_check_;               // whether or not to perform icp based refinement


  bool LoadParameters() {
    bool could_load_params = true;

    could_load_params &= nh_private_.getParam("div_factor", div_factor_);
    could_load_params &= nh_private_.getParam("use_abs_scale", use_abs_scale_);
    could_load_params &= nh_private_.getParam("max_corr_dist", max_corr_dist_);
    could_load_params &= nh_private_.getParam("max_iter", max_iter_);
    could_load_params &= nh_private_.getParam("tuple_scale", tuple_scale_);
    could_load_params &= nh_private_.getParam("tuple_max_cnt", tuple_max_cnt_);
    could_load_params &= nh_private_.getParam("stored_scene_feat", stored_scene_feat_);
    could_load_params &= nh_private_.getParam("stored_scene_mesh", stored_scene_mesh_);
    could_load_params &= nh_private_.getParam("lpm_config_file", lpm_config_file_);
    could_load_params &= nh_private_.getParam("do_geom_check", do_geom_check_);
    return could_load_params;
  }

  // PCL pointcloud callback
  void meshCallback(const sensor_msgs::PointCloud2::ConstPtr& msg);

  // CLI config callbacl
  void cliConfigCallback(const rviz_talking_view_controller::CLIEngine::ConstPtr& msg);

  // generate features for PCL point cloud
  void generateFeatures(const pcl::PointCloud<pcl::PointNormal>::Ptr& cloud,
                        Points &cloud_points, Feature &cloud_features);

  // Use LPM for ICP routine
  Matrix4f getLPMICPTF(const DP& scene_dp, Matrix4f& init_estimate);

  // registration TF publisher
  void publishTF(const Matrix4f& tf);

 public:
  FGR(ros::NodeHandle& nh, ros::NodeHandle& nh_private) {
    nh_ = nh;
    nh_private_ = nh_private;

    // by default not enabling. Wait for CLI config msg.
    do_relocalize_ = false;

    if (!LoadParameters()) {
      std::cout << "failed to load user settings!" << std::endl;
      exit(1);
    }

    // APP
    app_.SetUserParams(div_factor_, use_abs_scale_, max_corr_dist_, tuple_scale_,
                      tuple_max_cnt_);
    app_.PrintParams();
    std::cout << "ITERATION_NUMBER: " << max_iter_ << std::endl << std::endl;

    app_.ReadFeature(stored_scene_feat_.c_str()); // stored scene features

    // ROS
    // mesh subscriber
    subPcl = nh.subscribe<sensor_msgs::PointCloud2> ("/itm/pcl", 1, &FGR::meshCallback, this);
    // CLI config subscriber
    subCLI = nh.subscribe<rviz_talking_view_controller::CLIEngine> ("/itm/cli/config", 1, &FGR::cliConfigCallback, this);
    // TF publisher
    pose_msg_.header.frame_id = "parent_vi_global";
    pose_msg_.child_frame_id = "vi_global";
    pubPose_ = nh.advertise<geometry_msgs::TransformStamped>("itm/fgr_pose", 1);

    // LPM
    parent_scene_dp_ = new DP(DP::load(stored_scene_mesh_));
    icp_type_ = ICPType::ICP_LPM; // Change this as per need
    // load YAML config
    std::ifstream conf(lpm_config_file_.c_str());
    if (!conf.good())
    {
      std::cerr << "Cannot open ICP config file"; exit(1);
    }
    icp_.loadFromYaml(conf);
  };
  ~FGR(void) {};
};

void FGR::cliConfigCallback(const rviz_talking_view_controller::CLIEngine::ConstPtr& msg)
{
  cli_config_msg_ = *msg;
  if (cli_config_msg_.relocalize) do_relocalize_ = true;
}

void FGR::meshCallback(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
  std::cout << "Got a PCL msg!!" << std::endl;
  if (!do_relocalize_) return;
  // generate pcl pointcloud
  pcl::PCLPointCloud2 msg_pcl2;
  pcl_conversions::toPCL(*msg, msg_pcl2);
  pcl::PointCloud<pcl::PointNormal>::Ptr msg_cloud(new pcl::PointCloud<pcl::PointNormal>);
  pcl::fromPCLPointCloud2(msg_pcl2, *msg_cloud);

  // generate LPM point cloud
  DP msg_dp = PCLPointCloudToLPMPointCloud(msg_cloud);

  // generate features
  Points msg_points;
  Feature msg_features;
  generateFeatures(msg_cloud, msg_points, msg_features);

  // perform FGR
  if (app_.GetNumPcl() >= 2) {
    std::cout << "Updating PCL at index 1" << std::endl;
    app_.LoadFeature(msg_points, msg_features, 1);
    app_.NormalizePoints(1);
  } else {
    std::cout << "Uploading PCL to index 1" << std::endl;
    app_.LoadFeature(msg_points, msg_features);
    app_.NormalizePoints();
  }
  app_.AdvancedMatching();
  app_.OptimizePairwise(true, max_iter_);
  Matrix4f TF = app_.GetTrans();

  std::cout << "After FGR transformation:\n" << TF << std::endl;

  // ICP refinement using LPM
  if (do_geom_check_) {
    TF = getLPMICPTF(msg_dp, TF);
    std::cout << "After ICP refinement transformation:" << std::endl << TF << std::endl;
  }

  // publish resulting TF between current scene and the parent scene
  publishTF(TF);

  do_relocalize_ = false;
}

// Use LPM for ICP routine
Matrix4f FGR::getLPMICPTF(const DP& scene_dp, Matrix4f& init_estimate) {
  PM::TransformationParameters scene_tf = PM::TransformationParameters::Identity(4, 4);
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 4; ++col) {
      scene_tf(row, col) = init_estimate(row, col);
    }
  }
//  PM::TransformationParameters parent_scene_tf = PM::TransformationParameters::Identity(4, 4);

  PM::Transformation* rigidTrans;
  rigidTrans = PM::get().REG(Transformation).create("RigidTransformation");

  // TF sanity check
  if (!rigidTrans->checkParameters(scene_tf)) {
    std::cerr << std::endl
       << "Provided init_estimate is not rigid, identity will be used"
       << std::endl;
    scene_tf = PM::TransformationParameters::Identity(4, 4);
  }

  // initialize current scan by best pose estimate from previous run
  // Also transform scene to global ref frame
  const DP transformed_scene_dp = rigidTrans->compute(scene_dp, scene_tf);

  // Compute the transformation to express scene in parent scene ref
  PM::TransformationParameters T = icp_(transformed_scene_dp, *parent_scene_dp_);
  std::cout << "LPM match ratio: " << icp_.errorMinimizer->getWeightedPointUsedRatio() << std::endl;
//  std::cout << "ICP transformation:" << std::endl << T << std::endl;

  // Transform data to express it in ref
  DP data_out(transformed_scene_dp);
  icp_.transformations.apply(data_out, T);

  // Safe files to see the results
//  GlobalScene.save(LPMBaseDir_ + "test_ref.vtk")
//  GlobalScan.save(LPMBaseDir_ + "test_data_in.vtk");
//  data_out.save(LPMBaseDir_ + "test_data_out.vtk");

  // Refined scene TF
  PM::TransformationParameters refined_scene_tf = scene_tf * T;
  Matrix4f refined_scene_pose;
  refined_scene_pose.setIdentity();
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 4; ++col) {
      refined_scene_pose(row, col) = refined_scene_tf(row, col);
    }
  }
  return refined_scene_pose;
}

void FGR::publishTF(const Matrix4f& tf) {
//  std::cout << "pose: " << tf << std::endl;
  pose_msg_.header.stamp = ros::Time::now();

  Matrix3f pose_rot = tf.block<3,3>(0,0);
  Vector3f pose_trans = tf.block<3,1>(0,3);
//  std::cout << "pose rot: " << pose_rot << std::endl;
//  std::cout << "pose trans: " << pose_trans << std::endl;

  pose_msg_.transform.translation.x = pose_trans[0];
  pose_msg_.transform.translation.y = pose_trans[1];
  pose_msg_.transform.translation.z = pose_trans[2];

  Eigen::Quaternionf pose_q(pose_rot);
  pose_msg_.transform.rotation.x = pose_q.x();
  pose_msg_.transform.rotation.y = pose_q.y();
  pose_msg_.transform.rotation.z = pose_q.z();
  pose_msg_.transform.rotation.w = pose_q.w();

  pubPose_.publish(pose_msg_);
}

void FGR::generateFeatures(const pcl::PointCloud<pcl::PointNormal>::Ptr& cloud,
                           Points &cloud_points, Feature &cloud_features)
{
  // sub-sample the input cloud
  pcl::PointCloud<pcl::PointNormal>::Ptr cloud_subsample(new pcl::PointCloud<pcl::PointNormal>);
  cloud_subsample = cloud;
  std::cerr << "PointCloud before filtering: " << cloud->width * cloud->height
         << " data points (" << pcl::getFieldsList (*cloud) << ").";
  // Create the filtering object
  pcl::VoxelGrid<pcl::PointNormal> sor;
  sor.setInputCloud(cloud);
  sor.setLeafSize(0.01f, 0.01f, 0.01f);
  sor.filter(*cloud_subsample);
  std::cerr << "PointCloud after filtering: " << cloud_subsample->width * cloud_subsample->height
       << " data points (" << pcl::getFieldsList (*cloud_subsample) << ").";

  pcl::FPFHEstimationOMP<pcl::PointNormal, pcl::PointNormal, pcl::FPFHSignature33> fest;
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr pcl_features(new pcl::PointCloud<pcl::FPFHSignature33>());
  fest.setRadiusSearch(0.05);
  fest.setInputCloud(cloud_subsample);
  fest.setInputNormals(cloud_subsample);
  fest.compute(*pcl_features);

  int nV = cloud_subsample->size(), nDim = 33;
  for (int v = 0; v < nV; v++) {
    const pcl::PointNormal &pt = cloud_subsample->points[v];
    Vector3f c_pt(pt.x, pt.y, pt.z);
    cloud_points.push_back(c_pt);
    const pcl::FPFHSignature33 &feature = pcl_features->points[v];
    std::vector<float> feat_vec(std::begin(feature.histogram), std::end(feature.histogram));
    Map<VectorXf> c_feat(&feat_vec[0], nDim);
    cloud_features.push_back(c_feat);
  }
}

// PCL Point cloud to LPM Point cloud
DP FGR::PCLPointCloudToLPMPointCloud(const pcl::PointCloud<pcl::PointNormal>::ConstPtr& pcl_cloud) {

  DP::Labels feature_labels;
  feature_labels.push_back(DP::Label("x"));
  feature_labels.push_back(DP::Label("y"));
  feature_labels.push_back(DP::Label("z"));

  PM::Matrix features(4, pcl_cloud->size());
  for (int i = 0; i < pcl_cloud->size(); ++i) {
    features(0, i) = pcl_cloud->points[i].x;
    features(1, i) = pcl_cloud->points[i].y;
    features(2, i) = pcl_cloud->points[i].z;
    features(3, i) = 1.0;
    // check normal
//    std::cout << "normal: " << i << " " << pcl_cloud->points[i].normal_x << " " << pcl_cloud->points[i].normal_y << " " << pcl_cloud->points[i].normal_z << std::endl;
  }

  return DP(features, feature_labels);
}

int main(int argc, char** argv)
{
  ros::Time::init();
  ros::init(argc, argv, "fgr_node");
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");
  ROS_INFO("Starting fgr_node with node name %s", ros::this_node::getName().c_str());

  // FGR
  FGR fgr(nh, nh_private);

  ros::spin();

  return 0;
}
