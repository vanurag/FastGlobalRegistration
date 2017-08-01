#include <ros/ros.h>
#include <stdio.h>
#include "app.h"
#include <sensor_msgs/PointCloud2.h>

// PCL
#include <pcl/point_types.h>
#include <pcl/common/common_headers.h>
#include <pcl/common/transforms.h>
#include <pcl/io/ply_io.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/features/fpfh_omp.h>
#include <pcl/filters/voxel_grid.h>


class FGR
{
 private:
  ros::NodeHandle nh_, nh_private_;
  ros::Subscriber subPcl;

 public:
  // FGR app
  CApp app;

  // params
  float div_factor;                 // Division factor used for graduated non-convexity
  bool use_abs_scale;               // Measure distance in absolute scale (1) or in scale relative to the diameter of the model (0)
  float max_corr_dist;	            // Maximum correspondence distance (also see comment of USE_ABSOLUTE_SCALE)
  int max_iter;	                    // Maximum number of iteration
  float tuple_scale; 		            // Similarity measure used for tuples of feature points.
  int tuple_max_cnt;                // Maximum tuple numbers.
  std::string stored_scene_feat;    // path to stored features file of the scene


  bool LoadParameters() {
    bool could_load_params = true;

    could_load_params &= nh_private_.getParam("div_factor", div_factor);
    could_load_params &= nh_private_.getParam("use_abs_scale", use_abs_scale);
    could_load_params &= nh_private_.getParam("max_corr_dist", max_corr_dist);
    could_load_params &= nh_private_.getParam("max_iter", max_iter);
    could_load_params &= nh_private_.getParam("tuple_scale", tuple_scale);
    could_load_params &= nh_private_.getParam("tuple_max_cnt", tuple_max_cnt);
    could_load_params &= nh_private_.getParam("stored_scene_feat", stored_scene_feat);
    return could_load_params;
  }

  // PCL pointcloud callback
  void meshCallback(const sensor_msgs::PointCloud2::ConstPtr& msg);

  // generate features for PCL point cloud
  void generateFeatures(const pcl::PointCloud<pcl::PointNormal>::Ptr& cloud,
                        pcl::PointCloud<pcl::FPFHSignature33>::Ptr cloud_features);

  FGR(ros::NodeHandle& nh, ros::NodeHandle& nh_private) {
    nh_ = nh;
    nh_private_ = nh_private;

    subPcl = nh.subscribe<sensor_msgs::PointCloud2> ("/itm/pcl", 1, &FGR::meshCallback, this);
  };
  ~FGR(void) {};
};

void FGR::meshCallback(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
  std::cout << "Got a PCL msg!!" << std::endl;
  // generate pcl pointcloud

  // generate normals

  // perform FGR
  app.ReadFeature("bla");
  app.NormalizePoints();
  app.AdvancedMatching();
  app.OptimizePairwise(true, max_iter);
  app.WriteTrans("blaa");
}

void FGR::generateFeatures(const pcl::PointCloud<pcl::PointNormal>::Ptr& cloud,
                           pcl::PointCloud<pcl::FPFHSignature33>::Ptr cloud_features)
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
  fest.setRadiusSearch(0.05);
  fest.setInputCloud(cloud_subsample);
  fest.setInputNormals(cloud_subsample);
  fest.compute(*cloud_features);
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
  if (!fgr.LoadParameters()) {
    std::cout << "failed to load user settings!" << std::endl;
    return -1;
  }

  std::cout << "Using run-time params:" << std::endl;
  std::cout << "DIV_FACTOR: " << fgr.div_factor << std::endl;
  std::cout << "USE_ABSOLUTE_SCALE: " << fgr.use_abs_scale << std::endl;
  std::cout << "MAX_CORR_DIST: " << fgr.max_corr_dist << std::endl;
  std::cout << "ITERATION_NUMBER: " << fgr.max_iter << std::endl;
  std::cout << "TUPLE_SCALE: " << fgr.tuple_scale << std::endl;
  std::cout << "TUPLE_MAX_CNT: " << fgr.tuple_max_cnt << std::endl;
  std::cout << "Scene features file: " << fgr.stored_scene_feat << std::endl << std::endl;

  fgr.app.ReadFeature(fgr.stored_scene_feat.c_str()); // stored scene features

  ros::spin();

  return 0;
}
