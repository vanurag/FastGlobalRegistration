// Assume a point cloud with normal is given as
// pcl::PointCloud<pcl::PointNormal>::Ptr object

// PCL
#include <pcl/point_types.h>
#include <pcl/common/common_headers.h>
#include <pcl/common/transforms.h>
#include <pcl/io/ply_io.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/features/fpfh_omp.h>
#include <pcl/filters/voxel_grid.h>

int main(int argc, char *argv[])
{
  if (argc < 3) {
    std::cout << "Usage: ./FeatureGenerator <source: PLY file> <dest: Feature file>" << std::endl;
    exit(1);
  }

  // load ply file
  pcl::PointCloud<pcl::PointNormal> bla;
  pcl::PointCloud<pcl::PointNormal>::Ptr object(new pcl::PointCloud<pcl::PointNormal>);
  pcl::PLYReader Reader;
  Reader.read(argv[1], *object);

  // sub-sample the input cloud
  pcl::PointCloud<pcl::PointNormal>::Ptr object_subsample(new pcl::PointCloud<pcl::PointNormal>);
  object_subsample = object;
  std::cerr << "PointCloud before filtering: " << object->width * object->height
         << " data points (" << pcl::getFieldsList (*object) << ").";
  // Create the filtering object
  pcl::VoxelGrid<pcl::PointNormal> sor;
  sor.setInputCloud(object);
  sor.setLeafSize(0.01f, 0.01f, 0.01f);
  sor.filter(*object_subsample);
  std::cerr << "PointCloud after filtering: " << object_subsample->width * object_subsample->height
       << " data points (" << pcl::getFieldsList (*object_subsample) << ").";

  pcl::FPFHEstimationOMP<pcl::PointNormal, pcl::PointNormal, pcl::FPFHSignature33> fest;
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr object_features(new pcl::PointCloud<pcl::FPFHSignature33>());
  fest.setRadiusSearch(0.05);
  fest.setInputCloud(object_subsample);
  fest.setInputNormals(object_subsample);
  fest.compute(*object_features);

  FILE* fid = fopen(argv[2], "wb");
  int nV = object_subsample->size(), nDim = 33;
  fwrite(&nV, sizeof(int), 1, fid);
  fwrite(&nDim, sizeof(int), 1, fid);
  for (int v = 0; v < nV; v++) {
      const pcl::PointNormal &pt = object_subsample->points[v];
      float xyz[3] = {pt.x, pt.y, pt.z};
      fwrite(xyz, sizeof(float), 3, fid);
      const pcl::FPFHSignature33 &feature = object_features->points[v];
      fwrite(feature.histogram, sizeof(float), 33, fid);
  }
  fclose(fid);
}
