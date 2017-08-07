// ----------------------------------------------------------------------------
// -                       Fast Global Registration                           -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) Intel Corporation 2016
// Qianyi Zhou <Qianyi.Zhou@gmail.com>
// Jaesik Park <syncle@gmail.com>
// Vladlen Koltun <vkoltun@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------
#include <vector>
#include <flann/flann.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;
using namespace std;

typedef vector<Vector3f> Points;
typedef vector<VectorXf> Feature;

class CApp{
public:
  int GetNumPcl();
  void LoadFeature(const Points& pts, const Feature& feat);
  void LoadFeature(const Points& pts, const Feature& feat, const int& id);
	void ReadFeature(const char* filepath);
  void ReadFeature(const char* filepath, const int& id);
	void NormalizePoints();
  void NormalizePoints(const int& id);
	void AdvancedMatching();
	void WriteTrans(const char* filepath);
  Matrix4f GetTrans();
	double OptimizePairwise(bool decrease_mu_, int numIter_);
  void SetUserParams(const float& div_factor, const bool& use_abs_scale, const float& max_corr_dist,
                     const float& tuple_scale, const int& tuple_max_cnt) {
    div_factor_ = div_factor;
    use_abs_scale_ = use_abs_scale;
    max_corr_dist_ = max_corr_dist;
    tuple_scale_ = tuple_scale;
    tuple_max_cnt_ = tuple_max_cnt;
  }
  void PrintParams() {
    std::cout << "Using run-time params:" << std::endl;
    std::cout << "DIV_FACTOR: " << div_factor_ << std::endl;
    std::cout << "USE_ABSOLUTE_SCALE: " << use_abs_scale_ << std::endl;
    std::cout << "MAX_CORR_DIST: " << max_corr_dist_ << std::endl;
    std::cout << "TUPLE_SCALE: " << tuple_scale_ << std::endl;
    std::cout << "TUPLE_MAX_CNT: " << tuple_max_cnt_ << std::endl;
  }

  CApp() {
    div_factor_ = 1.4;
    use_abs_scale_ = true;
    max_corr_dist_ = 0.025;
    tuple_scale_ = 0.95;
    tuple_max_cnt_ = 1000;
  }

private:
	// containers
	vector<Points> pointcloud_;
	vector<Feature> features_;
	Matrix4f TransOutput_;
	vector<pair<int, int>> corres_;

  // user params
  float div_factor_;                  // Division factor used for graduated non-convexity
  bool use_abs_scale_;                // Measure distance in absolute scale (1) or in scale relative to the diameter of the model (0)
  float max_corr_dist_; 	            // Maximum correspondence distance (also see comment of USE_ABSOLUTE_SCALE)
  float tuple_scale_; 		            // Similarity measure used for tuples of feature points.
  int tuple_max_cnt_;                 // Maximum tuple numbers.

	// for normalization
	Points Means;
	float GlobalScale;
	float StartScale;

	// some internal functions
	void ReadFeature(const char* filepath, Points& pts, Feature& feat);

	void SearchFLANNTree(flann::Index<flann::L2<float>>* index,
		VectorXf& input,
		std::vector<int>& indices,
		std::vector<float>& dists,
		int nn);
};
