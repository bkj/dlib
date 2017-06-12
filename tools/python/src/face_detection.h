#ifndef DLIB_FACE_DETECTION_PY_
#define DLIB_FACE_DETECTION_PY_

#include <dlib/python.h>
#include <boost/shared_ptr.hpp>
#include <dlib/matrix.h>
#include <boost/python/slice.hpp>
#include <dlib/geometry/vector.h>
#include <dlib/geometry.h>
#include <dlib/dnn.h>
#include <dlib/image_transforms.h>
#include <dlib/image_processing.h>
#include "indexing.h"
#include <boost/python/args.hpp>

using namespace dlib;
using namespace std;
using namespace boost::python;

class face_detection_model_v1
{

public:

    face_detection_model_v1(const std::string& model_filename);
    boost::python::tuple detect_single (object pyimage, const int upsample_num_times);
    boost::python::list detect_multi (boost::python::list pyimages, const int upsample_num_times);
    std::vector<boost::python::tuple> _detect (std::vector<object> pyimages, const int upsample_num_times);

private:
    
    template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
    template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;

    template <typename SUBNET> using downsampler  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;
    template <typename SUBNET> using rcon5  = relu<affine<con5<45,SUBNET>>>;

    using net_type = loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;
    
    net_type net;
};

#endif