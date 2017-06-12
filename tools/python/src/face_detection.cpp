// Copyright (C) 2017  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

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


using namespace dlib;
using namespace std;
using namespace boost::python;

typedef matrix<double,0,1> cv;


class face_detection_model_v1
{

public:

    face_detection_model_v1(const std::string& model_filename)
    {
        deserialize(model_filename) >> net;
    }

    boost::python::tuple detect (
        object pyimage,
        const int upsample_num_times
    )
    {
        if (!is_rgb_python_image(pyimage)) {
            throw dlib::error("Unsupported image type, must be RGB image.");
        }

        matrix<rgb_pixel> img;
        assign_image(img, numpy_rgb_image(pyimage));
        
        unsigned int levels = upsample_num_times;
        while (levels > 0) {
            levels--;
            pyramid_up(img);
        }
                
        std::vector<dlib::rectangle> rectangles;
        std::vector<double> detection_confidences;
        for (auto& d : net(img)) {
            rectangles.push_back(d.rect);
            detection_confidences.push_back(d.detection_confidence);
        }
        
        return boost::python::make_tuple(rectangles, detection_confidences);
    }

private:
    
    template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
    template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;

    template <typename SUBNET> using downsampler  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;
    template <typename SUBNET> using rcon5  = relu<affine<con5<45,SUBNET>>>;

    using net_type = loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;
    
    net_type net;
};


// ----------------------------------------------------------------------------------------

void bind_face_detection()
{
    using boost::python::arg;
    {
    class_<face_detection_model_v1>("face_detection_model_v1", "face detection", init<std::string>())
        .def("__call__", &face_detection_model_v1::detect, (arg("img"), arg("upsample_num_times")=0), "face detection");
    }
}

