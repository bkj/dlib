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
#include "face_detection.h"
#include <boost/python/args.hpp>

using namespace dlib;
using namespace std;
using namespace boost::python;


face_detection_model_v1::face_detection_model_v1(const std::string& model_filename)
{
    deserialize(model_filename) >> net;
}

boost::python::tuple face_detection_model_v1::detect_single (
    object pyimage,
    const int upsample_num_times
)
{
    std::vector<object> pyimages(1, pyimage);
    return _detect(pyimages, upsample_num_times)[0];
}

boost::python::list face_detection_model_v1::detect_multi (
    boost::python::list pyimages,
    const int upsample_num_times
)
{
    return vector_to_python_list(_detect(python_list_to_vector<object>(pyimages), upsample_num_times));
}

std::vector<boost::python::tuple> face_detection_model_v1::_detect (
    std::vector<object> pyimages,
    const int upsample_num_times
)
{
    std::vector<matrix<rgb_pixel>> imgs;
    for (auto& pyimage : pyimages) {
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
        imgs.push_back(img);
    }
            
    std::vector<boost::python::tuple> output;
    for (auto& ds : net(imgs)) {
        std::vector<dlib::rectangle> rectangles;
        std::vector<double> detection_confidences;
        for (auto& d : ds) {
            rectangles.push_back(d.rect);
            detection_confidences.push_back(d.detection_confidence);
        }
        output.push_back(boost::python::make_tuple(rectangles, detection_confidences));
    }
    
    return output;
}


// ----------------------------------------------------------------------------------------

void bind_face_detection()
{
    using boost::python::arg;
    {
    class_<face_detection_model_v1>("face_detection_model_v1", "face detection", init<std::string>())
        .def("__call__", &face_detection_model_v1::detect_single, (arg("img"), arg("upsample_num_times")=0), "face detection")
        .def("__call__", &face_detection_model_v1::detect_multi, (arg("imgs"), arg("upsample_num_times")=0), "face detection (multiple)");
        
    }
}