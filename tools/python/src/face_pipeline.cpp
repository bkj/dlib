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
// #include "face_recognition.cpp"
// #include "face_detection.cpp"
#include <boost/python/args.hpp>


using namespace dlib;
using namespace std;
using namespace boost::python;


// class face_pipeline_v1
// {

// public:

//     face_pipeline_v1(
//         // const std::string& detector_filename,
//         // const std::string& shape_filename,
//         const std::string& face_filename
//     )
//     {
//         face_recognition_model_v1 rec(face_filename);
//     }

// };


// // ----------------------------------------------------------------------------------------

// void bind_face_pipeline()
// {
//     using boost::python::arg;
//     {
//     class_<face_pipeline_v1>("face_pipeline_v1", "face detection", init<std::string>());
//         // .def("__call__", &face_pipeline_v1::detect_single, (arg("img"), arg("upsample_num_times")=0), "face detection")
//         // .def("__call__", &face_pipeline_v1::detect_multi, (arg("imgs"), arg("upsample_num_times")=0), "face detection (multiple)");
        
//     }
// }

