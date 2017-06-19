// Copyright (C) 2017  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <dlib/python.h>
#include <boost/shared_ptr.hpp>
#include <dlib/matrix.h>
#include <boost/python/slice.hpp>
#include <dlib/geometry/vector.h>
#include <dlib/dnn.h>
#include <dlib/image_transforms.h>
#include "indexing.h"
#include "face_recognition.h"

using namespace dlib;
using namespace std;
using namespace boost::python;

face_recognition_model_v1::face_recognition_model_v1(const std::string& model_filename)
{
    deserialize(model_filename) >> net;

    cropper = make_shared<random_cropper>();
    cropper->set_chip_dims(150,150);
    cropper->set_randomly_flip(true);
    cropper->set_max_object_height(0.99999);
    cropper->set_background_crops_fraction(0);
    cropper->set_min_object_height(0.97);
    cropper->set_translate_amount(0.02);
    cropper->set_max_rotation_degrees(3);
}
    
std::vector<matrix<double,0,1>> face_recognition_model_v1::compute_batch_face_descriptors (
    boost::python::list pyimages,
    boost::python::list all_pyfaces,
    const int num_jitters
)
{   
    // Convert python list of list to vector of vectors
    std::vector<std::vector<full_object_detection>> all_faces;
    for (int i = 0; i < len(all_pyfaces); ++i) {
        std::vector<full_object_detection> faces;
        for (int j = 0; j < len(all_pyfaces[i]); ++j) {
            faces.push_back(extract<full_object_detection>(all_pyfaces[i][j]));
        }
        all_faces.push_back(faces);
    }

    return _compute_batch_face_descriptors(
        python_list_to_vector<object>(pyimages), 
        all_faces,
        num_jitters
    );
}
    
std::vector<matrix<double,0,1>> face_recognition_model_v1::_compute_batch_face_descriptors (
    std::vector<object> pyimages,
    const std::vector<std::vector<full_object_detection>>& all_faces,
    const int num_jitters
)
{
    dlib::array<matrix<rgb_pixel>> all_face_chips;
    for (int i = 0; i < pyimages.size(); ++i) {

        for (auto& f : all_faces[i]) {
            if (f.num_parts() != 68) {
                throw dlib::error("The full_object_detection must use the iBUG 300W 68 point face landmark style.");
            }
        }            

        if (!is_rgb_python_image(pyimages[i])) {
            throw dlib::error("Unsupported image type, must be RGB image.");
        }
        
        std::vector<chip_details> dets;
        for (auto& f : all_faces[i]) {
            dets.push_back(get_face_chip_details(f, 150, 0.25));
        }
        
        dlib::array<matrix<rgb_pixel>> face_chips;
        extract_image_chips(numpy_rgb_image(pyimages[i]), dets, face_chips);
        for (auto& c : face_chips) {
            all_face_chips.push_back(c);
        }
    }
    
    return _describe_chips(all_face_chips, num_jitters);
}

matrix<double,0,1> face_recognition_model_v1::compute_face_descriptor (
    object img,
    const full_object_detection& face,
    const int num_jitters
)
{
    std::vector<full_object_detection> faces(1, face);
    return compute_face_descriptors(img, faces, num_jitters)[0];
}

std::vector<matrix<double,0,1>> face_recognition_model_v1::compute_face_descriptors (
    object img,
    const std::vector<full_object_detection>& faces,
    const int num_jitters
)
{
    if (!is_rgb_python_image(img)) {
        throw dlib::error("Unsupported image type, must be RGB image.");
    }

    for (auto& f : faces) {
        if (f.num_parts() != 68) {
            throw dlib::error("The full_object_detection must use the iBUG 300W 68 point face landmark style.");
        }
    }

    std::vector<chip_details> dets;
    for (auto& f : faces) {
        dets.push_back(get_face_chip_details(f, 150, 0.25));
    }
    
    dlib::array<matrix<rgb_pixel>> face_chips;
    extract_image_chips(numpy_rgb_image(img), dets, face_chips);
    
    return _describe_chips(face_chips, num_jitters);
}

// Private
std::vector<matrix<rgb_pixel>> face_recognition_model_v1::jitter_image(
    const matrix<rgb_pixel>& img,
    const int num_jitters
)
{
    std::vector<matrix<rgb_pixel>> crops; 
    
    std::vector<mmod_rect> raw_boxes(1), ignored_crop_boxes;
    raw_boxes[0] = shrink_rect(get_rect(img),3);

    matrix<rgb_pixel> temp; 
    for (int i = 0; i < num_jitters; ++i)
    {
        (*cropper)(img, raw_boxes, temp, ignored_crop_boxes);
        crops.push_back(move(temp));
    }
    return crops;
}

std::vector<matrix<double,0,1>> face_recognition_model_v1::describe_chips (
    boost::python::list pyface_chips,
    const int num_jitters
)
{   
    dlib::array<matrix<rgb_pixel>> face_chips;
    for(auto pyface_chip : python_list_to_vector<object>(pyface_chips))
    {
        matrix<rgb_pixel> face_chip;
        assign_image(face_chip, numpy_rgb_image(pyface_chip));
        face_chips.push_back(face_chip);
    }
    return _describe_chips(face_chips, num_jitters);   
}


std::vector<matrix<double,0,1>> face_recognition_model_v1::_describe_chips (
    const dlib::array<matrix<rgb_pixel>>& face_chips,
    const int num_jitters
)
{
    std::vector<matrix<double,0,1>> face_descriptors;
    face_descriptors.reserve(face_chips.size());

    if (num_jitters <= 1) {
        // extract descriptors and convert from float vectors to double vectors
        for (auto& d : net(face_chips,16)) {
            face_descriptors.push_back(matrix_cast<double>(d));
        }
    } else {
        for (auto& fimg : face_chips) {
            face_descriptors.push_back(matrix_cast<double>(mean(mat(net(jitter_image(fimg,num_jitters),16)))));
        }
    }

    return face_descriptors;
}

// ----------------------------------------------------------------------------------------

void bind_face_recognition()
{
    using boost::python::arg;
    {
    class_<face_recognition_model_v1>("face_recognition_model_v1", "This object maps human faces into 128D vectors where pictures of the same person are mapped near to each other and pictures of different people are mapped far apart.  The constructor loads the face recognition model from a file. The model file is available here: http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2", init<std::string>())
        .def("compute_face_descriptor", &face_recognition_model_v1::compute_face_descriptor, (arg("img"),arg("face"),arg("num_jitters")=0),
            "Takes an image and a full_object_detection that references a face in that image and converts it into a 128D face descriptor. "
            "If num_jitters>1 then each face will be randomly jittered slightly num_jitters times, each run through the 128D projection, and the average used as the face descriptor."
            )
        .def("compute_face_descriptor", &face_recognition_model_v1::compute_face_descriptors, (arg("img"),arg("faces"),arg("num_jitters")=0),
            "Takes an image and an array of full_object_detections that reference faces in that image and converts them into 128D face descriptors.  "
            "If num_jitters>1 then each face will be randomly jittered slightly num_jitters times, each run through the 128D projection, and the average used as the face descriptor."
            )
        .def("compute_batch_face_descriptors", &face_recognition_model_v1::compute_batch_face_descriptors, (arg("imgs"),arg("faces"),arg("num_jitters")=0),
            "batched faces"
        )
        .def("describe_chips", &face_recognition_model_v1::describe_chips, (arg("imgs"),arg("num_jitters")=0),
            "batched faces"
        );

    }

    {
    typedef std::vector<full_object_detection> type;
    class_<type>("full_object_detections", "An array of full_object_detection objects.")
        .def(vector_indexing_suite<type>())
        .def("clear", &type::clear)
        .def("resize", resize<type>)
        .def_pickle(serialize_pickle<type>());
    }
}