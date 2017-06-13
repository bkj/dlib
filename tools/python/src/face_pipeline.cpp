// License: Boost Software License   See LICENSE.txt for the full license.

// !! Doesn't appear to end up being much faster than doing
// everything separately.  Plus is more restrictive (eg size of images).

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
#include "face_recognition.h"
#include <boost/python/args.hpp>


using namespace dlib;
using namespace std;
using namespace boost::python;

class face_pipeline_v1 {

    public:
        face_pipeline_v1(const std::string&, const std::string&, const std::string&);
        boost::python::list run(boost::python::list, int, int);
        std::vector<boost::python::tuple> _run(std::vector<object>, int, int);

    private: 
        
        // Detect
        template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
        template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;

        template <typename SUBNET> using downsampler  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;
        template <typename SUBNET> using rcon5  = relu<affine<con5<45,SUBNET>>>;

        using det_net_type = loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;
        
        det_net_type det;
        
        // Align
        shape_predictor sp;
        
        // Featurize
        face_recognition_model_v1 rec;
};

face_pipeline_v1::face_pipeline_v1(
    const std::string& det_filename,
    const std::string& shape_filename,
    const std::string& face_filename
) : rec(face_filename) {
    deserialize(det_filename) >> det;
    deserialize(shape_filename) >> sp;
}

boost::python::list face_pipeline_v1::run(
    boost::python::list pyimages,
    const int upsample_num_times,
    const int num_jitters
) {
    return vector_to_python_list(_run(python_list_to_vector<object>(pyimages), upsample_num_times, num_jitters));
}

std::vector<boost::python::tuple> face_pipeline_v1::_run(
    std::vector<object> pyimages,
    const int upsample_num_times,
    const int num_jitters
) {
    
    // Convert all images
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
    
    // Detect faces in all images
    std::vector<std::vector<dlib::mmod_rect>> all_face_detections = det(imgs);
    
    std::vector<int> all_inds;
    std::vector<dlib::rectangle> all_rects;
    dlib::array<matrix<rgb_pixel>> all_face_chips;
    for (int i = 0; i < imgs.size(); ++i) {
        
        // Get data for each detection
        std::vector<chip_details> cds;
        for(auto& face_detection : all_face_detections[i]) {
            
            // Image index
            all_inds.push_back(i);
            
            // Bounding box
            all_rects.push_back(face_detection.rect);
            
            // Chip details
            auto shape = sp(imgs[i], face_detection);
            cds.push_back(get_face_chip_details(shape, 150, 0.25));
        }
        
        // Extract face chips from image
        dlib::array<matrix<rgb_pixel>> face_chips;
        extract_image_chips(numpy_rgb_image(pyimages[i]), cds, face_chips);

        // Add to all face chips
        for (auto& face_chip : face_chips) {
            all_face_chips.push_back(face_chip);
        }
    }
    
    // Run model 
    std::vector<matrix<double,0,1>> all_feats = rec.describe_chips(all_face_chips, num_jitters);
    
    std::vector<boost::python::tuple> output;
    for(int i = 0; i < imgs.size(); ++i) {
        output.push_back(boost::python::make_tuple(all_inds[i], all_rects[i], all_feats[i]));
    }
    return output;
}



// ----------------------------------------------------------------------------------------

void bind_face_pipeline()
{
    using boost::python::arg;
    {
    class_<face_pipeline_v1>("face_pipeline_v1", "face pipeline", init<std::string, std::string, std::string>())
        .def("__call__", &face_pipeline_v1::run, (arg("img"), arg("upsample_num_times")=0, arg("num_jitters")=0), "face pipeline");
    }
}

