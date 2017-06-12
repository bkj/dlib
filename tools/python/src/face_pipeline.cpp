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
#include "face_recognition.h"
#include <boost/python/args.hpp>


using namespace dlib;
using namespace std;
using namespace boost::python;

class face_pipeline_v1 {

    public:

        face_pipeline_v1(const std::string&, const std::string&, const std::string&);
        int run(boost::python::list, int, int);

    private: 
        face_recognition_model_v1 rec;
        shape_predictor sp;
        face_detection_model_v1 det;
};

face_pipeline_v1::face_pipeline_v1(
    const std::string& det_filename,
    const std::string& shape_filename,
    const std::string& face_filename
) : det(det_filename), rec(face_filename) {
    deserialize(shape_filename) >> sp;
}

int face_pipeline_v1::run(
    boost::python::list pyimages,
    const int upsample_num_times,
    const int num_jitters
) {
    return 1; 
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

