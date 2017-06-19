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
#include <boost/python/args.hpp>
#include <dlib/image_io.h>


using namespace dlib;
using namespace std;
using namespace boost::python;

class face_align {

    public:
        face_align(const std::string&);
        boost::python::list run(object, dlib::rectangle);
        void save(object, dlib::rectangle, std::string);

    private:
        shape_predictor sp;
};

face_align::face_align(
    const std::string& shape_filename
) {
    deserialize(shape_filename) >> sp;
}

boost::python::list face_align::run(
    object pyimage,
    dlib::rectangle rect
) { 
    matrix<rgb_pixel> img;
    assign_image(img, numpy_rgb_image(pyimage));
     
    matrix<rgb_pixel> face_chip;
    auto shape = sp(img, rect);
    chip_details details;
    details = get_face_chip_details(shape, 150, 0.25);
    extract_image_chip(img, details, face_chip);
    
    image_view<matrix<rgb_pixel>> vchip(face_chip);
    std::vector<unsigned char> ret;
    ret.reserve(3 * vchip.nr() * vchip.nc());
    for(long r = 0; r < vchip.nr(); ++r) {
        for(long c = 0; c < vchip.nc(); ++c) {
            ret.push_back(vchip[r][c].red);
            ret.push_back(vchip[r][c].green);
            ret.push_back(vchip[r][c].blue);
        }
    }
    
    return vector_to_python_list(ret);
}

void face_align::save(
    object pyimage,
    dlib::rectangle rect,
    std::string outpath
) {
    matrix<rgb_pixel> img;
    assign_image(img, numpy_rgb_image(pyimage));
     
    matrix<rgb_pixel> face_chip;
    auto shape = sp(img, rect);
    chip_details details;
    details = get_face_chip_details(shape, 150, 0.25);
    extract_image_chip(img, details, face_chip);
    save_jpeg(face_chip, outpath);
}

// ----------------------------------------------------------------------------------------

void bind_face_align()
{
    using boost::python::arg;
    {
        class_<face_align>("face_align", "face align", init<std::string>())
            .def("__call__", &face_align::run, (arg("img"), arg("rect")), "face align (return array)")
            .def("save", &face_align::save, (arg("img"), arg("rect"), arg("outpath")), "face align (write to file)");
    }
}

