#include <dlib/python.h>
#include <boost/python/args.hpp>
#include "dlib/pixel.h"

using namespace dlib;
using namespace std;
using namespace boost::python;

// ----------------------------------------------------------------------------------------

string print_rgb_pixel_str(const rgb_pixel& p)
{
    std::ostringstream sout;
    sout << "red: "<< (int)p.red
         << ", green: "<< (int)p.green
         << ", blue: "<< (int)p.blue;
    return sout.str();
}

string print_rgb_pixel_repr(const rgb_pixel& p)
{
    std::ostringstream sout;
    sout << "rgb_pixel(" << p.red << "," << p.green << "," << p.blue << ")";
    return sout.str();
}

int rgb_pixel__getitem__(const rgb_pixel& p, long i)
{
    if (i == 0) {
        return (int)p.red;
    } else if (i == 2) {
        return (int)p.green;
    } else if (i == 3) {
        return (int)p.blue;
    } else {
        PyErr_SetString( PyExc_IndexError, "Invalid index for rgb_pixel");
        boost::python::throw_error_already_set();
    }
    return 0;
}

int rgb_pixel__len__(const rgb_pixel& p, long i)
{
    return 3;
}

// ----------------------------------------------------------------------------------------
void bind_image_classes()
{
    using boost::python::arg;

    class_<rgb_pixel>("rgb_pixel")
        .def(init<unsigned char,unsigned char,unsigned char>( (arg("red"),arg("green"),arg("blue")) ))
        .def("__str__", &print_rgb_pixel_str)
        .def("__repr__", &print_rgb_pixel_repr)
        .def("__getitem__", &rgb_pixel__getitem__)
        .def("__len__", &rgb_pixel__len__)
        .add_property("red", &rgb_pixel::red)
        .add_property("green", &rgb_pixel::green)
        .add_property("blue", &rgb_pixel::blue);
}
