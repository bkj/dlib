#include <dlib/python.h>
#include <boost/shared_ptr.hpp>
#include <dlib/matrix.h>
#include <boost/python/slice.hpp>
#include <dlib/geometry/vector.h>
#include <dlib/geometry.h>
#include "indexing.h"
#include <boost/python/args.hpp>

#include <dlib/clustering.h>

using namespace dlib;
using namespace std;
using namespace boost::python;

class pychinese_whispers {

    public:
        pychinese_whispers();
        boost::python::list run(boost::python::list);
};

pychinese_whispers::pychinese_whispers() {}

boost::python::list pychinese_whispers::run(
    boost::python::list pyedges
) { 
    
    boost::python::tuple pyedge;
    std::vector<sample_pair> edges;
    for(int i=0; i < boost::python::len(pyedges); ++i) {
        
        pyedge = extract<boost::python::tuple>(pyedges[i])();
        
        edges.push_back(sample_pair(
            extract<int>(pyedge[0]),
            extract<int>(pyedge[1])
        ));
        
    }
    
    std::vector<unsigned long> labels;
    const auto num_clusters = chinese_whispers(edges, labels);
    return vector_to_python_list(labels);
}


// ----------------------------------------------------------------------------------------

void bind_pychinese_whispers()
{
    using boost::python::arg;
    {
        class_<pychinese_whispers>("chinese_whispers", "chinese whispers clustering", init<>())
            .def("__call__", &pychinese_whispers::run, (arg("edges")), "chinese whispers clustering");
    }
}

