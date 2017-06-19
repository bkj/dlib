// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_FACE_RECOGNITION_PY_
#define DLIB_FACE_RECOGNITION_PY_

#include <dlib/python.h>
#include <boost/shared_ptr.hpp>
#include <dlib/matrix.h>
#include <boost/python/slice.hpp>
#include <dlib/geometry/vector.h>
#include <dlib/dnn.h>
#include <dlib/image_transforms.h>
#include "indexing.h"

using namespace dlib;
using namespace std;
using namespace boost::python;

class face_recognition_model_v1
{

public:

    face_recognition_model_v1(const std::string& model_filename);
    
    std::vector<matrix<double,0,1>> compute_batch_face_descriptors (
        boost::python::list,
        boost::python::list,
        const int
    );
    
    std::vector<matrix<double,0,1>> _compute_batch_face_descriptors (
        std::vector<object>,
        const std::vector<std::vector<full_object_detection>>&,
        const int
    );

    matrix<double,0,1> compute_face_descriptor (
        object,
        const full_object_detection&,
        const int
    );

    std::vector<matrix<double,0,1>> compute_face_descriptors (
        object,
        const std::vector<full_object_detection>&,
        const int
    );

    std::vector<matrix<double,0,1>> describe_chips (
        boost::python::list,
        const int
    );

    std::vector<matrix<double,0,1>> _describe_chips (
        const dlib::array<matrix<rgb_pixel>>&,
        const int
    );
    
private:

    std::shared_ptr<random_cropper> cropper;

    std::vector<matrix<rgb_pixel>> jitter_image(
        const matrix<rgb_pixel>&,
        const int
    );

    template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
    using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

    template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
    using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

    template <int N, template <typename> class BN, int stride, typename SUBNET> 
    using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

    template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
    template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

    template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
    template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
    template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
    template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
    template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

    using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                                alevel0<
                                alevel1<
                                alevel2<
                                alevel3<
                                alevel4<
                                max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                                input_rgb_image_sized<150>
                                >>>>>>>>>>>>;
    anet_type net;
};

#endif