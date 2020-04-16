#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
namespace py = pybind11;

#include "../tnl_indexing.h"

#include <TNL/Containers/Array.h>


// pybind11 should actually take care of this inside py::format_descriptor, but apparently it does not work...
// see https://github.com/pybind/pybind11/issues/2135
template< typename T, typename = void >
struct underlying_type
{
   using type = T;
};
template< typename T >
struct underlying_type< T, std::enable_if_t< std::is_enum< T >::value > >
{
   using type = std::underlying_type_t< T >;
};


template< typename ArrayType >
void export_Array(py::module & m, const char* name)
{
//    auto array = py::class_<ArrayType, TNL::Object>(m, name, py::buffer_protocol())
    auto array = py::class_<ArrayType>(m, name, py::buffer_protocol())
        .def(py::init<>())
        .def(py::init<int>())
        .def_static("getSerializationType", &ArrayType::getSerializationType)
        .def("getSerializationTypeVirtual", &ArrayType::getSerializationTypeVirtual)
        .def("setSize", &ArrayType::setSize)
        .def("setLike", &ArrayType::template setLike<ArrayType>)
        .def("swap", &ArrayType::swap)
        .def("reset", &ArrayType::reset)
        .def("getSize", &ArrayType::getSize)
        .def("setElement", &ArrayType::setElement)
        .def("getElement", &ArrayType::getElement)
        // operator=
        .def("assign", []( ArrayType& array, const ArrayType& other ) -> ArrayType& {
                return array = other;
            })
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("setValue", &ArrayType::setValue, py::arg("v"), py::arg("begin") = 0, py::arg("end") = 0)
        // TODO: better structuring
        .def("save", &ArrayType::save)
        .def("load", &ArrayType::load)

        .def("__str__", []( ArrayType & a ) {
                std::stringstream ss;
                ss << a;
                return ss.str();
            } )

        // Python buffer protocol: http://pybind11.readthedocs.io/en/master/advanced/pycpp/numpy.html
        .def_buffer( [](ArrayType & a) -> py::buffer_info {
            return py::buffer_info(
                // Pointer to buffer
                a.getData(),
                // Size of one scalar
                sizeof( typename ArrayType::ValueType ),
                // Python struct-style format descriptor
                py::format_descriptor<
                     typename underlying_type< typename ArrayType::ValueType >::type
                >::format(),
                // Number of dimensions
                1,
                // Buffer dimensions
                { a.getSize() },
                // Strides (in bytes) for each index
                { sizeof( typename ArrayType::ValueType ) }
            );
        })
    ;

    tnl_indexing< ArrayType >( array );
    tnl_slice_indexing< ArrayType >( array );
}
