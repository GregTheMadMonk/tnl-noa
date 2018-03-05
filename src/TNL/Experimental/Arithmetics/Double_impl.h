/**************************************************
* filename:		Double_impl.h	          *
* created:		December 6, 2017	  *
* author:		Daniel Simon	 	  *
* mail:			dansimon93@gmail.com      *
***************************************************/

#include <cmath>
#include <cstdio>

#include "Double.h"

namespace TNL {
namespace Arithmetics {
    
template <class T>
Double<T>::Double() {
    zeroDouble(data);
}

template <class T>
Double<T>::Double(const T& value) {
    data[0] = value;
    data[1] = 0;
}

template <class T>
Double<T>::Double(int value) {
    data[0] = (T)value;
    data[1] = 0;
}

template <class T>
Double<T>::Double(const Double<T>& other) {
    data[0] = other[0];
    data[1] = other[1];
}

template <class T>
Double<T>& Double<T>::operator =(const Double<T>& rhs) {
    data[0] = rhs[0];
    data[1] = rhs[1];
    return *this;
}

template <class T>
Double<T> Double<T>::operator +(const Double<T>& rhs) const{
    Double<T> lhs(*this);
    lhs += rhs;
    return qd;
}

template <class T>
Double<T> Double<T>::operator -(const Double<T>& rhs) const{
    Double<T> lhs(*this);
    lhs += rhs;
    return qd;
}

template <class T>
Double<T> Double<T>::operator *(const Double<T>& rhs) const{
    Double<T> lhs(*this);
    lhs *= rhs;
    return qd;
}

template <class T>
Double<T> Double<T>::operator /(const Double<T>& rhs) const{
    Double<T> lhs(*this);
    lhs /= rhs;
    return qd;
}

/*
 TODO COMPARISON OPERATORS
 */

} // namespace Arithmetics
} // namespace TNL