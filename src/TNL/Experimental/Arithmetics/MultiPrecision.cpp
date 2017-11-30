/**************************************************
* filename:		MultiPrecision.cpp	  *
* created:		November 11, 2017	  *
* author:		Daniel Simon	 	  *
* mail:			dansimon93@gmail.com      *
***************************************************/

#ifdef HAVE_GMP

#include "MultiPrecision.h"

/* CONSTRUCTORS */

MultiPrecision::MultiPrecision(){
    mpf_init(number);
}

MultiPrecision::MultiPrecision(int precision) {
    mpf_set_default_prec(precision);
}

MultiPrecision::MultiPrecision(double d){
    mpf_init_set_d(number, d);
}

/* OPERATORS IMPLEMENTATION */

MultiPrecision& MultiPrecision::operator=(const MultiPrecision& mp){
    mpf_set(number, mp.number);
    return *this;
}

MultiPrecision& MultiPrecision::operator-(){
    mpf_neg(this->number, this->number);
    return *this;
}

MultiPrecision& MultiPrecision::operator+=(const MultiPrecision& mp){
    mpf_add(this->number, this->number, mp.number);
    return *this;
}

MultiPrecision& MultiPrecision::operator-=(const MultiPrecision& mp){
    mpf_sub(this->number, this->number, mp.number);
    return *this;
}

MultiPrecision& MultiPrecision::operator*=(const MultiPrecision& mp){
    mpf_mul(this->number, this->number, mp.number);
    return *this;
}

MultiPrecision& MultiPrecision::operator/=(const MultiPrecision& mp){
    mpf_div(this->number, this->number, mp.number);
    return *this;
}

MultiPrecision MultiPrecision::operator+(const MultiPrecision& mp) const{
    MultiPrecision result = MultiPrecision(*this);
    result += mp;
    return result;
}

MultiPrecision MultiPrecision::operator-(const MultiPrecision& mp) const{
    MultiPrecision result (*this);
    result -= mp;
    return result;
}

MultiPrecision MultiPrecision::operator*(const MultiPrecision& mp) const{
    MultiPrecision result (*this);
    result *= mp;
    return result;
}

MultiPrecision MultiPrecision::operator/(const MultiPrecision& mp) const{
    MultiPrecision result (*this);
    result /= mp;
    return result;
}

bool MultiPrecision::operator==(const MultiPrecision &mp) const{
    MultiPrecision m (*this);
    if (mpf_cmp(m.number, mp.number) == 0)
        return true;
    else
        return false;
}

bool MultiPrecision::operator!=(const MultiPrecision &mp) const{
    return !(*this == mp);
}

bool MultiPrecision::operator<(const MultiPrecision &mp) const{
    MultiPrecision m (*this);
    if (mpf_cmp(m.number, mp.number) < 0)
        return true;
    else
        return false;
}

bool MultiPrecision::operator>(const MultiPrecision &mp) const{
    MultiPrecision m (*this);
    if (mpf_cmp(m.number, mp.number) > 0)
        return true;
    else
        return false;
}

bool MultiPrecision::operator>=(const MultiPrecision &mp) const{
    MultiPrecision m (*this);
    if (mpf_cmp(m.number, mp.number) >= 0)
        return true;
    else
        return false;
}

bool MultiPrecision::operator<=(const MultiPrecision &mp) const{
    MultiPrecision m (*this);
    if (mpf_cmp(m.number, mp.number) <= 0)
        return true;
    else
        return false;
}

/* METHODS */

void MultiPrecision::printMP(){
    int precision = mpf_get_default_prec();
    mpf_out_str(stdout, 10, precision, this->number); std::cout <<std::endl;
}

/* DESTRUCTOR */

MultiPrecision::~MultiPrecision(){
}

#endif