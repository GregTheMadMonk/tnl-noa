/**************************************************
* filename:		MultiPrecision.cpp	  *
* created:		November 11, 2017	  *
* author:		Daniel Simon	 	  *
* mail:			dansimon93@gmail.com      *
***************************************************/

#ifdef HAVE_GMP

#include "MultiPrecision.h"

/* INIT OF NUMBER */

MultiPrecision::MultiPrecision(mpf_t number){
    mpf_init(number);
}

MultiPrecision::MultiPrecision(mpf_t number, mp_bitcnt_t precision){
    mpf_init2(number, precision);
}

MultiPrecision::MultiPrecision(mp_bitcnt_t precision){
    mpf_set_default_prec(precision);
}

MultiPrecision::MultiPrecision(mpf_t number,  mpf_t n){
    mpf_init(n);
    mpf_init_set(number, n);
}

/* OPERATORS IMPLEMENTATION */

void MultiPrecision::operator=(const MultiPrecision& mp){
    mpf_set(number, mp.number);
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
    MultiPrecision result = MultiPrecision(*this);
    result -= mp;
    return result;
}

MultiPrecision MultiPrecision::operator*(const MultiPrecision& mp) const{
    MultiPrecision result = MultiPrecision(*this);
    result *= mp;
    return result;
}

MultiPrecision MultiPrecision::operator/(const MultiPrecision& mp) const{
    MultiPrecision result = MultiPrecision(*this);
    result /= mp;
    return result;
}

bool MultiPrecision::operator==(const MultiPrecision &mp) const{
    MultiPrecision m = MultiPrecision(*this);
    if (mpf_cmp(m.number, mp.number) == 0)
        return true;
    else
        return false;
}

bool MultiPrecision::operator!=(const MultiPrecision &mp) const{
    //MultiPrecision m = MultiPrecision(*this);
    //return !(m.number == mp.number);
    return !(*this == mp);
}

bool MultiPrecision::operator<(const MultiPrecision &mp) const{
    MultiPrecision m = MultiPrecision(*this);
    if (mpf_cmp(m.number, mp.number) < 0)
        return true;
    else
        return false;
}

bool MultiPrecision::operator>(const MultiPrecision &mp) const{
    MultiPrecision m = MultiPrecision(*this);
    if (mpf_cmp(m.number, mp.number) > 0)
        return true;
    else
        return false;
}

bool MultiPrecision::operator>=(const MultiPrecision &mp) const{
    MultiPrecision m = MultiPrecision(*this);
    if (mpf_cmp(m.number, mp.number) >= 0)
        return true;
    else
        return false;
}

bool MultiPrecision::operator<=(const MultiPrecision &mp) const{
    MultiPrecision m = MultiPrecision(*this);
    if (mpf_cmp(m.number, mp.number) <= 0)
        return true;
    else
        return false;
}

/* METHODS */
void MultiPrecision::printMP(int precision){
    mpf_out_str(stdout, 10, precision, this->number);
}

MultiPrecision MultiPrecision::abs(MultiPrecision r, const MultiPrecision a){
    mpf_abs(r.number, a.number);
    return r;
}

MultiPrecision MultiPrecision::sqrt(MultiPrecision r, const MultiPrecision a){
    mpf_sqrt(r.number, a.number);
    return r;
}


/* DESTRUCTOR */

MultiPrecision::~MultiPrecision(){
    mpf_clear(number);
}

#endif