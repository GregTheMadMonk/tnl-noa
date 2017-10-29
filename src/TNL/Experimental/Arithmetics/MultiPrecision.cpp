/**************************************************
* filename:		MultiPrecision.cpp	  *
* created:		October 22, 2017	  *
* author:		Daniel Simon	 	  *
* mail:			dansimon93@gmail.com      *
***************************************************/

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

/* ARITHMETIC FUNCTIONS */

MultiPrecision addition( MultiPrecision a, MultiPrecision b ){
    mpf_t result;
    mpf_init(result);
    mpf_add(result, a.number, b.number);
    return result;
}

MultiPrecision subtraction( MultiPrecision a, MultiPrecision b ){
    mpf_t result;
    mpf_init(result);
    mpf_sub(result, a.number, b.number);
    return result;
}

MultiPrecision multiplication( MultiPrecision a, MultiPrecision b ){
    mpf_t result;
    mpf_init(result);
    mpf_mul(result, a.number, b.number);
    return result;
}

MultiPrecision division( MultiPrecision a, MultiPrecision b ){
    mpf_t result;
    mpf_init(result);
    mpf_div(result, a.number, b.number);
    return result;
}

MultiPrecision sqrt( MultiPrecision a ){
    mpf_t result;
    mpf_init(result);
    mpf_sqrt(result, a.number);
    return result;
}

MultiPrecision power( MultiPrecision a, unsigned long int c ){
    mpf_t result;
    mpf_init(result);
    mpf_pow_ui(result, a.number, c);
    return result;
}

MultiPrecision negation( MultiPrecision a ){
    mpf_t result;
    mpf_init(result);
    mpf_neg(result, a.number);
    return result;
}

MultiPrecision abs( MultiPrecision a ){
    mpf_t result;
    mpf_init(result);
    mpf_abs(result, a.number);
    return result;
}

MultiPrecision mul_2exp( MultiPrecision a, mp_bitcnt_t b){
    mpf_t result;
    mpf_init(result);
    mpf_mul_2exp(result, a.number, b);
    return result;
}

MultiPrecision div_2exp( MultiPrecision a, mp_bitcnt_t b ){
    mpf_t result;
    mpf_init(result);
    mpf_div_2exp(result, a.number, b);
    return result;
}


/* OPERATOR OVERLOADING */


/* DESTRUCTOR */

MultiPrecision::~MultiPrecision(){
    mpf_clear(number);
}