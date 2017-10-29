/**************************************************             
* filename:             MultiPrecision.h          *
* created:              October 29, 2017          *
* author:               Daniel Simon              *
* mail:                 dansimon93@gmail.com      *
***************************************************/

/*IMPLEMENTATION OF GMP LIBRARY - FLOATING POINT FUNCTIONS*/
/* Source: https://gmplib.org/ */

#include <gmp.h>

class MultiPrecision{
public:
    mpf_t number; // number

    /* INITIALIZATION OF NUMBER */
    MultiPrecision(mpf_t number); // inits number to 0
    MultiPrecision(mpf_t number, mp_bitcnt_t precision); // inits number to 0 with bit precision
    MultiPrecision(mp_bitcnt_t precision); // sets the default precision
    MultiPrecision(mpf_t number,  mpf_t n); // assigns n value to number

    /* ARITHMETIC FUNCTIONS */
    MultiPrecision addition( MultiPrecision a, MultiPrecision b ); // result = a + b
    MultiPrecision subtraction( MultiPrecision a, MultiPrecision b ); // result = a - b
    MultiPrecision multiplication( MultiPrecision a, MultiPrecision b ); // result = a * b
    MultiPrecision division( MultiPrecision a, MultiPrecision b ); // result = a / b
    MultiPrecision sqrt( MultiPrecision a ); // result = sqrt(a)
    MultiPrecision power( MultiPrecision a, unsigned long int c ); // result = b ** c
    MultiPrecision negation( MultiPrecision a ); // result = -b
    MultiPrecision abs( MultiPrecision a ); // result = |a|
    MultiPrecision mul_2exp( MultiPrecision a, mp_bitcnt_t b );
    MultiPrecision div_2exp( MultiPrecision a, mp_bitcnt_t b );

    /* OPERATORS OVERLOADING */

    ~MultiPrecision(); // destructor
};
