/**************************************************             
* filename:             MultiPrecision.h          *
* created:              October 29, 2017          *
* author:               Daniel Simon              *
* mail:                 dansimon93@gmail.com      *
***************************************************/

/*IMPLEMENTATION OF GMP LIBRARY - FLOATING POINT FUNCTIONS*/
/* Source: https://gmplib.org/ */

#include<gmp.h>

class MultiPrecision{
public:
    mpf_t number; // number
    
    MultiPrecision(mpf_t n); // inits value to 0
    
    /* ARITHMETIC FUNCTIONS */
    MultiPrecision addition( MultiPrecision r, MultiPrecision a, MultiPrecision b ); // r = a + b
    MultiPrecision subtraction( MultiPrecision r, MultiPrecision a, MultiPrecision b ); // r = a - b
    MultiPrecision multiplication( MultiPrecision r, MultiPrecision a, MultiPrecision b ); // r = a * b
    MultiPrecision division( MultiPrecision r, MultiPrecision a, MultiPrecision b ); // r = a / b
    MultiPrecision sqrt( MultiPrecision r, MultiPrecision a ); // r = sqrt(a)  
    MultiPrecision power( MultiPrecision r, MultiPrecision b, unsigned long int c ); // r = b ** c
    MultiPrecision negation( MultiPrecision r, MultiPrecision b); // r = -b
    MultiPrecision abs( MultiPrecision r, MultiPrecision a ); // r = |a|
    
    void freeMultiPrecision(); // free memory
    ~MultiPrecision(); // destructor
};

