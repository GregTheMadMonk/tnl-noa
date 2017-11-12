/**************************************************
* filename:             MultiPrecision.h          *
* created:              November 11, 2017          *
* author:               Daniel Simon              *
* mail:                 dansimon93@gmail.com      *
***************************************************/

/*IMPLEMENTATION OF GMP LIBRARY - FLOATING POINT FUNCTIONS*/
/* Source: https://gmplib.org/ */

#ifdef HAVE_GMP

#include <gmp.h>

#endif

class MultiPrecision{
public:
    /* NUMBER */
    mpf_t number;

    /* INITIALIZATION OF NUMBER */
    MultiPrecision(mpf_t number); // inits number to 0
    MultiPrecision(mpf_t number, mp_bitcnt_t precision); // inits number to 0 with bit precision
    MultiPrecision(mp_bitcnt_t precision); // sets the default precision
    MultiPrecision(mpf_t number,  mpf_t n); // assigns n value to number

    /* OPERATORS */
    void operator=(const MultiPrecision& mp);
    MultiPrecision& operator-();
    MultiPrecision& operator+=(const MultiPrecision& mp);
    MultiPrecision& operator-=(const MultiPrecision& mp);
    MultiPrecision& operator*=(const MultiPrecision& mp);
    MultiPrecision& operator/=(const MultiPrecision& mp);
    MultiPrecision operator+(const MultiPrecision& mp) const;
    MultiPrecision operator-(const MultiPrecision& mp) const;
    MultiPrecision operator*(const MultiPrecision& mp) const;
    MultiPrecision operator/(const MultiPrecision& mp) const;
    bool operator==(const MultiPrecision &mp) const;
    bool operator!=(const MultiPrecision &mp) const;
    bool operator<(const MultiPrecision &mp) const;
    bool operator>(const MultiPrecision &mp) const;
    bool operator>=(const MultiPrecision &mp) const;
    bool operator<=(const MultiPrecision &mp) const;
    MultiPrecision& operator++(); // prefix
    MultiPrecision& operator--(); // prefix
    MultiPrecision operator++(int); // postfix
    MultiPrecision operator--(int); // postfix

    /* METHODS */
    void printMP(int precision);
    MultiPrecision abs(MultiPrecision r, const MultiPrecision a);
    MultiPrecision sqrt(MultiPrecision r, const MultiPrecision a);
    /// void printNumber(int digits, ostream& str = std::cout );  TODO

    /* DESTRUCTOR */
    ~MultiPrecision();
};

/// ostream& operator << ( ostream& str, const MultiPrecision& p ); TODO
