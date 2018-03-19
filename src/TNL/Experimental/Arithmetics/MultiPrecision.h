/**************************************************
* filename:             MultiPrecision.h          *
* created:              November 11, 2017         *
* author:               Daniel Simon              *
* mail:                 dansimon93@gmail.com      *
***************************************************/

/*IMPLEMENTATION OF GMP LIBRARY - FLOATING POINT FUNCTIONS*/
/* Source: https://gmplib.org/ */

#ifdef HAVE_GMP
#include <gmp.h>
#endif

namespace TNL {
namespace Arithmetics {
    
class MultiPrecision{
public:
    /* NUMBER */
    mpf_t number;

    /* CONSTRUCTORS */
    MultiPrecision(); // initialize number to 0
    explicit MultiPrecision(int); // assignment of signed long integer
    explicit MultiPrecision(double d); // assignment of double

    /* OPERATORS */
    MultiPrecision& operator=(const MultiPrecision& mp);
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
    MultiPrecision& operator++();
    MultiPrecision& operator--();
    MultiPrecision operator++(int);
    MultiPrecision operator--(int);

    /* OPERATORS FOR GOOGLE TEST*/
    bool operator==(const mpf_t &GMPnumber) const;
    
    /* METHODS */
    void printMP();
    static MultiPrecision setPrecision(int); // sets the default precision
    /// TODO void printNumber(int digits, ostream& str = std::cout );

    /* DESTRUCTOR */
    ~MultiPrecision();
};

MultiPrecision abs(const MultiPrecision);
MultiPrecision sqrt(const MultiPrecision);
MultiPrecision cqrt(const MultiPrecision);

} // namespace Arithmetics
} // namespace TNL
