/*
 *  QuadDouble.h
 *
 *  Created by Matěj Novotný on 27.10.2010
 *  Modified by Daniel Simon on 28.10.2017
 *
 *  INFO: Quad and Quadcpp merged in this single class.
 */


class QuadDouble
{
public:
    /*INIT*/
    double data[4];
    QuadDouble();
    explicit QuadDouble(const double&);
    explicit QuadDouble(int);
    QuadDouble(const QuadDouble&);

    /*OVERLOADED OPERATORS*/
    double& operator[](int);
    const double& operator[](int) const;
    QuadDouble& operator =(const QuadDouble&);
    QuadDouble& operator +=(const QuadDouble&);
    QuadDouble& operator -=(const QuadDouble&);
    QuadDouble& operator *=(const QuadDouble&);
    QuadDouble& operator /=(const QuadDouble&);
    QuadDouble& operator =(const double&);
    QuadDouble& operator +=(const double&);
    QuadDouble& operator -=(const double&);
    QuadDouble& operator *=(const double&);
    QuadDouble& operator /=(const double&);
    QuadDouble operator +(const QuadDouble&) const;
    QuadDouble operator -(const QuadDouble&) const;
    QuadDouble operator *(const QuadDouble&) const;
    QuadDouble operator /(const QuadDouble&) const;
    QuadDouble operator +(const double&) const;
    QuadDouble operator -(const double&) const;
    QuadDouble operator *(const double&) const;
    QuadDouble operator /(const double&) const;
    QuadDouble operator +();
    QuadDouble operator -();
    QuadDouble operator +() const;
    QuadDouble operator -() const;
    bool operator ==(const QuadDouble&) const;
    bool operator !=(const QuadDouble&) const;
    bool operator <(const QuadDouble&) const;
    bool operator >(const QuadDouble&) const;
    bool operator >=(const QuadDouble&) const;
    bool operator <=(const QuadDouble&) const;
    operator double() const;
};


/*NON MEMBER OPERATORS*/
QuadDouble operator +(const double&, const QuadDouble&);
QuadDouble operator -(const double&, const QuadDouble&);
QuadDouble operator *(const double&, const QuadDouble&);
QuadDouble operator /(const double&, const QuadDouble&);

/*NON MEMBER CLASS MATH FUNCTIONS*/
QuadDouble abs(const QuadDouble&);
QuadDouble sqrt(const QuadDouble&);

/*NON MEMBER HELP FUNCTIONS FOR QUAD*/
void quickTwoSum(double a, double b, double *s, double *e); // Addition of two doubles
void twoSum(double a, double b, double *s, double *e); // Addition of two doubles
void split(double a, double *a_hi, double *a_lo); // Split double into two 26 bits parts
void twoProd(double a, double b, double *p, double *e); // Multiplication of two doubles
void renormalize(double *a, double *b); // Normalization of number a
void doublePlusQuad(double b, const double *a, double *s); // Addition of double and quad-double
void doubleTimesQuad(double b, const double *a, double *s); // Multiplication of double and quad-double
void quadDivDouble(const double *a, double b, double *s); // Division of two doubles
void quadAdd(const double *a, const double *b, double *s); // Addition of two quad-doubles
void quadAddAccurate(const double *a, const double *b, double *s); // Addition of two quad-doubles ! slower algorhitm
void quadMul(const double *a, const double *b, double *s); // Multiplication of two quad-doubles
void quadMulQuick(const double *a, const double *b, double *s); // Multiplication of two quad-doubles ! faster algorithm
void quadDiv(const double *a, const double *b, double *s); // Division of two quad-doubles
void zeroQuad(double *a); // Reset quad-double
void printQuad(double *a); // Print of quad-double
