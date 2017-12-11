/**************************************************
* filename:		Double.h	          *
* created:		December 6, 2017	  *
* author:		Daniel Simon	 	  *
* mail:			dansimon93@gmail.com      *
***************************************************/

namespace TNL {
    
template <class T>
class Double
{
public:
    T data[2];
    
    Double();
    Double(const Double<T>& value);
    explicit Double(const T& value);
    explicit Double(int value);
    
    Double<T>& operator=(const Double<T>& rhs);
    Double<T>& operator-();
    Double<T>& operator+();
    Double<T>& operator+=(const Double<T>& rhs);
    Double<T>& operator-=(const Double<T>& rhs);
    Double<T>& operator*=(const Double<T>& rhs);
    Double<T>& operator/=(const Double<T>& rhs);
    Double<T> operator+(const Double<T>& rhs) const;
    Double<T> operator-(const Double<T>& rhs) const;
    Double<T> operator*(const Double<T>& rhs) const;
    Double<T> operator/(const Double<T>& rhs) const;
    bool operator==(const Double<T>& rhs) const;
    bool operator!=(const Double<T>& rhs) const;
    bool operator<(const Double<T>& rhs) const;
    bool operator>(const Double<T>& rhs) const;
    bool operator>=(const Double<T>& rhs) const;
    bool operator<=(const Double<T>& rhs) const;
    explicit operator T() const;
};

/// TODO
template <typename T>
void zeroDouble(T *d);
template <typename T>
void twoSum(T a, T b, T *s, T *e);
template <typename T>
void quickTwoSum(T a, T b, T *s, T *e);
template <typename T>
void split(T a, T *a_hi, T *a_lo);
template <typename T>
void twoProduct(T a, T b, T *p, T *e);
template <typename T>
void renormalize(T *a, T *b);
template <typename T>
void doubleAdd(const T *a, const T *b, T *s);
template <typename T>
void doubleMul(const T *a, const T *b, T *s);
template <typename T>
void printDouble(T *d);

} // namespace TNL