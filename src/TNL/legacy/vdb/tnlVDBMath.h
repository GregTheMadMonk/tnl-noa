#ifndef _TNLVDBMATH_H_INCLUDED_
#define _TNLVDBMATH_H_INCLUDED_

template< typename Index >
class tnlVDBMath
{
public:
    static Index power( Index number,
                        Index exponent )
    {
        Index result = 1;
        for( Index i = 0; i < exponent; i++ )
            result *= number;
        return result;
    }

};

#endif // _TNLVDBMATH_H_INCLUDED_
