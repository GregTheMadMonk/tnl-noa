#ifndef _TNLVDBMATH_H_INCLUDED_
#define _TNLVDBMATH_H_INCLUDED_

class tnlVDBMath
{
public:
    static int power( int number,
                      int exponent )
    {
        int result = 1;
        for( int i = 0; i < exponent; i++ )
            result *= number;
        return result;
    }

};

#endif // _TNLVDBMATH_H_INCLUDED_
