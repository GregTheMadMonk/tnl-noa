#ifndef _TNLVDBMATH_H_INCLUDED_
#define _TNLVDBMATH_H_INCLUDED_

class tnlVDBMath
{

    static int power( int number,
                      int exponent )
    {
        int result = number;
        for( int i = 1; i < exponent; i++ )
            result *= number;
        return result;
    }

};

#endif // _TNLVDBMATH_H_INCLUDED_
