/* 
 * File:   Multiple.h
 * Author: oberhuber
 *
 * Created on October 26, 2016, 2:12 PM
 */

#pragma once

#ifdef HAVE_GMP
#include <gmp.h>
#endif


class Multiple
{
   public:
   
      
      
   protected:
   
#ifdef HAVE_GMP
      mpz_t data;
#endif
};

