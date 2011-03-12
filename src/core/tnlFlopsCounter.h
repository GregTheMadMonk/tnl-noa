/***************************************************************************
                          tnlFlopsCountar.h
                             -------------------
    begin                : Jun 14, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/



#ifndef TNLFLOPSCOUNTER_H_
#define TNLFLOPSCOUNTER_H_


class tnlFlopsCounter
{
	bool compute_flops;

	long int adding_flops;

	long int multiplying_flops;

	long int dividing_flops;

	long int functions_flops;

	public:

	tnlFlopsCounter()
	: compute_flops( true ),
      adding_flops( 0 ),
      multiplying_flops( 0 ),
      dividing_flops( 0 )
	{};

	void recordAdding( const int ops = 1 )
	{
	   if( compute_flops ) adding_flops += ops;
	};

	void recordMultiplying( const int ops = 1 )
	{
		if( compute_flops ) multiplying_flops += ops;
	};

	void recordDividing( const int ops = 1 )
	{
		if( compute_flops ) dividing_flops += ops;
	};

       void recordFunction( const int ops = 1 )
       {
          if( compute_flops ) functions_flops += ops;
       };


	void turnFlopsCountingOn()
	{
		compute_flops = true;
	};

	void turnFlopsCountingOff()
	{
		compute_flops = false;
	};

	long int getFlops()
	{
		return adding_flops + multiplying_flops + dividing_flops + functions_flops;
	};

	void resetFlops()
	{
		adding_flops = 0;
		multiplying_flops = 0;
	   dividing_flops = 0;
	   functions_flops = 0;
	};
};

extern tnlFlopsCounter tnl_flops_counter;

#endif /* TNLFLOPSCOUNTER_H_ */
