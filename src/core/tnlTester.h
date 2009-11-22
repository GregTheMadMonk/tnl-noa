/***************************************************************************
                          tnlTester.h  -  description
                             -------------------
    begin                : Nov 21, 2009
    copyright            : (C) 2009 by Tomas Oberhuber
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

#ifndef TNLTESTER_H_
#define TNLTESTER_H_

#include <core/tnlString.h>
#include <core/tnlStack.h>

enum tnlTestResult { tnlTestPASS = 0,
                     tnlTestFAIL,
                     tnlTestNOT_IMPLEMENTED };

class tnlTester
{
	protected:

	tnlString current_unit;

	tnlStack< tnlString > tests_stack;

	int tests_passed_counter;

	int tests_failed_counter;

	int tests_not_implemented_counter;

	public:

	//! Constructor with no parameters.
	tnlTester();

	//! Call this before starting the unit tests
	/*! This method cannot be called recursively.
	 *  One must finish testing one unit before
	 *  starting a new one.
	 */
	void StartNewUnit( const char* unit_name );

	//! Call this to finish the current unit testing.
	void FinishUnit();

	//! Call this before starting new test.
	/*! This method can be called recursively
	 *  and so one can organise test into a
	 *  heirarchical structure.
	 */
	void StartNewTest( const char* test_description );

	//! Call this to finish the current test.
	void FinishTest( tnlTestResult test_result );

	int GetTestPASSNumber();

	int GetTestFAILNumber();

	int GetTestNOTIMPLEMENTDNumber();

	//! Print out tests statistics
	void PrintStatistics();
};

#endif /* TNLTESTER_H_ */
