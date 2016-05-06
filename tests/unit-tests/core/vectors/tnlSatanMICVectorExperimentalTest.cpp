/**
 *  Test develop by Satan for implementation of tnlVector on MIC
 * 
 *  1) Edit tnlVectorOperations.h with specialization for MIC (same prototypes of methods )
 *  2) Create tnlVectorOperationsMIC_impl.h (dummy, do nothing or retrun 0)
 *  3) Create test for this (this file)
 * 
 *  4) Lots of Troubles occured.... some problems in file read/write int tnlFile_impl.h (see it for details, soved by memcpy on MIC)
 *									there is no png.h for MIC (okey, but wo dont need it I thing)
 * 
 */


#include <iostream>
#include <omp.h>

#include <core/tnlMIC.h>
//#include <core/tnlFile.h>
#include <core/arrays/tnlArray.h>

#include <png.h>



//TUNE MACROS FOR YOUR FUNKY OUTPUT
#define SATANVERBOSE

#define SATANTEST(a) if((a)){cout << __LINE__ <<":\t OK" <<endl;}else{cout << __LINE__<<":\t FAIL" <<endl;}
inline void SatanSay( const char * message)
{
#ifdef SATANVERBOSE
	cout << message <<endl;
#endif
}

using namespace std;



int main(void)
{
    cout << "tnlVector on MIC test by Satan:" <<endl; //LOL
	
	#ifdef HAVE_ICPC
		cout << "ICPC in USE" <<endl; //LOL
	#endif
/*
 
		tnlFile soubor;
		soubor.open("/home/hanousek/pokus.tnl",tnlReadMode);
		soubor.close();*/
		
		//tnlVector<> vct;
		tnlArray<double> arr;

/*#pragma offload target(mic) 
		{
			cout << "Hello " <<endl;
		}
*/

		//tnlVector<tnlMIC> aa(10);
		
		//cout << aa <<endl;

		
    return 0;
}
