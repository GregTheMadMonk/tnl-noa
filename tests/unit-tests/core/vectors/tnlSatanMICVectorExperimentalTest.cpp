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

//#include <png.h>

using namespace std;

//TUNE MACROS FOR YOUR FUNKY OUTPUT
#define SATANVERBOSE

unsigned int errors=0;
unsigned int success=0;
#define SATANTEST(a) if((a)){cout << __LINE__ <<":\t OK" <<endl;success++;}else{cout << __LINE__<<":\t FAIL" <<endl;errors++;}
#define SATANTESTRESULT cout<<"SUCCES: "<<success<<endl<<"ERRRORS: "<<errors<<endl;
inline void SatanSay( const char * message)
{
#ifdef SATANVERBOSE
	cout << message <<endl;
#endif
}





int main(void)
{
    cout << "tnlVector on MIC test by Satan:" <<endl; //LOL
	
	#ifdef HAVE_ICPC
		cout << "ICPC in USE" <<endl; //LOL
	#endif

	#ifdef HAVE_MIC
		cout << "MIC in USE" <<endl; //LOL
	#endif

		tnlArray<double> arr;

#pragma offload target(mic) 
		{
			cout << "Hello " <<endl;
		}

		
    return 0;
}
