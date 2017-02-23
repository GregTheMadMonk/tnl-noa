/**
 *  Test develop by Satan for implementation of Vector on MIC
 * 
 *  1) Edit VectorOperations.h with specialization for MIC (same prototypes of methods )
 *  2) Create VectorOperationsMIC_impl.h (dummy, do nothing or retrun 0)
 *  3) Create test for this (this file)
 * 
 *  4) Lots of Troubles occured.... some problems in file read/write int tnlFile_impl.h (see it for details, soved by memcpy on MIC)
 *									there is no png.h for MIC (okey, but wo dont need it I thing)
 * 
 */


#include <iostream>
#include <omp.h>
#include <stdint.h>

#include <TNL/Devices/MIC.h>
#include <TNL/Containers/Vector.h>

//#include <png.h>

using namespace std;
using namespace TNL;
using namespace TNL::Containers;

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

/*class trida
{
public:
	int data[5];
	
public:
	trida()
	{
		cout << "konstrukor" <<endl;
	};
	
	~trida()
	{
		cout << "destruktor" <<endl;
	};
	
	__device_callable__ void fce(void )
	{
		cout  << "fce" <<endl;
	}
	
};*/




int main(void)
{
    cout << "Vector on MIC test by Satan:" <<endl; //LOL
	
	#ifdef HAVE_ICPC
		cout << "ICPC in USE" <<endl; //LOL
	#endif

	#ifdef HAVE_MIC
		cout << "MIC in USE" <<endl; //LOL
	#endif

#ifdef HAVE_MIC
/*		trida a;
		a.data[0]=0;
		a.data[1]=1;
		a.data[2]=2;
		a.data[3]=3;
		a.data[4]=4;
		
		TNLMICSTRUCT(a,trida);
		
		//satanstruct<sizeof(trida)> sa; 
		//trida sa;
        //memcpy((void*)& sa,(void*)& a, sizeof(trida));
		
		cout << "sizeof a:"<< sizeof(trida) <<endl;
		cout << "sizeof sa:" << sizeof(satanstruct<sizeof(trida)>) <<endl;
		
		//trida * kernela = (trida*) &sa;

		TNLMICSTRUCTUSE(a,trida);
		
		kernela->fce();
		for(int i=0;i<5;i++)
				printf("%d\n", kernela->data[i]);
		
#pragma offload target(mic) in( TNLMICSTRUCTOFF(a,trida) ) 
{
			cout << "MIC sizeof a:"<< sizeof(trida) <<endl;
			cout << "MIC sizeof sa:" <<sizeof(satanstruct<sizeof(trida)>) <<endl;
			
			TNLMICSTRUCTUSE(a,trida);
			kernela->fce();
			for(int i=0;i<5;i++)
				printf("%d\n", kernela->data[i]);
}*/
		Vector<double,Devices::MIC,int> aa(10);
		Vector<double,Devices::MIC,int> bb(10);
		Vector<double,Devices::MIC,int> cc(10);
		
		Vector<double,Devices::Host,int> aaa(10);
		Vector<double,Devices::Host,int> bbb(10);
		Vector<double,Devices::Host,int> ccc(10);

		for(int i=0;i<10;i++)
		{
			aa.setElement(i,i-5);
			aaa.setElement(i,i-5);
			bb.setElement(i,5-i);
			bbb.setElement(i,5-i);
			cc.setElement(i,10+i);
			ccc.setElement(i,10+i);
			
		}
		
		cout << aa <<endl <<aaa<< endl <<bb <<endl <<bbb<<endl;
		
		cout << bb.min()<< " " <<bb.max() <<endl;
		cout << bb.absMin()<< " " <<bb.absMax() <<endl;
		
		cout << aa.lpNorm(1) << " "<<  aa.lpNorm(2) <<" " << aa.lpNorm(0.5)<<" " << aa.lpNorm(3)<<endl;
		cout << aaa.lpNorm(1) << " "<<  aaa.lpNorm(2) << " "<< aaa.lpNorm(0.5) << " "<< aaa.lpNorm(3)<<endl;
		cout << aa.sum() << " "<< aaa.sum()<<endl;
		
		////
		cout << aa.differenceMax(bb) << " diffMax " << aaa.differenceMax(bbb) << endl;
		cout << aa.differenceMin(bb) << " diffMin " << aaa.differenceMin(bbb) << endl;
		cout << aa.differenceAbsMax(bb) << " diffAbsMax " << aaa.differenceAbsMax(bbb) << endl;
		cout << aa.differenceAbsMin(bb) << " diffAbsMin " << aaa.differenceAbsMin(bbb) << endl;
		cout << aa.differenceSum(bb) << " diffSum " << aaa.differenceSum(bbb) << endl;
		////
		cout << aa.differenceLpNorm(bb,1)<<" "<<  aa.differenceLpNorm(bb,2)<<" "<<  aa.differenceLpNorm(bb,0.5)<<" "<<  aa.differenceLpNorm(bb,3.0) <<endl;
		cout << aaa.differenceLpNorm(bbb,1)<<" "<<  aaa.differenceLpNorm(bbb,2)<<" "<<  aaa.differenceLpNorm(bbb,0.5)<<" "<<  aaa.differenceLpNorm(bbb,3.0) <<endl;
		////
		aa*=0.5;
		aaa*=0.5;
		cout << aa <<endl << aaa <<endl;
		cout << aa.scalarProduct(bb) << " scalarProduct " << aaa.scalarProduct(bbb) <<endl;
		
		aa.addVector(bb,2.0,3.0);
		aaa.addVector(bbb,2.0,3.0);
		cout << aa <<endl << aaa <<endl <<endl;
		
		aa.addVectors(bb,2.0,cc,1.0,-3.0);
		aaa.addVectors(bbb,2.0,ccc,1.0,-3.0);
		cout << aa <<endl << aaa <<endl;	
		
		aa.computeExclusivePrefixSum();
		aaa.computeExclusivePrefixSum();
		cout << aa <<endl << aaa <<endl;
		
		bb.computeExclusivePrefixSum(2,4);
		bbb.computeExclusivePrefixSum(2,4);
		cout << bb <<endl << bbb <<endl;
		
		cc.computePrefixSum();
		ccc.computePrefixSum();
		cout << cc <<endl << ccc <<endl;
		
		cc.computePrefixSum(2,4);
		ccc.computePrefixSum(2,4);
		cout << cc <<endl << ccc <<endl;
		
#endif
		
		
    return 0;
}
