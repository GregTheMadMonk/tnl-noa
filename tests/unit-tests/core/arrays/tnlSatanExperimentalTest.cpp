/**
 *  Experimentalní test pro získání zkušeností s TNL a (a strarým dobrým MIC, intel offloadingem a podobně)
 * 
 *  1) Přidat tento Hello world Test do projektu, přeložit a spustit
 *  2) Write SATANTEST makro for DUMMY testing
 *  3) Hello from MIC (number of threads)
 *  
 *  4) Transfer DATA to MIC from MIC.... mno...
 *  5) Persistant alloc ...... řeší to nocopy, veškeré alokace musí být řešeny "takto"
 *  6) shared array... no snadne reseni nevidim, omocime to 
 *  7) MIC_callable
 * 
 *  8) Implement Basic MIC in TNL
 *  9) tnlMIC.h -- very basic things
 *  10) tnlArrayMIC_impl.h --> 
 *                         -->a) search for how to hack Pointer translation (hacked by AUTOMAGIC acces to class memberers)
 *						   -->b) search for how to hack moving of non-bitwise copyable object to FUCKING device
 * 
 *						   -->c) waiting for other problems... 
 *								(It may be usefull to have CUDA version for experiments, how TNL works on coprocessor)
 *  11) tnlArrayMIC_impl.h --> support most functions, expect saving and loading.
 *  12) tnlArrayMIC_impl.h --> probably complete.
 * 
 *  13) Solving problems with uncompilable library coused by #include <png.h> in tnlview. Define macro #HAVE_MIC 
 *        and suurrond everithing MIC shit with IT.
 * 
 *  14) So reuse thist test once more.  
 * 
 */

#include <iostream>
#include <core/arrays/tnlArray.h>
#include <omp.h>

#include <core/tnlMIC.h>

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

using namespace std;

/*
 * ICPC offload překládá out of box, pro vypnutí přidat parametr -offload=none
 * 
 * Variables on MIC are maped to variables on CPU (hash table: hash is base adress on CPU, then contain MIC Address and length on MIC)
 * https://software.intel.com/en-us/blogs/2013/03/27/behind-the-scenes-offload-memory-management-on-the-intel-xeon-phi-coprocessor
 * 
 * KEY table for transfers:
 * https://software.intel.com/en-us/articles/effective-use-of-the-intel-compilers-offload-features
 * 
 * 
 * It seems that class members can be accesed in offload regionst in class methods without any in/out/inout.
 * It seems to be copied AUTOMAGICLY and without any pointer translations. (no verification in literature)
 * -->This allow me to stole pointer from MIC, but doesnt allow me to pass it as parametr of other function for offload
 * -->Only class members can use this pointer directly, I hope.
 * 
 * Fucking offload doesnt allow me to copy object with constructor to MIC. But I can alloc it With NOCOPY clauslue,
 * and upload simple sturcture containing all data and set up object on MIC (HOLY FUCK HACK)
 * I.E.: 
 *			param p =a.getparam();
 *			#pragma offload target(mic) nocopy(a) in(p)
 *			{
 *				a.setparam(p);
 *				...
 *			}
 *-->When MIC change object a, then use INOUT and getparam() setparam at the end of OFFLOAD
 * 
 * Other fucking hacks were written: 1) template <typename Device> class satanArrayCompare {};  in tnlArrayMIC_impl.h
 *                                      -->very similar to tnlArrayOperations, but expect first array to be on MIC and only for compare
 *                                   2) template <typename Type> struct satanHider{ Type * pointer} in tnlFile_impl.h
 *                                      -->use for passing untranslated pointer into MIC.      
 * 
 */

int main(void)
{
    cout << "tnlArray on MIC test by Satan:" <<endl; //LOL
	
	#ifdef HAVE_ICPC
		cout << "ICPC in USE" <<endl; //LOL
	#endif

	#ifdef HAVE_MIC
		cout << "MIC in USE" <<endl; //LOL
	#endif

		

//prepare arrays with data
	tnlArray<double,tnlMIC,int> aa(10);
	tnlArray<double,tnlMIC,int> ee(6);
	tnlArray<double,tnlHost,int> cc(5);

//fill it UP
tnlArrayParams<double,int> aa_params=aa.getParams();	
#pragma offload target(mic) nocopy(aa) in(aa_params)
{
	aa.setParams(aa_params);
	for(int i=0;i<aa.getSize();i++)
	{
		aa[i]=i;
	}
}

for(int i=0;i<5;i++)
{
	cc[i]=10+i;
}

//prepare arrays for funky tests
tnlArray<double,tnlMIC,int> bb(10);
tnlArray<double,tnlMIC,int> dd(0);

//TEST IT!
SatanSay("aa.getSize():");
SATANTEST(aa.getSize()==10);

SatanSay("Is aa filled correctly? (aa.getElement):");
for(int i=0;i<10;i++)
	SATANTEST(aa.getElement(i)==i);

SatanSay("Copy to bb(MIC->MIC) (=):");
bb=aa;
SATANTEST(aa.getSize()==bb.getSize());
for(int i=0;i<bb.getSize();i++)
	SATANTEST(bb.getElement(i)==i);

SatanSay("setLike:");
bb.setLike(cc);
SATANTEST(bb.getSize()==cc.getSize());
SatanSay("Copy (Host -> MIC) (=)");
bb=cc;
for(int i=0;i<bb.getSize();i++)
	SATANTEST(bb.getElement(i)==i+10);

SatanSay("setValue:");
bb.setValue(5);
for(int i=0;i<bb.getSize();i++)
	SATANTEST(bb.getElement(i)==5);

SatanSay("swap:");
aa.swap(bb);
SATANTEST(aa.getSize()==5||bb.getSize()==10);
for(int i=0;i<aa.getSize();i++)
{
	SATANTEST(aa.getElement(i)==5);
}
for(int i=0;i<bb.getSize();i++)
{
	SATANTEST(bb.getElement(i)==i);
}

SatanSay("(MIC -> MIC) ==");
aa.setLike(bb);
aa=bb;
SATANTEST(aa==bb);
SATANTEST(!(aa!=bb));
SATANTEST(aa!=ee);
bb.setElement(5,66);
SATANTEST(aa!=bb);

SatanSay("(Host -> MIC) !=");
aa.setLike(cc);
aa=cc;
SATANTEST(aa==cc);
aa.setElement(3,66);
SATANTEST(aa!=cc);

SatanSay("bidn (light test)");
dd.bind(bb,5);
SATANTEST(dd.getSize()==5);
SATANTEST(dd.getElement(1)==6);

//Mylsím, že není zdaleka testováno vše...

///////////////////////////////////////////////////////////////////////////////

SatanSay("File Array Test: \n");

//prepare arrays with data

aa.setSize(10);
ee.setSize(6);
cc.setSize(5);

//fill it UP
aa_params=aa.getParams();	
#pragma offload target(mic) nocopy(aa) in(aa_params)
{
	aa.setParams(aa_params);
	for(int i=0;i<aa.getSize();i++)
	{
		aa[i]=i;
	}
}

for(int i=0;i<5;i++)
{
	cc[i]=10+i;
}

tnlFile soubor;
soubor.open("/tmp/tnlArrayExperimentSave_cc.bin",tnlWriteMode);
cc.save(soubor);
soubor.close();

soubor.open("/tmp/tnlArrayExperimentSave_aa.bin",tnlWriteMode);
aa.save(soubor);
soubor.close();

ee.bind(aa,5,5);
ee.boundLoad("/tmp/tnlArrayExperimentSave_cc.bin");

SATANTEST( 10 == aa.getSize())
for(int i=0;i<5;i++)
{
	SATANTEST(aa.getElement(i)==i)
}
for(int i=5;i<10;i++)
{
	SATANTEST(aa.getElement(i)==i+5)
}

soubor.open("/tmp/tnlArrayExperimentSave_aa.bin",tnlReadMode);
cc.load(soubor);
soubor.close();

SATANTEST( 10 == cc.getSize())
for(int i=0;i<cc.getSize();i++)
{
	SATANTEST(cc.getElement(i)==i)
}

	SATANTESTRESULT;
    return 0;
}
