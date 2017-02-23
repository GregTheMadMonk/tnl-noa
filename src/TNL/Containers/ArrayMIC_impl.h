/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   tnlarrayMIC_impl.h
 * Author: hanouvit
 *
 * Created on 19. dubna 2016, 12:55
 */

/*
 * Because MIC Array differs from Host or Cuda Arrays, there is specialization for whole Array
 * 
 * It differ, becouse Im not able get pointer to MIC memory on processor, for funny pointer work with partially shared arrays.
 * So Im storing base adress and offset.
 * 
 *  */

#pragma once




#include <TNL/Devices/MIC.h>
#include <TNL/Assert.h>

#ifdef HAVE_MIC

namespace TNL {
namespace Containers {   


template< typename Element,
          typename Index >
struct ArrayParams {
      mutable Index size;
      mutable Element* allocationPointer;
      mutable Element* data;
      mutable int* referenceCounter;
};


template< typename Element,
          typename Index >
class Array <Element, Devices::MIC, Index> : public virtual Object
{
    
       protected:
      
      void releaseData() const; ///proč je tohle konst, když to mění objekt, a dost zásandě?

      //!Number of elements in array
      mutable Index size;

      mutable Element* allocationPointer;
      mutable Element* data;
     // mutable Index offset;  
        
     
      //! Pointer to data on MIC -- MUST be SET UP at beginign of PRAGMA traget
//      mutable Element* data;

      /****
       * Pointer to the originally allocated data. They might differ if one 
       * long array is partitioned into more shorter arrays. Each of them
       * must know the pointer on allocated data because the last one must
       * deallocate the array. If outer data (not allocated by TNL) are bind
       * then this pointer is zero since no deallocation is necessary.
       */
    //  mutable Element* allocationPointer;

      /****
       * Counter of objects sharing this array or some parts of it. The reference counter is
       * allocated after first sharing of the data between more arrays. This is to avoid
       * unnecessary dynamic memory allocation.
       */
      mutable int* referenceCounter;
      
      
    
   public:
      typedef Element ElementType;
      typedef Devices::MIC DeviceType;
      typedef Index IndexType;
      typedef Array< Element, Devices::Host, Index > HostType;
      typedef Array< Element, Devices::Cuda, Index > CudaType;
      typedef Array< Element, Devices::MIC, Index > ThisType;
      
      Array();
      
      Array( const IndexType& size );
      
      Array( Element* data,
                const IndexType& size );

      Array( Array< Element, Devices::MIC, Index >& array,
                const IndexType& begin = 0,
                const IndexType& size = 0 );

      static String getType();

      String getTypeVirtual() const;

      static String getSerializationType();

      virtual String getSerializationTypeVirtual() const;
      
      /*Holy fuck HACK*/
      __cuda_callable__ ArrayParams<Element,Index> getParams(void) const;
      __cuda_callable__ void setParams(ArrayParams<Element,Index> p);
      
      /****
       * This sets size of the array. If the array shares data with other arrays
       * these data are released. If the current data are not shared and the current
       * size is the same as the new one, nothing happens.
       */
      bool setSize( Index size );

     template< typename Array >
     bool setLike( const Array& array );

      void bind( Element* _data,
                 const Index _size );

      void bind( const Array< Element, Devices::MIC, Index >& array,
                 const IndexType& begin = 0,
                 const IndexType& size = 0 );

      template< int Size >
      void bind( StaticArray< Size, Element >& array );

      void swap( Array< Element, Devices::MIC, Index >& array );

      void reset();

      __cuda_callable__ Index getSize() const;

      void setElement( const Index& i, const Element& x );

      Element getElement( const Index& i ) const;

      __cuda_callable__ inline Element& operator[] ( const Index& i );

     __cuda_callable__ inline const Element& operator[] ( const Index& i ) const;

      Array< Element, Devices::MIC, Index >& operator = ( const Array< Element, Devices::MIC, Index >& array );

      template< typename ArrayT >
      Array< Element, Devices::MIC, Index >& operator = ( const ArrayT& array );

      template< typename Array >
      bool operator == ( const Array& array ) const;

      template< typename Array >
      bool operator != ( const Array& array ) const;

      void setValue( const Element& e );  //nastaveni hodnoty celymu poly

      __cuda_callable__ const Element* getData() const;

      __cuda_callable__ Element* getData();

      /*!
       * Returns true if non-zero size is set.
       */
     operator bool() const;

      //! This method measures data transfers done by this vector.
      /*!
       * Every time one touches this grid touches * size * sizeof( Real ) bytes are added
       * to transfered bytes in tnlStatistics.
       */
   #ifdef HAVE_NOT_CXX11
      template< typename IndexType2 >
      void touch( IndexType2 touches = 1 ) const;
   #else
      template< typename IndexType2 = Index >
      void touch( IndexType2 touches = 1 ) const;
   #endif      

      //! Method for saving the object to a file as a binary data.
      bool save( File& file ) const;

      //! Method for loading the object from a file as a binary data.
      bool load( File& file );
      
      //! This method loads data without reallocation. 
      /****
       * This is useful for loading data into shared arrays.
       * If the array was not initialize yet, common load is
       * performed. Otherwise, the array size must fit with
       * the size of array being loaded.
       */
      bool boundLoad( File& file );
      
      bool boundLoad( const String& fileName );
      
      using Object::load;

      using Object::save;

      ~Array();
      

};

/*template< typename Element, typename Index >
ostream& operator <<<Element,Devices::MIC,Index> ( ostream& str, const Array< Element, Devices::MIC, Index >& v );
*/
/////////////////////////////////////////////////////////////////////////////////////////////
//-----------------------------IMPLEMENTATION------------------------------------------------
/////////////////////////////////////////////////////////////////////////////////////////////

template< typename Element,
           typename Index >
Array< Element, Devices::MIC, Index >::
Array()
: size( 0 ),
 // offset( 0 ),
  data(0),
  allocationPointer( 0 ),
  referenceCounter( 0 )
{
};

template< typename Element,
           typename Index >
Array< Element, Devices::MIC, Index >::
Array( const IndexType& size )
: size( 0 ),
//  offset( 0 ),
  data( 0 ),
  allocationPointer( 0 ),
  referenceCounter( 0 )
{
   this->setSize( size );
}

template< typename Element,
           typename Index >
Array< Element, Devices::MIC, Index >::
Array( Element* data,
          const IndexType& size )
{
    TNL_ASSERT(false,std::cerr << "This constructor is NOT implemnted yet on MIC" << std::endl;);
}

template< typename Element,
           typename Index >
Array< Element, Devices::MIC, Index >::
Array( Array< Element, Devices::MIC, Index >& array,
          const IndexType& begin,
          const IndexType& size )
: size( size ),
//  offset( begin ),
  data( 0 ),
  allocationPointer( array.allocationPointer ),
  referenceCounter( 0 )
{
   TNL_ASSERT( begin < array.getSize(),
              std::cerr << " begin = " << begin << " array.getSize() = " << array.getSize() );
   TNL_ASSERT( begin + size  < array.getSize(),
              std::cerr << " begin = " << begin << " size = " << size <<  " array.getSize() = " << array.getSize() );
   if( ! this->size )
      this->size = array.getSize() - begin;
   if( array.allocationPointer )
   {
      if( array.referenceCounter )
      {
         this->referenceCounter = array.referenceCounter;
         *this->referenceCounter++;
      }
      else
      {
         this->referenceCounter = array.referenceCounter = new int;
         *this->referenceCounter = 2;            
      }
   }   
}


template< typename Element,typename Index >
String     
Array< Element, Devices::MIC, Index >::
getType()
{
   return String( "Array< " ) +
                     ::getType< Element >() + ", " +
                     DeviceType :: getDeviceType() + ", " +
                     ::getType< Index >() +
                     " >";
};

template< typename Element,
           typename Index >
String 
Array< Element, Devices::MIC, Index >::
getTypeVirtual() const
{
   return this->getType();
};

template< typename Element,
           typename Index >
String
Array< Element, Devices::MIC, Index >::
getSerializationType()
{
   return HostType::getType();
};

template< typename Element,
           typename Index >
String 
Array< Element, Devices::MIC, Index >::
getSerializationTypeVirtual() const
{   
   return this->getSerializationType();
};


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template< typename Element,
           typename Index >
__cuda_callable__
ArrayParams<Element,Index>
Array< Element, Devices::MIC, Index >::
getParams(void) const
{
    ArrayParams<Element,Index> p;
    p.size=this->size;
    p.allocationPointer=this->allocationPointer;
    p.data=this->data;
    p.referenceCounter=this->referenceCounter;
    return p;
}

template< typename Element,
           typename Index >
__cuda_callable__
void
Array< Element, Devices::MIC, Index >::
setParams(ArrayParams<Element,Index> p)
{
    this->size=p.size;
    this->allocationPointer=p.allocationPointer;
    this->data=p.data;
    this->referenceCounter=p.referenceCounter;
    
}


template< typename Element,
           typename Index >
void
Array< Element, Devices::MIC, Index >::
releaseData() const
{
   if( this->referenceCounter )
   {
      if( --*this->referenceCounter == 0 )
      {
         //ArrayOperations< Device >::freeMemory( this->allocationPointer );
         //free memory on MIC
        #pragma offload target(mic) 	//it may work -- look at tnlSatanExperimentalTest.cpp
	{
            free(allocationPointer);
	} ;
         delete this->referenceCounter;
         //std::cerr << "Deallocating reference counter " << this->referenceCounter << std::endl;
      }
   }
   else
      if( allocationPointer )
      {
        //free memory on MIC
        #pragma offload target(mic) 	//it may work -- look at tnlSatanExperimentalTest.cpp
	{
            free(allocationPointer);
	};
      }
   this->allocationPointer = 0;
   this->size = 0;
   this->data=0;
   this->referenceCounter = 0;
}


template< typename Element,
          typename Index >
bool
Array< Element, Devices::MIC, Index >::
setSize( const Index size )
{
   TNL_ASSERT( size >= 0,
              std::cerr << "You try to set size of Array to negative value."
                   << "New size: " << size << std::endl );
   if( this->size == size && allocationPointer && ! referenceCounter ) return true;
   this->releaseData();
   //ArrayOperations< Device >::allocateMemory( this->allocationPointer, size );
   //alloc memory on MIC
   #pragma offload target(mic)  //it may work -- look at tnlSatanExperimentalTest.cpp	
   {
       allocationPointer=(Element*)malloc(size*sizeof(Element));
   };
   this->size = size;
 //  this->offset=0;
   this->data=this->allocationPointer;
   return true;
};

template< typename Element,
           typename Index >
   template< typename ArrayT >
bool
Array< Element, Devices::MIC, Index >::
setLike( const ArrayT& array )
{
   TNL_ASSERT( array. getSize() >= 0,
              std::cerr << "You try to set size of Array to negative value."
                   << "Array size: " << array. getSize() << std::endl );
   return setSize( array.getSize() );
};


template< typename Element,
          typename Index >
void 
Array< Element, Devices::MIC, Index >::
reset()
{
   this->releaseData();
};


template< typename Element,
           typename Index >
void 
Array< Element, Devices::MIC, Index >::
bind( Element* data,
      const Index size )
{
   this->releaseData();
   this->data = data;
   this->size = size;   
}

template< typename Element,
           typename Index >
void
Array< Element, Devices::MIC, Index >::
bind( const Array< Element, Devices::MIC, Index >& array,
      const IndexType& begin,
      const IndexType& size )
{
   TNL_ASSERT( begin <= array.getSize(),
              std::cerr << " begin = " << begin << " array.getSize() = " << array.getSize() );
   TNL_ASSERT( begin + size  <= array.getSize(),
              std::cerr << " begin = " << begin << " size = " << size <<  " array.getSize() = " << array.getSize() );
   
   this->releaseData();
   if( size )
      this->size = size;
   else
      this->size = array.getSize() - begin;
   //this->data = const_cast< Element* >( &array.getData()[ begin ] );
   this->data = const_cast< Element* >( array.getData()+begin );
   this->allocationPointer = array.allocationPointer;
   if( array.allocationPointer )
   {
      if( array.referenceCounter )
      {
         this->referenceCounter = array.referenceCounter;
         ( *this->referenceCounter )++;
      }
      else
      {
         this->referenceCounter = array.referenceCounter = new int;
         *this->referenceCounter = 2;
         //std::cerr << "Allocating reference counter " << this->referenceCounter << std::endl;
      }
   }   
}

template< typename Element,
           typename Index >
   template< int Size >
void
Array< Element, Devices::MIC, Index >::
bind( StaticArray< Size, Element >& array )
{
  /*this->releaseData();
   this->size = Size;
   this->data = array.getData();*/
    
    TNL_ASSERT(false, cerr << "Not implemented jet for MIC :-(" <<endl );
}

template< typename Element,
          typename Index >
void 
Array< Element, Devices::MIC, Index >::
swap( Array< Element, Devices::MIC, Index >& array )
{
   TNL::swap( this->size, array.size );
   TNL::swap( this->data, array.data );
   TNL::swap( this->allocationPointer, array.allocationPointer );
   TNL::swap( this->referenceCounter, array.referenceCounter );
};

template< typename Element,
          typename Index >
__cuda_callable__ 
Index 
Array< Element, Devices::MIC, Index >::
getSize() const
{
    return this->size;
}


template< typename Element,
          typename Index >
void 
Array< Element, Devices::MIC, Index >::
setValue( const Element& e )
{
    Element ee=e;
    #pragma offload target(mic) in(ee)
    {
        for(int i=0;i<this->size;i++)
            data[i]=ee;
    }
}

template< typename Element,
          typename Index >
void 
Array< Element, Devices::MIC, Index >::
setElement( const Index& i, const Element& x )
{
    Element xx=x;
    Index ii=i;
    #pragma offload target(mic) in(xx,ii)
    {
        this->data[ii]=xx;
    }
}

template< typename Element,
          typename Index >
Element 
Array< Element, Devices::MIC, Index >::
getElement( const Index& i ) const
{
    Element x;
    Index ii=i;
    #pragma offload target(mic) in(ii) out(x)
    {
        x=this->data[i];
    }
    return x; 
}

template< typename Element,
           typename Index >
__cuda_callable__
Element* 
Array< Element, Devices::MIC, Index > :: 
getData()
{
   return this->data;
}

template< typename Element,
           typename Index >
__cuda_callable__
const Element* 
Array< Element, Devices::MIC, Index > :: 
getData() const
{
   return this->data;
}


template< typename Element,
           typename Index >
Array< Element, Devices::MIC, Index > :: 
operator bool() const
{
   return data != 0;
};


template< typename Element,
          typename Index >
__cuda_callable__
inline Element&
Array< Element, Devices::MIC, Index >::
operator[] ( const Index& i )
{
   TNL_ASSERT( 0 <= i && i < this->getSize(),
              std::cerr << "Wrong index for operator[] in Array "
                   << " index is " << i
                   << " and array size is " << this->getSize() );
   return this->data[ i ];
};

template< typename Element,
          typename Index >
__cuda_callable__
inline const Element&
Array< Element, Devices::MIC, Index >::
operator[] ( const Index& i ) const
{
   TNL_ASSERT( 0 <= i && i < this->getSize(),
              cerr << "Wrong index for operator[] in Array "
                   << " index is " << i
                   << " and array size is " << this->getSize() );
   return this->data[ i ];
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template< typename Element,
           typename Index >
Array< Element, Devices::MIC, Index >&
Array< Element, Devices::MIC, Index >::
operator = ( const Array< Element, Devices::MIC, Index >& array )
{
   TNL_ASSERT( array. getSize() == this->getSize(),
           std::cerr << "Source size: " << array. getSize() << std::endl
                << "Target size: " << this->getSize() << std::endl );
  
    ArrayParams<Element,Index>params= array.getParams();  
    #pragma offload target(mic) in(params)
    {   
        for(int i=0;i<params.size;i++)
        {
            data[i]=params.data[i];
        }
    }
   return ( *this );
};

template< typename Element,
           typename Index >
   template< typename ArrayT >
Array< Element, Devices::MIC, Index >&
Array< Element, Devices::MIC, Index >::
operator = ( const ArrayT& array )
{
 
   TNL_ASSERT( array. getSize() == this->getSize(),
           std::cerr << "Source size: " << array. getSize() << endl
                << "Target size: " << this->getSize() << std::endl );
   

   
    if( std::is_same< typename ArrayT::DeviceType, Devices::MIC >::value )
    {
        //MIC -> MIC
        std::cout << "MIC to MIC transfer.." <<std::endl;
        std::cout << "Zajímalo by mne zda se tento kód někdy spustí... SNAD NE, protože nic nedělá " << std::endl;
        std::cout << "S implementaci je trochu háčk" <<std::endl; 
     
    }
    else
    {
        if(std::is_same< typename ArrayT::DeviceType, Devices::Host >::value)
        {
            //Host -> MIC
            //cout << "Host to MIC" <<endl;
            const typename Array::ElementType * sdata = array.getData();
            #pragma offload target(mic) in(sdata:length(this->size))
            {   
                 for(int i=0;i<this->size;i++) //MIC má memcpy, ale tohle je KISS
                 {
                    data[i]=sdata[i];
                 }
            }          
        }
        else
        {
            //?? ->MIC
            std::cout << "Extremly slow transfer at = of Array to MIC count:" << this->size <<std::endl;
            for(int i=0;i<this->size;i++)
            {
                this->setElement(i,array.getElement(i));
            }            
        }
    }
   
   return ( *this );
};

//Obdoba TNL ARRAY OPERATIONS... možná by se to tam mělo pak přepsat...
template <typename Device> 
class satanArrayCompare{};

template <>
class satanArrayCompare< Devices::Host >{
    
    public:
    //expected ArrayA is Devices::MIC 
    template< typename ArrayA, typename ArrayB>
    static bool compare( ArrayA& A, ArrayB& B  )
    {              
        if(A.getSize()!=B.getSize())
        {
            return false;
        }
        else
        {    
            ArrayParams<typename ArrayA::ElementType,typename ArrayA::IndexType> A_params=A.getParams();
            const typename ArrayB::ElementType * B_data=B.getData();
            typename ArrayB::IndexType B_size=B.getSize();
            bool result=true;
            typename ArrayA::ThisType AA;
            #pragma offload target(mic) nocopy(AA) in(A_params) in(B_data:length(B_size)) inout(result)
            {
                
                AA.setParams(A_params);               
                for(int i=0;i<B_size;i++)
                {
                   if(AA[i]!=B_data[i])
                    {
                        result=false;
                        break;
                    }
                }
            }
            return result;  
        }    
    };
};

template <>
class satanArrayCompare< Devices::MIC >{
    
     public:
    //expected ArrayA is on MIC 
    template< typename ArrayA, typename ArrayB>
    static bool compare( ArrayA& A, ArrayB& B  )
    {
        if(A.getSize()!=B.getSize())
        {
            return false;
        }
        else
        {    
            ArrayParams<typename ArrayA::ElementType,typename ArrayA::IndexType> A_params=A.getParams();
            ArrayParams<typename ArrayB::ElementType,typename ArrayB::IndexType> B_params=B.getParams();
            typename ArrayA::ThisType AA;
            typename ArrayB::ThisType BB;
            bool result=true;
            #pragma offload target(mic) nocopy(AA,BB) in(A_params,B_params) inout(result)
            {
                AA.setParams(A_params);
                BB.setParams(B_params);
                for(int i=0;i<AA.getSize();i++)
                {
                    if(AA[i]!=BB[i])
                    {
                        result=false;
                        break;
                    }
                }
            }
            return result;  
        }    
    };
};


template< typename Element,
          typename Index >
   template< typename ArrayT >
bool
Array< Element, Devices::MIC, Index >::
operator == ( const ArrayT& array ) const
{
    //cout << "Comparing: " << this->getType() << ": " <<this->getSize() << " and " << array.getType() <<": "<< array.getSize() <<endl;
    return satanArrayCompare< typename ArrayT :: DeviceType> :: compare((*this),array);

}


template< typename Element,
          typename Index >
   template< typename ArrayT >
bool Array< Element, Devices::MIC, Index > :: 
operator != ( const ArrayT& array ) const
{
   return ! ( ( *this ) == array );
}


template< typename Element,
          typename Index >
Array< Element, Devices::MIC, Index >::
~Array()
{
   this->releaseData();
}

template< typename Element,
          typename Index >
bool Array< Element, Devices::MIC, Index > :: 
save( File& file ) const
{
   if( ! Object :: save( file ) )
      return false;
#ifdef HAVE_NOT_CXX11
   if( ! file. write< const Index, tnlHost >( &this->size ) )
      return false;
#else            
   if( ! file. write( &this->size ) )
      return false;
#endif
   if( this->size != 0 && ! ArrayIO< Element, Devices::MIC, Index >::save( file, this->data, this->size ) )
   {
      std::cerr << "I was not able to save " << this->getType()
           << " with size " << this->getSize() << std::endl;
      return false;
   }
   return true;
};

template< typename Element,
          typename Index >
bool
Array< Element, Devices::MIC, Index >::
load( File& file )
{
   if( ! Object :: load( file ) )
      return false;
   Index _size;
#ifdef HAVE_NOT_CXX11
   if( ! file. read< Index, tnlHost >( &_size ) )
      return false;
#else   
   if( ! file. read( &_size ) )
      return false;
#endif      
   if( _size < 0 )
   {
      std::cerr << "Error: The size " << _size << " of the file is not a positive number or zero." << std::endl;
      return false;
   }
   setSize( _size );
   if( _size )
   {
      if( ! ArrayIO< Element, Devices::MIC, Index >::load( file, this->data, this->size ) )
      {
         std::cerr << "I was not able to load " << this->getType()
                    << " with size " << this->getSize() << std::endl;
         return false;
      }
   }
   return true;
}

template< typename Element,
          typename Index >
bool
Array< Element, Devices::MIC, Index >::
boundLoad( File& file )
{
   if( ! Object :: load( file ) )
      return false;
   Index _size;
#ifdef HAVE_NOT_CXX11
   if( ! file. read< Index, tnlHost >( &_size ) )
      return false;
#else   
   if( ! file. read( &_size ) )
      return false;
#endif      
   if( _size < 0 )
   {
      std::cerr << "Error: The size " << _size << " of the file is not a positive number or zero." << std::endl;
      return false;
   }
   if( this->getSize() != 0 )
   {
      if( this->getSize() != _size )
      {
         std::cerr << "Error: The current array size is not zero and it is different from the size of" << std::endl
                   << "the array being loaded. This is not possible. Call method reset() before." << std::endl;
         return false;
      }
   }
   else setSize( _size );
   if( _size )
   {
      if( ! ArrayIO< Element, Devices::MIC, Index >::load( file, this->data, this->size ) )
      {
         std::cerr << "I was not able to load " << this->getType()
                    << " with size " << this->getSize() << std::endl;
         return false;
      }
   }
   return true;
}

template< typename Element,
          typename Index >
bool
Array< Element, Devices::MIC, Index >::
boundLoad( const String& fileName )
{
   File file;
   if( ! file. open( fileName, tnlReadMode ) )
   {
      std::cerr << "I am not bale to open the file " << fileName << " for reading." << std::endl;
      return false;
   }
   if( ! this->boundLoad( file ) )
      return false;
   if( ! file. close() )
   {
      std::cerr << "An error occurred when I was closing the file " << fileName << "." << std::endl;
      return false;
   }
   return true;   
}

template< typename Element,
           typename Index >
   template< typename IndexType2 >
void Array< Element, Devices::MIC, Index > :: touch( IndexType2 touches ) const
{
   //TODO: implement
};

} // namespace Containers
} // namespace TNL

#endif



