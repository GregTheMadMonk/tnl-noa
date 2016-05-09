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
 * Because MIC Array differs from Host or Cuda Arrays, there is specialization for whole tnlArray
 * 
 * It differ, becouse Im not able get pointer to MIC memory on processor, for funny pointer work with partially shared arrays.
 * So Im storing base adress and offset.
 * 
 *  */

#ifndef TNLARRAYMIC_IMPL_H
#define TNLARRAYMIC_IMPL_H

#ifdef HAVE_MIC

#include <core/tnlMIC.h>

template< typename Element,
          typename Index >
struct tnlArrayParams {
      mutable Index size;
      mutable Element* allocationPointer;
      mutable Element* data;
      mutable int* referenceCounter;
};


template< typename Element,
          typename Index >
class tnlArray <Element, tnlMIC, Index> : public virtual tnlObject
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
      typedef tnlMIC DeviceType;
      typedef Index IndexType;
      typedef tnlArray< Element, tnlHost, Index > HostType;
      typedef tnlArray< Element, tnlCuda, Index > CudaType;
      typedef tnlArray< Element, tnlMIC, Index > ThisType;
      
      tnlArray();
      
      tnlArray( const IndexType& size );
      
      tnlArray( Element* data,
                const IndexType& size );

      tnlArray( tnlArray< Element, tnlMIC, Index >& array,
                const IndexType& begin = 0,
                const IndexType& size = 0 );

      static tnlString getType();

      tnlString getTypeVirtual() const;

      static tnlString getSerializationType();

      virtual tnlString getSerializationTypeVirtual() const;
      
      /*Holy fuck HACK*/
      __device_callable__ tnlArrayParams<Element,Index> getParams(void) const;
      __device_callable__ void setParams(tnlArrayParams<Element,Index> p);
      
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

      void bind( const tnlArray< Element, tnlMIC, Index >& array,
                 const IndexType& begin = 0,
                 const IndexType& size = 0 );

      template< int Size >
      void bind( tnlStaticArray< Size, Element >& array );

      void swap( tnlArray< Element, tnlMIC, Index >& array );

      void reset();

      __device_callable__ Index getSize() const;

      void setElement( const Index& i, const Element& x );

      Element getElement( const Index& i ) const;

      __device_callable__ inline Element& operator[] ( const Index& i );

     __device_callable__ inline const Element& operator[] ( const Index& i ) const;

      tnlArray< Element, tnlMIC, Index >& operator = ( const tnlArray< Element, tnlMIC, Index >& array );

      template< typename Array >
      tnlArray< Element, tnlMIC, Index >& operator = ( const Array& array );

      template< typename Array >
      bool operator == ( const Array& array ) const;

      template< typename Array >
      bool operator != ( const Array& array ) const;

      void setValue( const Element& e );  //nastaveni hodnoty celymu poly

      __device_callable__ const Element* getData() const;

      __device_callable__ Element* getData();

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
    //  template< typename IndexType2 = Index >
    //  void touch( IndexType2 touches = 1 ) const;
   #endif      

      //! Method for saving the object to a file as a binary data.
      bool save( tnlFile& file ) const;

      //! Method for loading the object from a file as a binary data.
      bool load( tnlFile& file );
      
      //! This method loads data without reallocation. 
      /****
       * This is useful for loading data into shared arrays.
       * If the array was not initialize yet, common load is
       * performed. Otherwise, the array size must fit with
       * the size of array being loaded.
       */
      bool boundLoad( tnlFile& file );
      
      bool boundLoad( const tnlString& fileName );
      
      using tnlObject::load;

      using tnlObject::save;

      ~tnlArray();
      

};

/*template< typename Element, typename Index >
ostream& operator <<<Element,tnlMIC,Index> ( ostream& str, const tnlArray< Element, tnlMIC, Index >& v );
*/
/////////////////////////////////////////////////////////////////////////////////////////////
//-----------------------------IMPLEMENTATION------------------------------------------------
/////////////////////////////////////////////////////////////////////////////////////////////

template< typename Element,
           typename Index >
tnlArray< Element, tnlMIC, Index >::
tnlArray()
: size( 0 ),
 // offset( 0 ),
  data(0),
  allocationPointer( 0 ),
  referenceCounter( 0 )
{
};

template< typename Element,
           typename Index >
tnlArray< Element, tnlMIC, Index >::
tnlArray( const IndexType& size )
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
tnlArray< Element, tnlMIC, Index >::
tnlArray( Element* data,
          const IndexType& size )
{
    tnlAssert(false,cerr << "This constructor is NOT implemnted yet on MIC" <<endl;);
}

template< typename Element,
           typename Index >
tnlArray< Element, tnlMIC, Index >::
tnlArray( tnlArray< Element, tnlMIC, Index >& array,
          const IndexType& begin,
          const IndexType& size )
: size( size ),
//  offset( begin ),
  data( 0 ),
  allocationPointer( array.allocationPointer ),
  referenceCounter( 0 )
{
   tnlAssert( begin < array.getSize(),
              std::cerr << " begin = " << begin << " array.getSize() = " << array.getSize() );
   tnlAssert( begin + size  < array.getSize(),
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
tnlString     
tnlArray< Element, tnlMIC, Index >::
getType()
{
   return tnlString( "tnlArray< " ) +
                     ::getType< Element >() + ", " +
                     DeviceType :: getDeviceType() + ", " +
                     ::getType< Index >() +
                     " >";
};

template< typename Element,
           typename Index >
tnlString 
tnlArray< Element, tnlMIC, Index >::
getTypeVirtual() const
{
   return this->getType();
};

template< typename Element,
           typename Index >
tnlString
tnlArray< Element, tnlMIC, Index >::
getSerializationType()
{
   return HostType::getType();
};

template< typename Element,
           typename Index >
tnlString 
tnlArray< Element, tnlMIC, Index >::
getSerializationTypeVirtual() const
{   
   return this->getSerializationType();
};


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template< typename Element,
           typename Index >
__device_callable__
tnlArrayParams<Element,Index>
tnlArray< Element, tnlMIC, Index >::
getParams(void) const
{
    tnlArrayParams<Element,Index> p;
    p.size=this->size;
    p.allocationPointer=this->allocationPointer;
    p.data=this->data;
    p.referenceCounter=this->referenceCounter;
    return p;
}

template< typename Element,
           typename Index >
__device_callable__
void
tnlArray< Element, tnlMIC, Index >::
setParams(tnlArrayParams<Element,Index> p)
{
    this->size=p.size;
    this->allocationPointer=p.allocationPointer;
    this->data=p.data;
    this->referenceCounter=p.referenceCounter;
    
}


template< typename Element,
           typename Index >
void
tnlArray< Element, tnlMIC, Index >::
releaseData() const
{
   if( this->referenceCounter )
   {
      if( --*this->referenceCounter == 0 )
      {
         //tnlArrayOperations< Device >::freeMemory( this->allocationPointer );
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
tnlArray< Element, tnlMIC, Index >::
setSize( const Index size )
{
   tnlAssert( size >= 0,
              cerr << "You try to set size of tnlArray to negative value."
                   << "New size: " << size << endl );
   if( this->size == size && allocationPointer && ! referenceCounter ) return true;
   this->releaseData();
   //tnlArrayOperations< Device >::allocateMemory( this->allocationPointer, size );
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
   template< typename Array >
bool
tnlArray< Element, tnlMIC, Index >::
setLike( const Array& array )
{
   tnlAssert( array. getSize() >= 0,
              cerr << "You try to set size of tnlArray to negative value."
                   << "Array size: " << array. getSize() << endl );
   return setSize( array.getSize() );
};


template< typename Element,
          typename Index >
void 
tnlArray< Element, tnlMIC, Index >::
reset()
{
   this->releaseData();
};


template< typename Element,
           typename Index >
void 
tnlArray< Element, tnlMIC, Index >::
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
tnlArray< Element, tnlMIC, Index >::
bind( const tnlArray< Element, tnlMIC, Index >& array,
      const IndexType& begin,
      const IndexType& size )
{
   tnlAssert( begin <= array.getSize(),
              std::cerr << " begin = " << begin << " array.getSize() = " << array.getSize() );
   tnlAssert( begin + size  <= array.getSize(),
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
tnlArray< Element, tnlMIC, Index >::
bind( tnlStaticArray< Size, Element >& array )
{
  /*this->releaseData();
   this->size = Size;
   this->data = array.getData();*/
    
    tnlAssert(false, cerr << "Not implemented jet for MIC :-(" <<endl );
}

template< typename Element,
          typename Index >
void 
tnlArray< Element, tnlMIC, Index >::
swap( tnlArray< Element, tnlMIC, Index >& array )
{
   ::swap( this->size, array.size );
   ::swap( this->data, array.data );
   ::swap( this->allocationPointer, array.allocationPointer );
   ::swap( this->referenceCounter, array.referenceCounter );
};

template< typename Element,
          typename Index >
__device_callable__ 
Index 
tnlArray< Element, tnlMIC, Index >::
getSize() const
{
    return this->size;
}


template< typename Element,
          typename Index >
void 
tnlArray< Element, tnlMIC, Index >::
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
tnlArray< Element, tnlMIC, Index >::
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
tnlArray< Element, tnlMIC, Index >::
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
__device_callable__
Element* 
tnlArray< Element, tnlMIC, Index > :: 
getData()
{
   return this->data;
}

template< typename Element,
           typename Index >
__device_callable__
const Element* 
tnlArray< Element, tnlMIC, Index > :: 
getData() const
{
   return this->data;
}


template< typename Element,
           typename Index >
tnlArray< Element, tnlMIC, Index > :: 
operator bool() const
{
   return data != 0;
};


template< typename Element,
          typename Index >
__device_callable__
inline Element&
tnlArray< Element, tnlMIC, Index >::
operator[] ( const Index& i )
{
   tnlAssert( 0 <= i && i < this->getSize(),
              cerr << "Wrong index for operator[] in tnlArray "
                   << " index is " << i
                   << " and array size is " << this->getSize() );
   return this->data[ i ];
};

template< typename Element,
          typename Index >
__device_callable__
inline const Element&
tnlArray< Element, tnlMIC, Index >::
operator[] ( const Index& i ) const
{
   tnlAssert( 0 <= i && i < this->getSize(),
              cerr << "Wrong index for operator[] in tnlArray "
                   << " index is " << i
                   << " and array size is " << this->getSize() );
   return this->data[ i ];
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template< typename Element,
           typename Index >
tnlArray< Element, tnlMIC, Index >&
tnlArray< Element, tnlMIC, Index >::
operator = ( const tnlArray< Element, tnlMIC, Index >& array )
{
   tnlAssert( array. getSize() == this->getSize(),
           cerr << "Source size: " << array. getSize() << endl
                << "Target size: " << this->getSize() << endl );
  
    tnlArrayParams<Element,Index>params= array.getParams();  
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
   template< typename Array >
tnlArray< Element, tnlMIC, Index >&
tnlArray< Element, tnlMIC, Index >::
operator = ( const Array& array )
{
 
   tnlAssert( array. getSize() == this->getSize(),
           cerr << "Source size: " << array. getSize() << endl
                << "Target size: " << this->getSize() << endl );
   

   
    if( std::is_same< typename Array::DeviceType, tnlMIC >::value )
    {
        //MIC -> MIC
        cout << "MIC to MIC transfer.." <<endl;
        cout << "Zajímalo by mne zda se tento kód někdy spustí... SNAD NE, protože nic nedělá " << endl;
        cout << "S implementaci je trochu háčk" <<endl; 
     
    }
    else
    {
        if(std::is_same< typename Array::DeviceType, tnlHost >::value)
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
            cout << "Extremly slow transfer at = of tnlArray to MIC count:" << this->size <<endl;
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
class satanArrayCompare< tnlHost >{
    
    public:
    //expected ArrayA is tnlMIC 
    template< typename ArrayA, typename ArrayB>
    static bool compare( ArrayA& A, ArrayB& B  )
    {              
        if(A.getSize()!=B.getSize())
        {
            return false;
        }
        else
        {    
            tnlArrayParams<typename ArrayA::ElementType,typename ArrayA::IndexType> A_params=A.getParams();
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
class satanArrayCompare< tnlMIC >{
    
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
            tnlArrayParams<typename ArrayA::ElementType,typename ArrayA::IndexType> A_params=A.getParams();
            tnlArrayParams<typename ArrayB::ElementType,typename ArrayB::IndexType> B_params=B.getParams();
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
   template< typename Array >
bool
tnlArray< Element, tnlMIC, Index >::
operator == ( const Array& array ) const
{
    //cout << "Comparing: " << this->getType() << ": " <<this->getSize() << " and " << array.getType() <<": "<< array.getSize() <<endl;
    return satanArrayCompare< typename Array :: DeviceType> :: compare((*this),array);

}


template< typename Element,
          typename Index >
   template< typename Array >
bool tnlArray< Element, tnlMIC, Index > :: 
operator != ( const Array& array ) const
{
   return ! ( ( *this ) == array );
}


template< typename Element,
          typename Index >
tnlArray< Element, tnlMIC, Index >::
~tnlArray()
{
   this->releaseData();
}

template< typename Element,
          typename Index >
bool tnlArray< Element, tnlMIC, Index > :: 
save( tnlFile& file ) const
{
   if( ! tnlObject :: save( file ) )
      return false;
#ifdef HAVE_NOT_CXX11
   if( ! file. write< const Index, tnlHost >( &this->size ) )
      return false;
#else            
   if( ! file. write( &this->size ) )
      return false;
#endif
   if( this->size != 0 && ! tnlArrayIO< Element, tnlMIC, Index >::save( file, this->data, this->size ) )
   {
      cerr << "I was not able to save " << this->getType()
           << " with size " << this->getSize() << endl;
      return false;
   }
   return true;
};

template< typename Element,
          typename Index >
bool
tnlArray< Element, tnlMIC, Index >::
load( tnlFile& file )
{
   if( ! tnlObject :: load( file ) )
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
      cerr << "Error: The size " << _size << " of the file is not a positive number or zero." << endl;
      return false;
   }
   setSize( _size );
   if( _size )
   {
      if( ! tnlArrayIO< Element, tnlMIC, Index >::load( file, this->data, this->size ) )
      {
         cerr << "I was not able to load " << this->getType()
                    << " with size " << this->getSize() << endl;
         return false;
      }
   }
   return true;
}

template< typename Element,
          typename Index >
bool
tnlArray< Element, tnlMIC, Index >::
boundLoad( tnlFile& file )
{
   if( ! tnlObject :: load( file ) )
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
      cerr << "Error: The size " << _size << " of the file is not a positive number or zero." << endl;
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
      if( ! tnlArrayIO< Element, tnlMIC, Index >::load( file, this->data, this->size ) )
      {
         cerr << "I was not able to load " << this->getType()
                    << " with size " << this->getSize() << endl;
         return false;
      }
   }
   return true;
}

template< typename Element,
          typename Index >
bool
tnlArray< Element, tnlMIC, Index >::
boundLoad( const tnlString& fileName )
{
   tnlFile file;
   if( ! file. open( fileName, tnlReadMode ) )
   {
      cerr << "I am not bale to open the file " << fileName << " for reading." << endl;
      return false;
   }
   if( ! this->boundLoad( file ) )
      return false;
   if( ! file. close() )
   {
      cerr << "An error occurred when I was closing the file " << fileName << "." << endl;
      return false;
   }
   return true;   
}

#endif

#endif /* TNLARRAYMIC_IMPL_H */


