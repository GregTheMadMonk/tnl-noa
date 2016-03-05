/***************************************************************************
                          tnlDicomSeries_impl.h  -  description
                             -------------------
    begin                : Jul 19, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.                                       
     
     Tomas Oberhuber     tomas.oberhuber@fjfi.cvut.cz
     Jiri Kafka          kafka9@seznam.cz
     Pavel Neskudla
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include <core/images/tnlDicomSeries.h>
#include <core/images/tnlDicomSeriesInfo.h>
#include <dirent.h>


int findLastIndexOf(tnlString &str, const char* c)
{
    for (int i = str.getLength(); i > -1; i--)
    {
        char *a = &(str.operator [](i-1));
        if(*a == *c)
            return i;
    }
    return -1;
}

int filter(const struct dirent *dire)
{
    //check it is not DIR or unknowm d_type
    if(dire->d_type == DT_UNKNOWN && dire->d_type == DT_DIR)
        return 0;

    return 1;
}

inline tnlDicomSeries::tnlDicomSeries( const tnlString& filePath)
{
#ifdef HAVE_DCMTK_H
    dicomImage = 0;
    pixelData = 0;
#endif    
    imagesInfo.imagesCount = 0;
    imagesInfo.maxColorValue = 0;
    imagesInfo.minColorValue = 128000;

    if( !loadDicomSeries( filePath ) )
        isLoaded = false;
    else
        isLoaded = true;
}

inline tnlDicomSeries::~tnlDicomSeries()
{
    int length = dicomSeriesHeaders.getSize();
    for(int i = 0; i < length; i++)
    {
        tnlDicomHeader *header = dicomSeriesHeaders[i];
        delete header;
        header = 0;
    }

#ifdef HAVE_DCMTK_H    
    if(dicomImage)
        delete dicomImage;

    if(pixelData)
        delete pixelData;
#endif    
}

template< typename Real,
          typename Device,
          typename Index,
          typename Vector >
bool
tnlDicomSeries::
getImage( const int imageIdx,
          const tnlGrid< 2, Real, Device, Index >& grid,
          const tnlRegionOfInterest< int > roi,
          Vector& vector )
{
#ifdef HAVE_DCMTK_H
   const Uint16* imageData = this->getData( imageIdx );
   typedef tnlGrid< 2, Real, Device, Index > GridType;
   typename GridType::Cell cell( grid );
   
   Index i, j;
   int position( 0 );
   for( i = 0; i < this->height; i ++ )
   {
      for( j = 0; j < this->width; j ++ )
      {
         if( roi.isIn( i, j ) )
         {
            cell.getCoordinates().x() = j - roi.getLeft();
            cell.getCoordinates().y() = roi.getBottom() - 1 - i;
            cell.refresh();
            //Index cellIndex = grid.getCellIndex( CoordinatesType( j - roi.getLeft(),
            //                                                      roi.getBottom() - 1 - i ) );
            Uint16 col = imageData[ position ];
            vector.setElement( cell.getIndex(), ( Real ) col / ( Real ) 65535 );
            //cout << vector.getElement( cellIndex ) << " ";
         }
         position++;
      }
      //cout << endl;
   }
   return true;
#else
   cerr << "DICOM format is not supported in this build of TNL." << endl;
   return false;     
#endif
}

inline bool tnlDicomSeries::retrieveFileList( const tnlString& filePath)
{
    tnlString filePathString(filePath);
    tnlString suffix(filePath.getString(), filePathString.getLength() - 3);

    /***
     * Check DICOM files
     */
   if( suffix != "ima" && suffix != "dcm" )
   {
       cerr << "The given file is not a DICOM file." << endl;
      return false;
   }

   int fileNamePosition = findLastIndexOf( filePathString, "/" );

   /***
    * Parse file path
    */
   tnlString fileName(filePath.getString(), fileNamePosition);
   tnlString directoryPath(filePath.getString(), 0, filePathString.getLength() - fileNamePosition);

   int separatorPosition = findLastIndexOf(fileName, "_");
   if (separatorPosition == -1)
   {
      //try another separator
      separatorPosition = findLastIndexOf(fileName, "-");
   }
   if( separatorPosition == -1 )
      return false;
   else
   {
      //numbered files
      tnlString fileNamePrefix(fileName.getString(), 0, fileName.getLength() - separatorPosition);

      struct dirent **dirp;
      tnlList<tnlString > files;

      //scan and sort directory
      int ndirs = scandir(directoryPath.getString(), &dirp, filter, alphasort);
      for(int i = 0 ; i < ndirs; ++i)
      {
         files.Append( tnlString((char *)dirp[i]->d_name));
         delete dirp[i];
      }

      for (int i = 0; i < files.getSize(); i++)
      {
         //check if file prefix contained
         if (strstr(files[ i ].getString(), fileNamePrefix.getString()))
         {
            fileList.Append( directoryPath + files[ i ] );
         }
      }
   }
   return true;
}

inline bool tnlDicomSeries::loadImage( const tnlString& filePath, int number)
{
#ifdef HAVE_DCMTK_H
   //load header
   tnlDicomHeader *header = new tnlDicomHeader();
   dicomSeriesHeaders.setSize( fileList.getSize() );
   dicomSeriesHeaders.setElement( number, header );
   if( !header->loadFromFile( filePath ) )
      return false;

   //check series UID
   const tnlString& seriesUID = dicomSeriesHeaders[ 0 ]->getSeriesInfo().getSeriesInstanceUID();
   if( seriesUID != header->getSeriesInfo().getSeriesInstanceUID() )
      return false;

   //load image
   if( dicomImage ) delete dicomImage;
   dicomImage = NULL;

   dicomImage = new DicomImage( filePath.getString() );

   if(dicomImage->getFrameCount() > 1)
   {
      cout << filePath <<" not supported format-Dicom Image has more than one frame";
      return false;
   }

   if(!dicomImage->isMonochrome())
   {
      cout << filePath <<" not supported format--Dicom Image is not monochrome";
      return false;
   }

    if (dicomImage != NULL)
    {
        EI_Status imageStatus = dicomImage->getStatus();
        if (imageStatus == EIS_Normal)
        {
            //ok - image loaded
        }
        else if (EIS_MissingAttribute)
        {
            //bitmap is propably old ARC/NEMA format
            cerr << "Error: cannot load DICOM image(ACR/NEMA) (" << DicomImage::getString (dicomImage->getStatus()) << ")" << endl;

            delete dicomImage;
            dicomImage = NULL;
            return false;
        }
        else
        {
            delete dicomImage;
            dicomImage = NULL;
            cerr << "Error: cannot load DICOM image (" << DicomImage::getString (dicomImage->getStatus()) << ")" << endl;
            return false;
        }
    }

    if(number == 0)
    {
        this->height = dicomImage->getHeight();
    }
    else if(dicomImage->getHeight() != this->height)
    {
        cerr << filePath <<" image has bad height value\n";
    }

    if(number == 0)
    {
        this->width = dicomImage->getWidth ();
    }
    else if(dicomImage->getWidth() != this->width)
    {
        cerr << filePath <<" image has bad width value\n";
    }

    if(number == 0)
    {
        imagesInfo.bps = dicomImage->getDepth ();
    }
    else if( dicomImage->getDepth() != imagesInfo.bps )
    {
        cerr << filePath <<" image has bad bps value\n";
    }

    //update vales
    double min, max;
    dicomImage->getMinMaxValues( min, max );
    if(imagesInfo.minColorValue > min)
    {
        imagesInfo.minColorValue = min;
    }

    if(imagesInfo.maxColorValue < max)
    {
        imagesInfo.maxColorValue = max;
    }

    const unsigned long size = dicomImage->getOutputDataSize(16);
    //number of unsigned ints to allocate
    imagesInfo.frameUintsCount = size / sizeof(Uint16);
    if (number == 0)
    {//perform allocation only once
        imagesInfo.frameSize = size;
        if (pixelData)
            delete pixelData;
        pixelData = new Uint16[imagesInfo.frameUintsCount * fileList.getSize()];
    }
    else
    {//check image size for compatibility
        if( imagesInfo.frameSize != size )
        {
            cerr << filePath << " image has bad frame size value\n";
            return false;
        }
    }

    dicomImage->setMinMaxWindow();
    double center, width;
    dicomImage->getWindow(center,width);
    imagesInfo.window.center = center;
    imagesInfo.window.width = width ;
    dicomImage->setWindow(imagesInfo.window.center, imagesInfo.window.width);

    void *target = pixelData + (imagesInfo.frameUintsCount * imagesInfo.imagesCount);
    dicomImage->getOutputData(target,size,16);
    imagesInfo.imagesCount++;

    //delete image object - data are stored separately
    delete dicomImage;
    dicomImage = NULL;
    return true;
#else
    cerr << "DICOM format is not supported in this build of TNL." << endl;
    return false;
#endif    
}


inline bool tnlDicomSeries::loadDicomSeries( const tnlString& filePath )
{
   /***
    * Load list of files
    */
   if( ! retrieveFileList( filePath ) )
   {
      cerr << "I am not able to retrieve the files of the DICOM series in " << filePath << "." << endl;
      return false;
   }

   //load images
   int imagesCountToLoad = fileList.getSize();
   for( int i=0; i < imagesCountToLoad; i++ )
   {
      if( !loadImage( fileList[ i ].getString(),i ) )
      {
         cerr << fileList[ i ] << " skipped";
      }
   }
   return true;
}

inline int tnlDicomSeries::getImagesCount()
{
    return imagesInfo.imagesCount;
}

#ifdef HAVE_DCMTK_H
inline const Uint16 *tnlDicomSeries::getData( int imageNumber )
{
    return &pixelData[ imageNumber * imagesInfo.frameUintsCount ];
}
#endif

inline int tnlDicomSeries::getColorCount()
{
    return imagesInfo.colorsCount;
}

inline int tnlDicomSeries::getBitsPerSampleCount()
{
    return imagesInfo.bps;
}

inline int tnlDicomSeries::getMinColorValue()
{
    return imagesInfo.minColorValue;
}

inline WindowCenterWidth tnlDicomSeries::getWindowDefaults()
{
    return imagesInfo.window;
}

inline int tnlDicomSeries::getMaxColorValue()
{
    return imagesInfo.maxColorValue;
}

inline void tnlDicomSeries::freeData()
{
#ifdef HAVE_DCMTK_H
    if (pixelData)
        delete pixelData;
    pixelData = NULL;
#endif    
}

inline tnlDicomHeader &tnlDicomSeries::getHeader(int image)
{
    //check user argument
    if((image > 0) | (image <= dicomSeriesHeaders.getSize()))
        return *dicomSeriesHeaders.getElement(image);
}

inline bool tnlDicomSeries::isDicomSeriesLoaded()
{
    return isLoaded;
}

