template < typename T >
T **Allocate2DArray( int nRows, int nCols)
{
    //(step 1) allocate memory for array of elements of column
    T **ppi = new T*[nRows];

    //(step 2) allocate memory for array of elements of each row
    T *curPtr = new T [nRows * nCols];

    // Now point the pointers in the right place
    for( int i = 0; i < nRows; ++i)
    {
        *(ppi + i) = curPtr;
         curPtr += nCols;
    }
    return ppi;
}

template < typename T >
void Free2DArray(T** Array)
{
    delete [] *Array;
    delete [] Array;
}