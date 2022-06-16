#ifndef TA_MEMORY_H
#define TA_MEMORY_H

#if !defined( _MANAGED ) && !defined( _JAVA )
   #ifndef TA_COMMON_H
      #include "ta_common.h"
   #endif

   #include <stdlib.h> 

   /* Interface macros */
   #define TA_Malloc(a)       malloc(a)
   #define TA_Realloc(a,b)    realloc((a),(b))
   #define TA_Free(a)         free(a)

   #define FREE_IF_NOT_NULL(x) { if((x)!=NULL) {TA_Free((void *)(x)); (x)=NULL;} }

#endif /* !defined(_MANAGED) && !defined( _JAVA ) */


/* ARRAY : Macros to manipulate arrays of value type.
 *
 * Using temporary array of double and integer are often needed for the 
 * TA functions.
 *
 * These macros allow basic operations to alloc/copy/free array of value type.
 *
 * These macros works in plain old C/C++, managed C++.and Java.
 * 
 * (Use ARRAY_REF and ARRAY_INT_REF for double/integer arrays).
 */
#if defined( _MANAGED )
   #define ARRAY_VTYPE_REF(type,name)             cli::array<type>^ name
   #define ARRAY_VTYPE_LOCAL(type,name,size)      cli::array<type>^ name = gcnew cli::array<type>(size)
   #define ARRAY_VTYPE_ALLOC(type,name,size)      name = gcnew cli::array<type>(size)
   #define ARRAY_VTYPE_COPY(type,dest,src,size)   cli::array<type>::Copy( src, 0, dest, 0, size )
   #define ARRAY_VTYPE_MEMMOVE(type,dest,destIdx,src,srcIdx,size) cli::array<type>::Copy( src, srcIdx, dest, destIdx, size )
   #define ARRAY_VTYPE_FREE(type,name)
   #define ARRAY_VTYPE_FREE_COND(type,cond,name)   
#elif defined( _JAVA )
   #define ARRAY_VTYPE_REF(type,name)             type []name
   #define ARRAY_VTYPE_LOCAL(type,name,size)      type []name = new type[size]
   #define ARRAY_VTYPE_ALLOC(type,name,size)      name = new type[size]
   #define ARRAY_VTYPE_COPY(type,dest,src,size)   System.arraycopy(src,0,dest,0,size)
   #define ARRAY_VTYPE_MEMMOVE(type,dest,destIdx,src,srcIdx,size) System.arraycopy(src,srcIdx,dest,destIdx,size)
   #define ARRAY_VTYPE_FREE(type,name)
   #define ARRAY_VTYPE_FREE_COND(type,cond,name)
#else
   #define ARRAY_VTYPE_REF(type,name)             type *name
   #define ARRAY_VTYPE_LOCAL(type,name,size)      type name[size]
   #define ARRAY_VTYPE_ALLOC(type,name,size)      name = (type *)TA_Malloc( sizeof(type)*(size))
   #define ARRAY_VTYPE_COPY(type,dest,src,size)   memcpy(dest,src,sizeof(type)*(size))
   #define ARRAY_VTYPE_MEMMOVE(type,dest,destIdx,src,srcIdx,size) memmove( &dest[destIdx], &src[srcIdx], (size)*sizeof(type) )
   #define ARRAY_VTYPE_FREE(type,name)            TA_Free(name)
   #define ARRAY_VTYPE_FREE_COND(type,cond,name)  if( cond ){ TA_Free(name); }
#endif

/* ARRAY : Macros to manipulate arrays of double. */
#define ARRAY_REF(name)             ARRAY_VTYPE_REF(double,name)
#define ARRAY_LOCAL(name,size)      ARRAY_VTYPE_LOCAL(double,name,size)
#define ARRAY_ALLOC(name,size)      ARRAY_VTYPE_ALLOC(double,name,size)
#define ARRAY_COPY(dest,src,size)   ARRAY_VTYPE_COPY(double,dest,src,size)
#define ARRAY_MEMMOVE(dest,destIdx,src,srcIdx,size) ARRAY_VTYPE_MEMMOVE(double,dest,destIdx,src,srcIdx,size)
#define ARRAY_FREE(name)            ARRAY_VTYPE_FREE(double,name)
#define ARRAY_FREE_COND(cond,name)  ARRAY_VTYPE_FREE_COND(double,cond,name)

/* ARRAY : Macros to manipulate arrays of integer. */
#define ARRAY_INT_REF(name)             ARRAY_VTYPE_REF(int,name)
#define ARRAY_INT_LOCAL(name,size)      ARRAY_VTYPE_LOCAL(int,name,size)
#define ARRAY_INT_ALLOC(name,size)      ARRAY_VTYPE_ALLOC(int,name,size)
#define ARRAY_INT_COPY(dest,src,size)   ARRAY_VTYPE_COPY(int,dest,src,size)
#define ARRAY_INT_MEMMOVE(dest,destIdx,src,srcIdx,size) ARRAY_VTYPE_MEMMOVE(int,dest,destIdx,src,srcIdx,size)
#define ARRAY_INT_FREE(name)            ARRAY_VTYPE_FREE(int,name)
#define ARRAY_INT_FREE_COND(cond,name)  ARRAY_VTYPE_FREE_COND(int,cond,name)

/* Access to "Globals"
 *
 * The globals here just means that these variables are accessible from
 * all technical analysis functions.
 *
 * Depending of the language/platform, the globals might be in reality
 * a private member variable of an object...
 */
#if defined( _MANAGED )
   #define TA_GLOBALS_UNSTABLE_PERIOD(x,y) (Globals->unstablePeriod[(int)(FuncUnstId::y)])
   #define TA_GLOBALS_COMPATIBILITY        (Globals->compatibility)
#elif defined( _JAVA )
   #define TA_GLOBALS_UNSTABLE_PERIOD(x,y) (this.unstablePeriod[FuncUnstId.y.ordinal()])
   #define TA_GLOBALS_COMPATIBILITY        (this.compatibility)
#else
   #define TA_GLOBALS_UNSTABLE_PERIOD(x,y) (TA_Globals->unstablePeriod[x])
   #define TA_GLOBALS_COMPATIBILITY        (TA_Globals->compatibility)
#endif



/* CIRCBUF : Circular Buffer Macros.
 *
 * The CIRCBUF is like a FIFO buffer (First In - First Out), except
 * that the rate of data coming out is the same as the rate of
 * data coming in (for simplification and speed optimization).
 * In other word, when you add one new value, you must also consume
 * one value (if not consume, the value is lost).
 *
 * The CIRCBUF size is unlimited, so it will automatically allocate and
 * de-allocate memory as needed. In C/C++. when small enough, CIRCBUF will 
 * instead use a buffer "allocated" on the stack (automatic variable).
 * 
 * Multiple CIRCBUF can be used within the same function. To make that
 * possible the first parameter of the MACRO is an "Id" that can be
 * any string.
 *
 * The macros offer the advantage to work in C/C++ and managed C++.
 * 
 * CIRCBUF_PROLOG(Id,Type,Size);
 *          Will declare all the needed variables. 2 variables are
 *          important: 
 *                 1) 'Id' will be a ptr of the specified Type.
 *                 2) 'Id'_Idx indicates from where to consume and 
 *                     to add the data.
 *
 *          Important: You must consume the oldest data before
 *                     setting the new data!
 *
 *          The Size must be reasonable since it might "allocate"
 *          an array of this size on the stack (each element are 'Type').
 *
 * CIRCBUF_CONSTRUCT(Id,Type,Size);
 *         Must be called prior to use the remaining macros. Must be
 *         followed by CIRCBUF_DESTROY when leaving the function.
 *         The Size here can be large. If the static Size specified
 *         with CIRCBUF_PROLOG is not sufficient, this MACRO will
 *         allocate a new buffer from the Heap.
 *
 * CIRCBUF_DESTROY(Id,Size);
 *         Must be call prior to leave the function.
 *
 * CIRCBUF_NEXT(Id);
 *         Move forward the indexes.
 *
 * Example:
 *     TA_RetCode MyFunc( int size )
 *     {
 *        CIRCBUF_PROLOG(MyBuf,int,4);
 *        int i, value;
 *        ...
 *        CIRCBUF_CONSTRUCT(MyBuf,int,size);
 *        ...
 *        // 1st Loop: Fill MyBuf with initial values
 *        //           (must be done).
 *        value = 0;
 *        for( i=0; i < size; i++ )
 *        {
 *           // Set the data
 *           MyBuf[MyBuf_Idx] = value++;
 *           CIRCBUF_NEXT(MyBuf);
 *        }
 *
 *        // 2nd Loop: Get and Add subsequent values
 *        //           in MyBuf (optional)
 *        for( i=0; i < 3; i++ )
 *        {
 *           // Consume the data (must be done first)
 *           printf( "%d ", MyBuf[MyBuf_Idx] );
 *
 *           // Set the new data (must be done second)
 *           MyBuf[MyBuf_Idx] = value++;
 *
 *           // Move the index forward
 *           CIRCBUF_NEXT(MyBuf);
 *        }
 *
 *        // 3rd Loop: Empty MyBuf (optional)
 *        for( i=0; i < size; i++ )
 *        {
 *           printf( "%d ", MyBuf[MyBuf_Idx] );
 *           CIRCBUF_NEXT(MyBuf);
 *        }
 *
 *        CIRCBUF_DESTROY(MyBuf);
 *        return TA_SUCCESS;
 *     }
 *
 *
 * A call to MyFunc(5) will output:
 *    0 1 2 3 4 5 6 7
 *
 * The value 0 to 4 are added by the 1st loop.
 * The value 5 to 7 are added by the 2nd loop.
 *
 * The value 0 to 2 are displayed by the 2nd loop.
 * The value 3 to 7 are displayed by the 3rd loop.
 *
 * Because the size 5 is greater than the 
 * value provided in CIRCBUF_PROLOG, a buffer will
 * be dynamically allocated (and freed).
 */
#if defined( _MANAGED )

#define CIRCBUF_PROLOG(Id,Type,Size) int Id##_Idx = 0; \
                                     cli::array<Type>^ Id; \
                                     int maxIdx_##Id = (Size-1)

/* Use this macro instead if the Type is a class or a struct. */
#define CIRCBUF_PROLOG_CLASS(Id,Type,Size) int Id##_Idx = 0; \
                                           cli::array<Type^>^ Id; \
                                           int maxIdx_##Id = (Size-1)

#define CIRCBUF_INIT(Id,Type,Size) \
   { \
      if( Size <= 0 ) \
         return ENUM_VALUE(RetCode,TA_ALLOC_ERR,AllocErr); \
      Id = gcnew cli::array<Type>(Size); \
      if( !Id ) \
         return ENUM_VALUE(RetCode,TA_ALLOC_ERR,AllocErr); \
      maxIdx_##Id = (Size-1); \
   }

#define CIRCBUF_INIT_CLASS(Id,Type,Size) \
   { \
      if( Size <= 0 ) \
         return ENUM_VALUE(RetCode,TA_ALLOC_ERR,AllocErr); \
      Id = gcnew cli::array<Type^>(Size); \
      for( int _##Id##_index=0; _##Id##_index<Id->Length; _##Id##_index++) \
      { \
         Id[_##Id##_index]=gcnew Type(); \
      } \
      if( !Id ) \
         return ENUM_VALUE(RetCode,TA_ALLOC_ERR,AllocErr); \
      maxIdx_##Id = (Size-1); \
   }

#define CIRCBUF_INIT_LOCAL_ONLY(Id,Type) \
   { \
      Id = gcnew cli::array<Type>(maxIdx_##Id+1); \
      if( !Id ) \
         return ENUM_VALUE(RetCode,TA_ALLOC_ERR,AllocErr); \
   }

#define CIRCBUF_DESTROY(Id)

/* Use this macro to access the member when type is a class or a struct. */
#define CIRCBUF_REF(x) (x)->

#elif defined(_JAVA)

#define CIRCBUF_PROLOG(Id,Type,Size) int Id##_Idx = 0; \
                                     Type []Id; \
                                     int maxIdx_##Id = (Size-1)

/* Use this macro instead if the Type is a class or a struct. */
#define CIRCBUF_PROLOG_CLASS(Id,Type,Size) int Id##_Idx = 0; \
                                           Type []Id; \
                                           int maxIdx_##Id = (Size-1)

#define CIRCBUF_INIT(Id,Type,Size) \
   { \
      if( Size <= 0 ) \
         return ENUM_VALUE(RetCode,TA_ALLOC_ERR,AllocErr); \
      Id = new Type[Size]; \
      maxIdx_##Id = (Size-1); \
   }

#define CIRCBUF_INIT_CLASS(Id,Type,Size) \
   { \
      if( Size <= 0 ) \
         return ENUM_VALUE(RetCode,TA_ALLOC_ERR,AllocErr); \
      Id = new Type[Size]; \
      for( int _##Id##_index=0; _##Id##_index<Id.length; _##Id##_index++) \
      { \
         Id[_##Id##_index]=new Type(); \
      } \
      maxIdx_##Id = (Size-1); \
   }

#define CIRCBUF_INIT_LOCAL_ONLY(Id,Type) \
   { \
      Id = new Type[maxIdx_##Id+1]; \
   }

#define CIRCBUF_DESTROY(Id)

/* Use this macro to access the member when type is a class or a struct. */
#define CIRCBUF_REF(x) (x).

#else

#define CIRCBUF_PROLOG(Id,Type,Size) Type local_##Id[Size]; \
                                  int Id##_Idx; \
                                  Type *Id; \
                                  int maxIdx_##Id

/* Use this macro instead if the Type is a class or a struct. */
#define CIRCBUF_PROLOG_CLASS(Id,Type,Size) CIRCBUF_PROLOG(Id,Type,Size)

#define CIRCBUF_INIT(Id,Type,Size) \
   { \
      if( Size < 1 ) \
         return TA_INTERNAL_ERROR(137); \
      if( (int)Size > (int)(sizeof(local_##Id)/sizeof(Type)) ) \
      { \
         Id = TA_Malloc( sizeof(Type)*Size ); \
         if( !Id ) \
            return TA_ALLOC_ERR; \
      } \
      else \
         Id = &local_##Id[0]; \
      maxIdx_##Id = (Size-1); \
      Id##_Idx = 0; \
   }

#define CIRCBUF_INIT_CLASS(Id,Type,Size) CIRCBUF_INIT(Id,Type,Size)

#define CIRCBUF_INIT_LOCAL_ONLY(Id,Type) \
   { \
      Id = &local_##Id[0]; \
      maxIdx_##Id = (int)(sizeof(local_##Id)/sizeof(Type))-1; \
      Id##_Idx = 0; \
   }

#define CIRCBUF_DESTROY(Id) \
   { \
      if( Id != &local_##Id[0] ) \
         TA_Free( Id ); \
   }

/* Use this macro to access the member when Type is a class or a struct. */
#define CIRCBUF_REF(x) (x).

#endif

#define CIRCBUF_NEXT(Id) \
   { \
      Id##_Idx++; \
      if( Id##_Idx > maxIdx_##Id ) \
         Id##_Idx = 0; \
   }


#endif

