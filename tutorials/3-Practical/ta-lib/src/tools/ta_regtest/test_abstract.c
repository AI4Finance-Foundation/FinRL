/* TA-LIB Copyright (c) 1999-2007, Mario Fortier
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or
 * without modification, are permitted provided that the following
 * conditions are met:
 *
 * - Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in
 *   the documentation and/or other materials provided with the
 *   distribution.
 *
 * - Neither name of author nor the names of its contributors
 *   may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* List of contributors:
 *
 *  Initial  Name/description
 *  -------------------------------------------------------------------
 *  MF       Mario Fortier
 *  AC       Angelo Ciceri
 *
 *
 * Change history:
 *
 *  MMDDYY BY   Description
 *  -------------------------------------------------------------------
 *  112703 MF   First version.
 *  030104 MF   Add tests for TA_GetLookback
 *  062504 MF   Add test_default_calls.
 *  110206 AC   Change volume and open interest to double
 *  082607 MF   Add profiling feature.
 */

/* Description:
 *         Regression testing of the functionality provided
 *         by the ta_abstract module.
 *
 *         Also perform call to all functions for the purpose 
 *         of profiling (doExtensiveProfiling option).
 */

/**** Headers ****/
#ifdef WIN32
   #include "windows.h"
#else
   #include "time.h"
#endif

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include "ta_test_priv.h"

/**** External functions declarations. ****/
/* None */

/**** External variables declarations. ****/
extern int doExtensiveProfiling;

extern double gDataOpen[];
extern double gDataHigh[];
extern double gDataLow[];
extern double gDataClose[];

extern int nbProfiledCall;
extern double timeInProfiledCall;
extern double worstProfiledCall;
extern int insufficientClockPrecision;

/**** Global variables definitions.    ****/
/* None */

/**** Local declarations.              ****/
typedef enum 
{
	PROFILING_10000,
	PROFILING_8000,
	PROFILING_5000,
    PROFILING_2000,
	PROFILING_1000,
	PROFILING_500,
	PROFILING_100
} ProfilingType;

/**** Local functions declarations.    ****/
static ErrorNumber testLookback(TA_ParamHolder *paramHolder );
static ErrorNumber test_default_calls(void);
static ErrorNumber callWithDefaults( const char *funcName,
									 const double *input,
									 const int *input_int, int size );
static ErrorNumber callAndProfile( const char *funcName, ProfilingType type );

/**** Local variables definitions.     ****/
static double inputNegData[100];
static double inputZeroData[100];
static double inputRandFltEpsilon[100];
static double inputRandDblEpsilon[100];
static double inputRandomData[2000];

static int    inputNegData_int[100];
static int    inputZeroData_int[100];
static int    inputRandFltEpsilon_int[100];
static int    inputRandDblEpsilon_int[100];
static int    inputRandomData_int[2000];

static double output[10][2000];
static int    output_int[10][2000];

/**** Global functions definitions.   ****/
ErrorNumber test_abstract( void )
{
   ErrorNumber retValue;
   TA_RetCode retCode;
   TA_ParamHolder *paramHolder;
   const TA_FuncHandle *handle;
   int i;
   const char *xmlArray;

   printf( "Testing Abstract interface\n" );
   
   retValue = allocLib();
   if( retValue != TA_TEST_PASS )
      return retValue;    

   /* Verify TA_GetLookback. */
   retCode = TA_GetFuncHandle( "STOCH", &handle );
   if( retCode != TA_SUCCESS )
   {
      printf( "Can't get the function handle [%d]\n", retCode );
      return TA_ABS_TST_FAIL_GETFUNCHANDLE;   
   }
                             
   retCode = TA_ParamHolderAlloc( handle, &paramHolder );
   if( retCode != TA_SUCCESS )
   {
      printf( "Can't allocate the param holder [%d]\n", retCode );
      return TA_ABS_TST_FAIL_PARAMHOLDERALLOC;
   }

   retValue = testLookback(paramHolder);
   if( retValue != TA_SUCCESS )
   {
      printf( "testLookback() failed [%d]\n", retValue );
      TA_ParamHolderFree( paramHolder );
      return retValue;
   }      

   retCode = TA_ParamHolderFree( paramHolder );
   if( retCode != TA_SUCCESS )
   {
      printf( "TA_ParamHolderFree failed [%d]\n", retCode );
      return TA_ABS_TST_FAIL_PARAMHOLDERFREE;
   }

   retValue = freeLib();
   if( retValue != TA_TEST_PASS )
      return retValue;

   /* Call all the TA functions through the abstract interface. */
   retValue = allocLib();
   if( retValue != TA_TEST_PASS )
      return retValue;

   retValue = test_default_calls();
   if( retValue != TA_TEST_PASS )
   {
      printf( "TA-Abstract default call failed\n" );
      return retValue;
   }

   retValue = freeLib();
   if( retValue != TA_TEST_PASS )
      return retValue;
   
   /* Verify that the TA_FunctionDescription is null terminated
    * and as at least 500 characters (less is guaranteed bad...)
    */
   xmlArray = TA_FunctionDescriptionXML();
   for( i=0; i < 1000000; i++ )
   {
      if( xmlArray[i] == 0x0 )
         break;
   }

   if( i < 500) 
   {
      printf( "TA_FunctionDescriptionXML failed. Size too small.\n" );
      return TA_ABS_TST_FAIL_FUNCTION_DESC_SMALL;
   }

   if( i == 1000000 )
   {
      printf( "TA_FunctionDescriptionXML failed. Size too large (missing null?).\n" );
      return TA_ABS_TST_FAIL_FUNCTION_DESC_LARGE;
   }

   return TA_TEST_PASS; /* Succcess. */
}

/**** Local functions definitions.     ****/
static ErrorNumber testLookback( TA_ParamHolder *paramHolder )
{
  TA_RetCode retCode;
  int lookback;

  /* Change the parameters of STOCH and verify that TA_GetLookback respond correctly. */
  retCode = TA_SetOptInputParamInteger( paramHolder, 0, 3 );
  if( retCode != TA_SUCCESS )
  {
     printf( "TA_SetOptInputParamInteger call failed [%d]\n", retCode );
     return TA_ABS_TST_FAIL_OPTINPUTPARAMINTEGER;
  }

  retCode = TA_SetOptInputParamInteger( paramHolder, 1, 4 );
  if( retCode != TA_SUCCESS )
  {
     printf( "TA_SetOptInputParamInteger call failed [%d]\n", retCode );
     return TA_ABS_TST_FAIL_OPTINPUTPARAMINTEGER;
  }

  retCode = TA_SetOptInputParamInteger( paramHolder, 2, (TA_Integer)TA_MAType_SMA );
  if( retCode != TA_SUCCESS )
  {
     printf( "TA_SetOptInputParamInteger call failed [%d]\n", retCode );
     return TA_ABS_TST_FAIL_OPTINPUTPARAMINTEGER;
  }

  retCode = TA_SetOptInputParamInteger( paramHolder, 3, 4 );
  if( retCode != TA_SUCCESS )
  {
     printf( "TA_SetOptInputParamInteger call failed [%d]\n", retCode );
     return TA_ABS_TST_FAIL_OPTINPUTPARAMINTEGER;
  }

  retCode = TA_SetOptInputParamInteger( paramHolder, 4, (TA_Integer)TA_MAType_SMA );
  if( retCode != TA_SUCCESS )
  {
     printf( "TA_SetOptInputParamInteger call failed [%d]\n", retCode );
     return TA_ABS_TST_FAIL_OPTINPUTPARAMINTEGER;
  }

  retCode = TA_GetLookback(paramHolder,&lookback);
  if( retCode != TA_SUCCESS )
  {
     printf( "TA_GetLookback failed [%d]\n", retCode );
     return TA_ABS_TST_FAIL_GETLOOKBACK_CALL_1;
  }

  if( lookback != 8 )
  {
     printf( "TA_GetLookback failed [%d != 8]\n", lookback );
     return TA_ABS_TST_FAIL_GETLOOKBACK_1;
  }

  /* Change one parameter and check again. */
  retCode = TA_SetOptInputParamInteger( paramHolder, 3, 3 );
  if( retCode != TA_SUCCESS )
  {
     printf( "TA_SetOptInputParamInteger call failed [%d]\n", retCode );
     return TA_ABS_TST_FAIL_OPTINPUTPARAMINTEGER;
  }

  retCode = TA_GetLookback(paramHolder,&lookback);
  if( retCode != TA_SUCCESS )
  {
     printf( "TA_GetLookback failed [%d]\n", retCode );
     return TA_ABS_TST_FAIL_GETLOOKBACK_CALL_2;
  }

  if( lookback != 7 )
  {
     printf( "TA_GetLookback failed [%d != 7]\n", lookback );
     return TA_ABS_TST_FAIL_GETLOOKBACK_2;
  }
  
  return TA_TEST_PASS;
}


static void testDefault( const TA_FuncInfo *funcInfo, void *opaqueData )
{
	static int nbFunctionDone = 0;
   ErrorNumber *errorNumber;
   errorNumber = (ErrorNumber *)opaqueData;
   if( *errorNumber != TA_TEST_PASS )
      return;

#define CALL(x) { \
	*errorNumber = callWithDefaults( funcInfo->name, x, x##_int, sizeof(x)/sizeof(double) ); \
	if( *errorNumber != TA_TEST_PASS ) { \
	   printf( "Failed for [%s][%s]\n", funcInfo->name, #x ); \
       return; \
	} \
}
   /* Do not test value outside the ]0..1[ domain for the "Math" groups. */
   if( (strlen(funcInfo->group) < 4) || 
	   !((tolower(funcInfo->group[0]) == 'm') && 
	     (tolower(funcInfo->group[1]) == 'a') &&
	     (tolower(funcInfo->group[2]) == 't') &&
	     (tolower(funcInfo->group[3]) == 'h')))
   {	   
      CALL( inputNegData );
      CALL( inputZeroData );
      CALL( inputRandFltEpsilon );
      CALL( inputRandDblEpsilon );
   }

   CALL( inputRandomData );

#undef CALL

#define CALL(x) { \
	*errorNumber = callAndProfile( funcInfo->name, x ); \
	if( *errorNumber != TA_TEST_PASS ) { \
	   printf( "Failed for [%s][%s]\n", funcInfo->name, #x ); \
       return; \
	} \
}
   if( doExtensiveProfiling /*&& (nbFunctionDone<5)*/ )
   {
	   nbFunctionDone++;
	   printf( "%s ", funcInfo->name );
       CALL( PROFILING_100 );
       CALL( PROFILING_500 );
	   CALL( PROFILING_1000 );
       CALL( PROFILING_2000 );
       CALL( PROFILING_5000 );
       CALL( PROFILING_8000 );
	   CALL( PROFILING_10000 );
	   printf( "\n" );
   }
}

static ErrorNumber callWithDefaults( const char *funcName, const double *input, const int *input_int, int size )
{
   TA_ParamHolder *paramHolder;
   const TA_FuncHandle *handle;
   const TA_FuncInfo *funcInfo;
   const TA_InputParameterInfo *inputInfo;
   const TA_OutputParameterInfo *outputInfo;

   TA_RetCode retCode;
   unsigned int i;
   int j;
   int outBegIdx, outNbElement, lookback;

   retCode = TA_GetFuncHandle( funcName, &handle );
   if( retCode != TA_SUCCESS )
   {
      printf( "Can't get the function handle [%d]\n", retCode );
      return TA_ABS_TST_FAIL_GETFUNCHANDLE;   
   }
                             
   retCode = TA_ParamHolderAlloc( handle, &paramHolder );
   if( retCode != TA_SUCCESS )
   {
      printf( "Can't allocate the param holder [%d]\n", retCode );
      return TA_ABS_TST_FAIL_PARAMHOLDERALLOC;
   }

   TA_GetFuncInfo( handle, &funcInfo );

   for( i=0; i < funcInfo->nbInput; i++ )
   {
      TA_GetInputParameterInfo( handle, i, &inputInfo );
	  switch(inputInfo->type)
	  {
	  case TA_Input_Price:
         TA_SetInputParamPricePtr( paramHolder, i,
			 inputInfo->flags&TA_IN_PRICE_OPEN?input:NULL,
			 inputInfo->flags&TA_IN_PRICE_HIGH?input:NULL,
			 inputInfo->flags&TA_IN_PRICE_LOW?input:NULL,
			 inputInfo->flags&TA_IN_PRICE_CLOSE?input:NULL,
			 inputInfo->flags&TA_IN_PRICE_VOLUME?input:NULL, NULL );
		 break;
	  case TA_Input_Real:
         TA_SetInputParamRealPtr( paramHolder, i, input );
		 break;
	  case TA_Input_Integer:
         TA_SetInputParamIntegerPtr( paramHolder, i, input_int );
         break;
	  }
   }

   for( i=0; i < funcInfo->nbOutput; i++ )
   {
      TA_GetOutputParameterInfo( handle, i, &outputInfo );
	  switch(outputInfo->type)
	  {
	  case TA_Output_Real:
	     TA_SetOutputParamRealPtr(paramHolder,i,&output[i][0]);         
         for( j=0; j < 2000; j++ )
            output[i][j] = TA_REAL_MIN;
		 break;
	  case TA_Output_Integer:
	     TA_SetOutputParamIntegerPtr(paramHolder,i,&output_int[i][0]);
         for( j=0; j < 2000; j++ )
            output_int[i][j] = TA_INTEGER_MIN;
		 break;
	  }
   }

   /* Do the function call. */
   retCode = TA_CallFunc(paramHolder,0,size-1,&outBegIdx,&outNbElement);
   if( retCode != TA_SUCCESS )
   {
      printf( "TA_CallFunc() failed zero data test [%d]\n", retCode );
      TA_ParamHolderFree( paramHolder );
      return TA_ABS_TST_FAIL_CALLFUNC_1;
   }      

   /* Verify consistency with Lookback */
   retCode = TA_GetLookback( paramHolder, &lookback );
   if( retCode != TA_SUCCESS )
   {
      printf( "TA_GetLookback() failed zero data test [%d]\n", retCode );
      TA_ParamHolderFree( paramHolder );
      return TA_ABS_TST_FAIL_CALLFUNC_2;
   }
   
   if( outBegIdx != lookback )
   {
      printf( "TA_GetLookback() != outBegIdx [%d != %d]\n", lookback, outBegIdx );
      TA_ParamHolderFree( paramHolder );
      return TA_ABS_TST_FAIL_CALLFUNC_3;
   }                       

   /* TODO Add back nan/inf tests. 
   for( i=0; i < funcInfo->nbOutput; i++ )
   {
	  switch(outputInfo->type)
	  {
	  case TA_Output_Real:	     
		for( j=0; j < outNbElement; j++ )
		{
			if( trio_isnan(output[i][j]) ||
                trio_isinf(output[i][j]))
			{
				printf( "Failed for output[%d][%d] = %e\n", i, j, output[i][j] );
				return TA_ABS_TST_FAIL_INVALID_OUTPUT;
			}
		}
		break;
	  case TA_Output_Integer:	     
		break;
	  }
   }*/

   /* Do another function call where startIdx == endIdx == 0.
    * In that case, outBegIdx should ALWAYS be zero.
    */
   retCode = TA_CallFunc(paramHolder,0,0,&outBegIdx,&outNbElement);
   if( retCode != TA_SUCCESS )
   {
      printf( "TA_CallFunc() failed data test 4 [%d]\n", retCode );
      TA_ParamHolderFree( paramHolder );
      return TA_ABS_TST_FAIL_CALLFUNC_4;
   }

   if( outBegIdx != 0 )
   {
      printf( "failed outBegIdx=%d when startIdx==endIdx==0\n", outBegIdx );
      TA_ParamHolderFree( paramHolder );
      return TA_ABS_TST_FAIL_STARTEND_ZERO;
   }

   retCode = TA_ParamHolderFree( paramHolder );
   if( retCode != TA_SUCCESS )
   {
      printf( "TA_ParamHolderFree failed [%d]\n", retCode );
      return TA_ABS_TST_FAIL_PARAMHOLDERFREE;
   }

   return TA_TEST_PASS;
}

static ErrorNumber test_default_calls(void)
{
   ErrorNumber errNumber;
   unsigned int i;
   unsigned int sign;
   double tempDouble;

   errNumber = TA_TEST_PASS;

   for( i=0; i < sizeof(inputNegData)/sizeof(double); i++ )
   {
      inputNegData[i] = -((double)((int)i));
	  inputNegData_int[i] = -(int)i;
   }

   for( i=0; i < sizeof(inputZeroData)/sizeof(double); i++ )
   {
      inputZeroData[i] = 0.0;
	  inputZeroData_int[i] = (int)inputZeroData[i];
   }

   for( i=0; i < sizeof(inputRandomData)/sizeof(double); i++ )
   {
      /* Make 100% sure input range is ]0..1[ */
	  tempDouble = (double)rand() / ((double)(RAND_MAX)+(double)(1));
      while( (tempDouble <= 0.0) || (tempDouble >= 1.0) ) 
	  {
		  tempDouble = (double)rand() / ((double)(RAND_MAX)+(double)(1));
	  }
      inputRandomData[i] = tempDouble;
      inputRandomData_int[i] = (int)inputRandomData[i];
   }

   for( i=0; i < sizeof(inputRandFltEpsilon)/sizeof(double); i++ )
   {
       sign= (unsigned int)rand()%2;
       inputRandFltEpsilon[i] = (sign?1.0:-1.0)*(FLT_EPSILON);
       inputRandFltEpsilon_int[i] = sign?TA_INTEGER_MIN:TA_INTEGER_MAX;
   }

   for( i=0; i < sizeof(inputRandFltEpsilon)/sizeof(double); i++ )
   {
       sign= (unsigned int)rand()%2;
       inputRandFltEpsilon[i] = (sign?1.0:-1.0)*(DBL_EPSILON);
       inputRandFltEpsilon_int[i] = sign?1:-1;
   }

   if( doExtensiveProfiling )
   {
		   printf( "\n[PROFILING START]\n" );
   }

   TA_ForEachFunc( testDefault, &errNumber );

   if( doExtensiveProfiling )
   {
		   printf( "[PROFILING END]\n" );
   }
   

   return errNumber;
}

static ErrorNumber callAndProfile( const char *funcName, ProfilingType type )
{
   TA_ParamHolder *paramHolder;
   const TA_FuncHandle *handle;
   const TA_FuncInfo *funcInfo;
   const TA_InputParameterInfo *inputInfo;
   const TA_OutputParameterInfo *outputInfo;

   TA_RetCode retCode;
   int h, i, j, k;   
   int outBegIdx, outNbElement;

   /* Variables to control iteration and corresponding input size */
   int nbInnerLoop, nbOuterLoop;
   int stepSize;
   int inputSize;

   /* Variables measuring the execution time */
#ifdef WIN32
   LARGE_INTEGER startClock;
   LARGE_INTEGER endClock;
#else
   clock_t startClock;
   clock_t endClock;
#endif
   double clockDelta;
   int nbProfiledCallLocal;
   double timeInProfiledCallLocal;
   double worstProfiledCallLocal;

   nbProfiledCallLocal = 0;
   timeInProfiledCallLocal = 0.0;
   worstProfiledCallLocal = 0.0;
   nbInnerLoop = nbOuterLoop = stepSize = inputSize = 0;

   switch( type )
   {
   case PROFILING_10000:
	   nbInnerLoop = 1;
	   nbOuterLoop = 100;
	   stepSize = 10000;
	   inputSize = 10000;
	   break;
   case PROFILING_8000:
	   nbInnerLoop = 2;
	   nbOuterLoop = 50;
	   stepSize = 2000;
	   inputSize = 8000;
       break;
   case PROFILING_5000:
	   nbInnerLoop = 2;
	   nbOuterLoop = 50;
	   stepSize = 5000;
	   inputSize = 5000;
	   break;
   case PROFILING_2000:
	   nbInnerLoop = 5;
	   nbOuterLoop = 20;
	   stepSize = 2000;
	   inputSize = 2000;
	   break;
   case PROFILING_1000:
	   nbInnerLoop = 10;
	   nbOuterLoop = 10;
	   stepSize = 1000;
	   inputSize = 1000;
	   break;
   case PROFILING_500:
	   nbInnerLoop = 20;
	   nbOuterLoop = 5;
	   stepSize = 500;
	   inputSize = 500;
	   break;
   case PROFILING_100:
	   nbInnerLoop = 100;
	   nbOuterLoop = 1;
	   stepSize = 100;
	   inputSize = 100;
	   break;
   }

   retCode = TA_GetFuncHandle( funcName, &handle );
   if( retCode != TA_SUCCESS )
   {
      printf( "Can't get the function handle [%d]\n", retCode );
      return TA_ABS_TST_FAIL_GETFUNCHANDLE;   
   }
                             
   retCode = TA_ParamHolderAlloc( handle, &paramHolder );
   if( retCode != TA_SUCCESS )
   {
      printf( "Can't allocate the param holder [%d]\n", retCode );
      return TA_ABS_TST_FAIL_PARAMHOLDERALLOC;
   }

   TA_GetFuncInfo( handle, &funcInfo );

   for( i=0; i < (int)funcInfo->nbOutput; i++ )
   {
      TA_GetOutputParameterInfo( handle, i, &outputInfo );
	  switch(outputInfo->type)
	  {
	  case TA_Output_Real:
	     TA_SetOutputParamRealPtr(paramHolder,i,&output[i][0]);         
         for( j=0; j < 2000; j++ )
            output[i][j] = TA_REAL_MIN;
		 break;
	  case TA_Output_Integer:
	     TA_SetOutputParamIntegerPtr(paramHolder,i,&output_int[i][0]);
         for( j=0; j < 2000; j++ )
            output_int[i][j] = TA_INTEGER_MIN;
		 break;
	  }
   }

   for( h=0; h < 2; h++ )
   {
   for( i=0; i < nbOuterLoop; i++ )
   {
	   for( j=0; j < nbInnerLoop; j++ )
	   {
		   /* Prepare input. */
		   for( k=0; k < (int)funcInfo->nbInput; k++ )
		   {
			  TA_GetInputParameterInfo( handle, k, &inputInfo );
			  switch(inputInfo->type)
			  {
			  case TA_Input_Price:
				 TA_SetInputParamPricePtr( paramHolder, k,
					 inputInfo->flags&TA_IN_PRICE_OPEN?&gDataOpen[j*stepSize]:NULL,
					 inputInfo->flags&TA_IN_PRICE_HIGH?&gDataHigh[j*stepSize]:NULL,
					 inputInfo->flags&TA_IN_PRICE_LOW?&gDataLow[j*stepSize]:NULL,
					 inputInfo->flags&TA_IN_PRICE_CLOSE?&gDataClose[j*stepSize]:NULL,
					 inputInfo->flags&TA_IN_PRICE_VOLUME?&gDataClose[j*stepSize]:NULL, NULL );
				 break;
			  case TA_Input_Real:
				 TA_SetInputParamRealPtr( paramHolder, k, &gDataClose[j*stepSize] );
				 break;
			  case TA_Input_Integer:
				 printf( "\nError: Integer input not yet supported for profiling.\n" );
				 return TA_ABS_TST_FAIL_CALLFUNC_1;
				 break;
			  }
		   }

           #ifdef WIN32
              QueryPerformanceCounter(&startClock);
           #else
              startClock = clock();
           #endif

		   /* Do the function call. */
		   retCode = TA_CallFunc(paramHolder,0,inputSize-1,&outBegIdx,&outNbElement);
		   if( retCode != TA_SUCCESS )
		   {
		      printf( "TA_CallFunc() failed zero data test [%d]\n", retCode );
		      TA_ParamHolderFree( paramHolder );
		      return TA_ABS_TST_FAIL_CALLFUNC_1;
		   }

		   #ifdef WIN32
			   QueryPerformanceCounter(&endClock);
			   clockDelta = (double)((__int64)endClock.QuadPart - (__int64) startClock.QuadPart);
		   #else
			   endClock = clock();
			   clockDelta = (double)(endClock - startClock);
		   #endif

		   /* Setup global profiling info. */
		   if( clockDelta <= 0 )
		   {
			   printf( "Error: Insufficient timer precision to perform benchmarking on this platform.\n" );
			   return TA_ABS_TST_FAIL_CALLFUNC_1;
		   }
		   else
		   {	   
			   if( clockDelta > worstProfiledCall )
			      worstProfiledCall = clockDelta;
			   timeInProfiledCall += clockDelta;
			   nbProfiledCall++;
		   }

		   /* Setup local profiling info for this particular function. */
		   if( clockDelta > worstProfiledCallLocal )
			   worstProfiledCallLocal = clockDelta;
		   timeInProfiledCallLocal += clockDelta;
		   nbProfiledCallLocal++;
	   }
   }
   }

   /* Output statistic (remove worst call, average the others. */
   printf( "%g ", (timeInProfiledCallLocal-worstProfiledCallLocal)/(double)(nbProfiledCallLocal-1));

   retCode = TA_ParamHolderFree( paramHolder );
   if( retCode != TA_SUCCESS )
   {
      printf( "TA_ParamHolderFree failed [%d]\n", retCode );
      return TA_ABS_TST_FAIL_PARAMHOLDERFREE;
   }

   return TA_TEST_PASS;
}
