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
 *
 *
 * Change history:
 *
 *  MMDDYY BY   Description
 *  -------------------------------------------------------------------
 *  112400 MF   First version.
 *  122506 MF   Add tests for MININDEX,MAXINDEX,MINMAX and MINMAXINDEX.
 */

/* Description:
 *     Test the min/max related functions.
 *
 */

/**** Headers ****/
#include <stdio.h>
#include <string.h>

#include "ta_test_priv.h"
#include "ta_test_func.h"
#include "ta_utility.h"
#include "ta_memory.h"

/**** External functions declarations. ****/
/* None */

/**** External variables declarations. ****/
/* None */

/**** Global variables definitions.    ****/
/* None */

/**** Local declarations.              ****/
typedef enum {
TA_MIN_TEST,
TA_MAX_TEST,
TA_MINMAX_TEST,
TA_MININDEX_TEST,
TA_MAXINDEX_TEST,
TA_MINMAXINDEX_TEST
} TA_TestId;

typedef struct
{
   TA_Integer doRangeTestFlag;

   TA_TestId  theFunction;

   TA_Integer startIdx;
   TA_Integer endIdx;

   TA_Integer optInTimePeriod;
 
   TA_RetCode expectedRetCode;

   TA_Integer oneOfTheExpectedOutRealIndex0;
   TA_Real    oneOfTheExpectedOutReal0;

   TA_Integer expectedBegIdx;
   TA_Integer expectedNbElement;
} TA_Test;

typedef struct
{
   const TA_Real *input;
   unsigned int nbElement;
} TA_RefTest;

typedef struct
{
   const TA_Test *test;
   const TA_Real *close;
} TA_RangeTestParam;

/**** Local functions declarations.    ****/
static ErrorNumber do_test( const TA_History *history,
                            const TA_Test *test );

static TA_RetCode referenceMin( TA_Integer    startIdx,
                                TA_Integer    endIdx,
                                const TA_Real inReal[],
                                TA_Integer    optInTimePeriod,
                                TA_Integer   *outBegIdx,
                                TA_Integer   *outNbElement,
                                TA_Real       outReal[] );

static TA_RetCode referenceMax( TA_Integer    startIdx,
                                TA_Integer    endIdx,
                                const TA_Real inReal[],
                                TA_Integer    optInTimePeriod,
                                TA_Integer   *outBegIdx,
                                TA_Integer   *outNbElement,
                                TA_Real       outReal[] );

static ErrorNumber testCompareToReference( const TA_Real *input, int nbElement );

/**** Local variables definitions.     ****/

static TA_Test tableTest[] =
{
   /**********************/
   /*      MIN TEST      */
   /**********************/

   /* No output value. */
   { 0, TA_MIN_TEST, 1, 1,  14, TA_SUCCESS, 0, 0, 0, 0},

   /* One value tests. */
   { 0, TA_MIN_TEST, 14,  14, 14, TA_SUCCESS, 0, 91.125,  14, 1},

   /* Index too low test. */
   { 0, TA_MIN_TEST, 0,  15, 14, TA_SUCCESS, 0, 91.125,     13, 3},
   { 0, TA_MIN_TEST, 1,  15, 14, TA_SUCCESS, 0, 91.125,     13, 3},
   { 0, TA_MIN_TEST, 2,  16, 14, TA_SUCCESS, 0, 91.125,     13, 4},
   { 0, TA_MIN_TEST, 2,  16, 14, TA_SUCCESS, 1, 91.125,     13, 4},
   { 0, TA_MIN_TEST, 2,  16, 14, TA_SUCCESS, 2, 91.125,     13, 4},
   { 0, TA_MIN_TEST, 0,  14, 14, TA_SUCCESS, 0, 91.125,     13, 2},
   { 0, TA_MIN_TEST, 0,  13, 14, TA_SUCCESS, 0, 91.125,     13, 1},

   /* Middle of data test. */
   { 0, TA_MIN_TEST, 20,  21, 14, TA_SUCCESS, 0, 89.345,   20, 2 },
   { 0, TA_MIN_TEST, 20,  21, 14, TA_SUCCESS, 1, 87.94,    20, 2 },

   /* Misc tests: 2 and 14 periods */
   { 1, TA_MIN_TEST, 0, 251, 14, TA_SUCCESS,      0, 91.125,  13,  252-13 }, /* First Value */
   { 0, TA_MIN_TEST, 0, 251, 14, TA_SUCCESS,      1, 91.125,  13,  252-13 },
   { 0, TA_MIN_TEST, 0, 251, 14, TA_SUCCESS,      2, 91.125,  13,  252-13 },
   { 0, TA_MIN_TEST, 0, 251, 14, TA_SUCCESS,      3, 91.125,  13,  252-13 },
   { 0, TA_MIN_TEST, 0, 251, 14, TA_SUCCESS,      4, 89.75,   13,  252-13 },
   { 0, TA_MIN_TEST, 0, 251, 14, TA_SUCCESS, 252-14, 107.75,  13,  252-13 },  /* Last Value */

   { 1, TA_MIN_TEST, 0, 251, 2, TA_SUCCESS,      0, 91.5,  1,  252-1 }, /* First Value */
   { 0, TA_MIN_TEST, 0, 251, 2, TA_SUCCESS,      1, 91.5,  1,  252-1 },
   { 0, TA_MIN_TEST, 0, 251, 2, TA_SUCCESS,      2, 93.97,  1,  252-1 },
   { 0, TA_MIN_TEST, 0, 251, 2, TA_SUCCESS,      3, 93.97,  1,  252-1 },
   { 0, TA_MIN_TEST, 0, 251, 2, TA_SUCCESS,      4, 94.5,   1,  252-1 },
   { 0, TA_MIN_TEST, 0, 251, 2, TA_SUCCESS, 252-2, 109.19,  1,  252-1 },  /* Last Value */

   /**********************/
   /*      MAX TEST      */
   /**********************/

   /* One value tests. */
   { 0, TA_MAX_TEST, 14,  14, 14, TA_SUCCESS, 0, 98.815,  14, 1},

   /* Index too low test. */
   { 0, TA_MAX_TEST, 0,  15, 14, TA_SUCCESS, 0, 98.815,     13, 3},
   { 0, TA_MAX_TEST, 1,  15, 14, TA_SUCCESS, 0, 98.815,     13, 3},
   { 0, TA_MAX_TEST, 2,  16, 14, TA_SUCCESS, 0, 98.815,     13, 4},
   { 0, TA_MAX_TEST, 2,  16, 14, TA_SUCCESS, 1, 98.815,     13, 4},
   { 0, TA_MAX_TEST, 2,  16, 14, TA_SUCCESS, 2, 98.815,     13, 4},
   { 0, TA_MAX_TEST, 0,  14, 14, TA_SUCCESS, 0, 98.815,     13, 2},
   { 0, TA_MAX_TEST, 0,  13, 14, TA_SUCCESS, 0, 98.815,     13, 1},

   /* Middle of data test. */
   { 0, TA_MAX_TEST, 20,  21, 14, TA_SUCCESS,  0, 98.815,   20, 2  },
   { 0, TA_MAX_TEST, 20,  21, 14, TA_SUCCESS,  1, 98.815,   20, 2  },
   { 0, TA_MAX_TEST, 20,  99, 14, TA_SUCCESS,  6, 93.405,   20, 80 },
   { 0, TA_MAX_TEST, 20,  99, 14, TA_SUCCESS,  6, 93.405,   20, 80 },
   { 0, TA_MAX_TEST, 20,  99, 14, TA_SUCCESS, 13, 89.78,    20, 80 },

   /* Misc tests: 1, 2 and 14 periods */
   { 1, TA_MAX_TEST, 0, 251, 14, TA_SUCCESS,      0, 98.815,  13,  252-13 }, /* First Value */
   { 0, TA_MAX_TEST, 0, 251, 14, TA_SUCCESS,      1, 98.815,  13,  252-13 },
   { 0, TA_MAX_TEST, 0, 251, 14, TA_SUCCESS,      2, 98.815,  13,  252-13 },
   { 0, TA_MAX_TEST, 0, 251, 14, TA_SUCCESS,      3, 98.815,  13,  252-13 },
   { 0, TA_MAX_TEST, 0, 251, 14, TA_SUCCESS,      4, 98.815,  13,  252-13 },
   { 0, TA_MAX_TEST, 0, 251, 14, TA_SUCCESS, 252-14, 110.69,  13,  252-13 },  /* Last Value */

   { 1, TA_MAX_TEST, 0, 251, 2, TA_SUCCESS,      0, 92.5,  1,  252-1 }, /* First Value */
   { 0, TA_MAX_TEST, 0, 251, 2, TA_SUCCESS,      1, 95.155,  1,  252-1 },
   { 0, TA_MAX_TEST, 0, 251, 2, TA_SUCCESS,      2, 95.155, 1,  252-1 },
   { 0, TA_MAX_TEST, 0, 251, 2, TA_SUCCESS,      3, 95.5, 1,  252-1 },
   { 0, TA_MAX_TEST, 0, 251, 2, TA_SUCCESS,      4, 95.5,  1,  252-1 },
   { 0, TA_MAX_TEST, 0, 251, 2, TA_SUCCESS,      5, 95.0,  1,  252-1 },
   { 0, TA_MAX_TEST, 0, 251, 2, TA_SUCCESS, 252-2, 109.69, 1,  252-1 },  /* Last Value */

  /*************************************/
  /*  MINMAX and INDEX Functions tests */
  /*************************************/

   /* Note: for now only range tests done on these */
   { 1, TA_MINMAX_TEST, 0, 251, 14, TA_SUCCESS, 0, 91.125,  13,  252-13 },
   { 1, TA_MINMAXINDEX_TEST, 0, 251, 14, TA_SUCCESS, 0, 91.125,  13,  252-13 },
   { 1, TA_MININDEX_TEST, 0, 251, 14, TA_SUCCESS, 0, 91.125,  13,  252-13 },
   { 1, TA_MAXINDEX_TEST, 0, 251, 14, TA_SUCCESS, 0, 91.125,  13,  252-13 }
};

#define NB_TEST (sizeof(tableTest)/sizeof(TA_Test))

static TA_Real testSerie1[]  = {9,8,7,6,5};
static TA_Real testSerie2[]  = {3,7,9,10,15,33,50};
static TA_Real testSerie3[]  = {0,0,0,1,2,0,0,0};
static TA_Real testSerie4[]  = {0,0,0,2,1,0,0,0};
static TA_Real testSerie5[]  = {2,0,0,0,0,0,0,0};
static TA_Real testSerie6[]  = {0,0,0,0,0,0,0,1};
static TA_Real testSerie7[]  = {-3,2};
static TA_Real testSerie8[]  = {2,-2};
static TA_Real testSerie9[]  = {4,2,3};
static TA_Real testSerie10[] = {3,3,-3,2,-1,0,2};

static TA_RefTest tableRefTest[] =
{
  {testSerie1, sizeof(testSerie1)/sizeof(TA_Real)},
  {testSerie2, sizeof(testSerie2)/sizeof(TA_Real)},
  {testSerie3, sizeof(testSerie3)/sizeof(TA_Real)},
  {testSerie4, sizeof(testSerie4)/sizeof(TA_Real)},
  {testSerie5, sizeof(testSerie5)/sizeof(TA_Real)},
  {testSerie6, sizeof(testSerie6)/sizeof(TA_Real)},
  {testSerie7, sizeof(testSerie7)/sizeof(TA_Real)},
  {testSerie8, sizeof(testSerie8)/sizeof(TA_Real)},
  {testSerie9, sizeof(testSerie10)/sizeof(TA_Real)}
};

#define NB_TEST_REF (sizeof(tableRefTest)/sizeof(TA_RefTest))

/**** Global functions definitions.   ****/
ErrorNumber test_func_minmax( TA_History *history )
{
   unsigned int i;
   ErrorNumber retValue;

   for( i=0; i < NB_TEST; i++ )
   {
      if( (int)tableTest[i].expectedNbElement > (int)history->nbBars )
      {
         printf( "%s Failed Bad Parameter for Test #%d (%d,%d)\n", __FILE__,
                 i, tableTest[i].expectedNbElement, history->nbBars );
         return TA_TESTUTIL_TFRR_BAD_PARAM;
      }

      retValue = do_test( history, &tableTest[i] );
      if( retValue != 0 )
      {
         printf( "%s Failed Test #%d (Code=%d)\n", __FILE__,
                 i, retValue );
         return retValue;
      }
   }

   /* Do tests against a local reference which is the non-optimized implementation */
   for( i=0; i < NB_TEST_REF; i++ )
   {
      retValue = testCompareToReference( tableRefTest[i].input,
                                         tableRefTest[i].nbElement );
      if( retValue != 0 )
      {
         printf( "%s Failed Ref Test #%d (Code=%d)\n", __FILE__,
                 i, retValue );
         return retValue;
      }
   }

   /* All test succeed. */
   return TA_TEST_PASS; 
}

/**** Local functions definitions.     ****/
static TA_RetCode rangeTestFunction( TA_Integer    startIdx,
                                     TA_Integer    endIdx,
                                     TA_Real      *outputBuffer,
                                     TA_Integer   *outputBufferInt,
                                     TA_Integer   *outBegIdx,
                                     TA_Integer   *outNbElement,
                                     TA_Integer   *lookback,
                                     void         *opaqueData,
                                     unsigned int  outputNb,
                                     unsigned int *isOutputInteger )
{
   TA_RetCode retCode;
   TA_RangeTestParam *testParam;
   TA_Real *dummyBufferReal;
   TA_Real *out1Real;
   TA_Real *out2Real;

   TA_Integer *dummyBufferInt;
   TA_Integer *out1Int;
   TA_Integer *out2Int;
  
   (void)outputNb;
   (void)outputBufferInt;

   *isOutputInteger = 0;

   testParam = (TA_RangeTestParam *)opaqueData;   

   dummyBufferReal = TA_Malloc( ((endIdx-startIdx)+1)*sizeof(TA_Real));
   if( !dummyBufferReal )
     return TA_ALLOC_ERR;

   dummyBufferInt = TA_Malloc( ((endIdx-startIdx)+1)*sizeof(TA_Integer));
   if( !dummyBufferInt )
   {
      TA_Free( dummyBufferReal );
      return TA_ALLOC_ERR;
   }

   switch( outputNb )
   {
   case 0:
      out1Real = outputBuffer;
      out2Real = dummyBufferReal; 
      out1Int  = outputBufferInt;
      out2Int  = dummyBufferInt; 
      break;
   case 1:
      out1Real = dummyBufferReal;
      out2Real = outputBuffer;      
      out1Int  = dummyBufferInt;
      out2Int  = outputBufferInt;
      break;
   default:
      TA_Free( dummyBufferReal );
      return TA_BAD_PARAM;
   }

   switch( testParam->test->theFunction )
   {
   case TA_MIN_TEST:   
      retCode = TA_MIN( startIdx,
                        endIdx,
                        testParam->close,
                        testParam->test->optInTimePeriod,                        
                        outBegIdx,
                        outNbElement,
                        outputBuffer );
      *lookback  = TA_MIN_Lookback( testParam->test->optInTimePeriod );
      break;
   
   case TA_MAX_TEST:   
      retCode = TA_MAX( startIdx,
                        endIdx,
                        testParam->close,
                        testParam->test->optInTimePeriod,
                        outBegIdx,
                        outNbElement,                        
                        outputBuffer );
      *lookback = TA_MAX_Lookback( testParam->test->optInTimePeriod );
      break;

   case TA_MINMAX_TEST:
      retCode = TA_MINMAX( startIdx,
                        endIdx,
                        testParam->close,
                        testParam->test->optInTimePeriod,
                        outBegIdx,
                        outNbElement,                        
                        out1Real, out2Real );
      *lookback = TA_MINMAX_Lookback( testParam->test->optInTimePeriod );            
      break;

   case TA_MINMAXINDEX_TEST:
      retCode = TA_MINMAXINDEX( startIdx,
                        endIdx,
                        testParam->close,
                        testParam->test->optInTimePeriod,
                        outBegIdx,
                        outNbElement,                        
                        out1Int, out2Int );
      *lookback = TA_MINMAXINDEX_Lookback( testParam->test->optInTimePeriod );
      *isOutputInteger = 1;
      break;

   case TA_MININDEX_TEST:
      retCode = TA_MININDEX( startIdx,
                        endIdx,
                        testParam->close,
                        testParam->test->optInTimePeriod,
                        outBegIdx,
                        outNbElement,                        
                        out1Int );
      *lookback = TA_MININDEX_Lookback( testParam->test->optInTimePeriod );
      *isOutputInteger = 1;
      break;

   case TA_MAXINDEX_TEST:
      retCode = TA_MAXINDEX( startIdx,
                        endIdx,
                        testParam->close,
                        testParam->test->optInTimePeriod,
                        outBegIdx,
                        outNbElement,                        
                        out1Int );
      *lookback = TA_MAXINDEX_Lookback( testParam->test->optInTimePeriod );
      *isOutputInteger = 1;
      break;

   default:
      retCode = TA_INTERNAL_ERROR(129);
      break;
   }

   TA_Free( dummyBufferReal );
   TA_Free( dummyBufferInt );

   return retCode;
}

static ErrorNumber do_test( const TA_History *history,
                            const TA_Test *test )
{
   TA_RetCode retCode;
   ErrorNumber errNb;
   TA_Integer outBegIdx;
   TA_Integer outNbElement;
   TA_RangeTestParam testParam;

   /* Set to NAN all the elements of the gBuffers.  */
   clearAllBuffers();

   /* Build the input. */
   setInputBuffer( 0, history->open, history->nbBars );
   setInputBuffer( 1, history->open, history->nbBars );

   CLEAR_EXPECTED_VALUE(0);

   /* Do a systematic test of most of the
    * possible startIdx/endIdx range.
    */
   testParam.test  = test;
   testParam.close = history->close;

   if( test->doRangeTestFlag )
   {
      errNb = doRangeTest( rangeTestFunction, 
                           TA_FUNC_UNST_NONE,
                           (void *)&testParam, 1, 0 );
      if( errNb != TA_TEST_PASS )
         return errNb;
   }


   /* Make a simple first call. */
   if( test->theFunction == TA_MIN_TEST )
   {
      retCode = TA_MIN( test->startIdx,
                        test->endIdx,
                        gBuffer[0].in,
                        test->optInTimePeriod,                        
                        &outBegIdx,
                        &outNbElement,
                        gBuffer[0].out0 );
   }
   else if( test->theFunction == TA_MAX_TEST )
   {
      retCode = TA_MAX( test->startIdx,
                        test->endIdx,
                        gBuffer[0].in,
                        test->optInTimePeriod,                        
                        &outBegIdx,
                        &outNbElement,
                        gBuffer[0].out0 );
   }
   else
   {
      /* For now, tests only MIN and MAX. Only range check tests implemented. */
      return TA_TEST_PASS;    
   }

   errNb = checkDataSame( gBuffer[0].in, history->open,history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;

   CHECK_EXPECTED_VALUE( gBuffer[0].out0, 0 );

   outBegIdx = outNbElement = 0;

   /* Make another call where the input and the output are the
    * same buffer.
    */
   CLEAR_EXPECTED_VALUE(0);
   if( test->theFunction == TA_MIN_TEST )
   {
      retCode = TA_MIN( test->startIdx,
                        test->endIdx,
                        gBuffer[1].in,
                        test->optInTimePeriod,                        
                        &outBegIdx,
                        &outNbElement,
                        gBuffer[1].in );
   }
   else if( test->theFunction == TA_MAX_TEST )
   {
      retCode = TA_MAX( test->startIdx,
                        test->endIdx,
                        gBuffer[1].in,
                        test->optInTimePeriod,
                        &outBegIdx,
                        &outNbElement,
                        gBuffer[1].in );
   }

   /* The previous call should have the same output as this call.
    *
    * checkSameContent verify that all value different than NAN in
    * the first parameter is identical in the second parameter.
    */
   errNb = checkSameContent( gBuffer[0].out0, gBuffer[1].in );
   if( errNb != TA_TEST_PASS )
      return errNb;

   CHECK_EXPECTED_VALUE( gBuffer[1].in, 0 );

   if( errNb != TA_TEST_PASS )
      return errNb;


   return TA_TEST_PASS;
}


/* These reference functions were the original non-optimized
 * version of TA_MIN and TA_MAX.
 *
 * TA-Lib might implement a faster algorithm, at the cost
 * of complexity. Consequently, it is important to verify the
 * equivalence between the optimize and non-optimized version.
 */
static TA_RetCode referenceMin( TA_Integer    startIdx,
                                TA_Integer    endIdx,
                                const TA_Real inReal[],
                                TA_Integer    optInTimePeriod,
                                TA_Integer   *outBegIdx,
                                TA_Integer   *outNbElement,
                                TA_Real       outReal[] )
{
   TA_Real lowest, tmp;
   TA_Integer outIdx, nbInitialElementNeeded;
   TA_Integer trailingIdx, today, i;

   /* Identify the minimum number of price bar needed
    * to identify at least one output over the specified
    * period.
    */
   nbInitialElementNeeded = (optInTimePeriod-1);

   /* Move up the start index if there is not
    * enough initial data.
    */
   if( startIdx < nbInitialElementNeeded )
      startIdx = nbInitialElementNeeded;

   /* Make sure there is still something to evaluate. */
   if( startIdx > endIdx )
   {
      *outBegIdx    = 0;
      *outNbElement = 0;
      return TA_SUCCESS;
   }

   /* Proceed with the calculation for the requested range.
    * Note that this algorithm allows the input and
    * output to be the same buffer.
    */
   outIdx = 0;
   today       = startIdx;
   trailingIdx = startIdx-nbInitialElementNeeded;
   
   while( today <= endIdx )
   {
      lowest = inReal[trailingIdx++];
      for( i=trailingIdx; i <= today; i++ )
      {
         tmp = inReal[i];
         if( tmp < lowest) lowest= tmp;
      }

      outReal[outIdx++] = lowest;
      today++;
   }

   /* Keep the outBegIdx relative to the
    * caller input before returning.
    */
   *outBegIdx    = startIdx;
   *outNbElement = outIdx;

   return TA_SUCCESS;
}

static TA_RetCode referenceMax( TA_Integer    startIdx,
                                TA_Integer    endIdx,
                                const TA_Real inReal[],
                                TA_Integer    optInTimePeriod,
                                TA_Integer   *outBegIdx,
                                TA_Integer   *outNbElement,
                                TA_Real       outReal[] )
{
   /* Insert local variables here. */
   TA_Real highest, tmp;
   TA_Integer outIdx, nbInitialElementNeeded;
   TA_Integer trailingIdx, today, i;


#ifndef TA_FUNC_NO_RANGE_CHECK

   /* Validate the requested output range. */
   if( startIdx < 0 )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx < startIdx))
      return TA_OUT_OF_RANGE_END_INDEX;

   /* Validate the parameters. */
   if( !inReal ) return TA_BAD_PARAM;
   /* min/max are checked for optInTimePeriod. */
   if( optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 30;

   if( outReal == NULL )
      return TA_BAD_PARAM;

#endif /* TA_FUNC_NO_RANGE_CHECK */

   /* Insert TA function code here. */

   /* Identify the minimum number of price bar needed
    * to identify at least one output over the specified
    * period.
    */
   nbInitialElementNeeded = (optInTimePeriod-1);

   /* Move up the start index if there is not
    * enough initial data.
    */
   if( startIdx < nbInitialElementNeeded )
      startIdx = nbInitialElementNeeded;

   /* Make sure there is still something to evaluate. */
   if( startIdx > endIdx )
   {
      *outBegIdx    = 0;
      *outNbElement = 0;
      return TA_SUCCESS;
   }

   /* Proceed with the calculation for the requested range.
    * Note that this algorithm allows the input and
    * output to be the same buffer.
    */
   outIdx = 0;
   today       = startIdx;
   trailingIdx = startIdx-nbInitialElementNeeded;
   
   while( today <= endIdx )
   {
      highest = inReal[trailingIdx++];
      for( i=trailingIdx; i <= today; i++ )
      {
         tmp = inReal[i];
         if( tmp > highest ) highest = tmp;
      }

      outReal[outIdx++] = highest;
      today++;
   }

   /* Keep the outBegIdx relative to the
    * caller input before returning.
    */
   *outBegIdx    = startIdx;
   *outNbElement = outIdx;

   return TA_SUCCESS;
}

static ErrorNumber testCompareToReference( const TA_Real *input, int nbElement )
{
   TA_Integer outBegIdx, outNbElement;
   TA_RetCode retCode;

   TA_Integer outBegIdxRef, outNbElementRef;
   TA_RetCode retCodeRef;

   int period, startIdx, endIdx, testNb;

   ErrorNumber errNb;

   outBegIdxRef = outNbElementRef = -1;

   /* Do a systematic tests, even for failure cases. */
   for( testNb=0; testNb <= 1; testNb++ ) /* 0=TA_MIN, 1=TA_MAX */
   {      
      for( period=2; period <= nbElement; period++ )
      {
         for( startIdx=0; startIdx < nbElement; startIdx++ )
         {
            for( endIdx=0; (endIdx < nbElement) && (startIdx <= endIdx); endIdx++ )
            {            
               /* Set to NAN all the elements of the gBuffers.
                * Note: These buffer are used as an attempt to detect
                *       out-of-bound writing in the output.
                */
               clearAllBuffers();

               /* Build the input. */
               setInputBuffer( 0, input, nbElement );

               /* Get the reference output. */
               if( testNb == 0 )
                  retCodeRef = referenceMin( startIdx, endIdx, input, period, 
                                             &outBegIdxRef, &outNbElementRef, gBuffer[0].out0 );
               else
                  retCodeRef = referenceMax( startIdx, endIdx, input, period, 
                                             &outBegIdxRef, &outNbElementRef, gBuffer[0].out0 );

               /* Verify that the input was preserved */
               errNb = checkDataSame( gBuffer[0].in, input, nbElement );
               if( errNb != TA_TEST_PASS )
                  return errNb;

               /* Get the TA-Lib implementation output. */
               if( testNb == 0 )
                  retCode = TA_MIN( startIdx, endIdx, input, period, 
                                    &outBegIdx, &outNbElement, gBuffer[1].out0 );
               else
                  retCode = TA_MAX( startIdx, endIdx, input, period, 
                                    &outBegIdx, &outNbElement, gBuffer[1].out0 );

               /* Verify that the input was preserved */
               errNb = checkDataSame( gBuffer[0].in, input, nbElement );
               if( errNb != TA_TEST_PASS )
                  return errNb;

               /* The reference and TA-LIB should have the same output. */
               if( retCode != retCodeRef )
               {
                  printf( "Failure: retCode != retCodeRef\n" );
                  return TA_REGTEST_OPTIMIZATION_REF_ERROR;
               }

               if( outBegIdx != outBegIdxRef )
               {
                  printf( "Failure: outBegIdx != outBegIdxRef\n" );
                  return TA_REGTEST_OPTIMIZATION_REF_ERROR;
               }

               if( outNbElement != outNbElementRef )
               {
                  printf( "Failure: outNbElement != outNbElementRef\n" );
                  return TA_REGTEST_OPTIMIZATION_REF_ERROR;
               }

               /* checkSameContent verify that all value different than NAN in
                * the first parameter is identical in the second parameter.
                */
               errNb = checkSameContent( gBuffer[0].out0, gBuffer[1].out0 );
               if( errNb != TA_TEST_PASS )
                  return errNb;

               if( retCode == TA_SUCCESS )
               {
                  /* Make another test using the same input/output buffer.
                   * The output should still be the same.
                   */
                  if( testNb == 0 )
                     retCode = TA_MIN( startIdx, endIdx, gBuffer[0].in, period, 
                                       &outBegIdx, &outNbElement, gBuffer[0].in );
                  else
                     retCode = TA_MAX( startIdx, endIdx, gBuffer[0].in, period, 
                                       &outBegIdx, &outNbElement, gBuffer[0].in );

                  /* The reference and TA-LIB should have the same output. */
                  if( retCode != retCodeRef )
                  {
                     printf( "Failure: retCode != retCodeRef (2)\n" );
                     return TA_REGTEST_OPTIMIZATION_REF_ERROR;
                  }

                  if( outBegIdx != outBegIdxRef )
                  {
                     printf( "Failure: outBegIdx != outBegIdxRef (2)\n" );
                     return TA_REGTEST_OPTIMIZATION_REF_ERROR;
                  }

                  if( outNbElement != outNbElementRef )
                  {
                     printf( "Failure: outNbElement != outNbElementRef (2)\n" );
                     return TA_REGTEST_OPTIMIZATION_REF_ERROR;
                  }

                  /* checkSameContent verify that all value different than NAN in
                   * the first parameter is identical in the second parameter.
                   */
                  errNb = checkSameContent( gBuffer[0].out0, gBuffer[0].in );
                  if( errNb != TA_TEST_PASS )
                     return errNb;
               }
            }
         }
      }  
   }

   return TA_TEST_PASS;
}
