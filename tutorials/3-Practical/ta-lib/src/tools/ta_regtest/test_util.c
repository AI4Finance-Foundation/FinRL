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
 *
 */

/* Description:
 *     Provide utility function internally used in ta_regtest only.
 */

/**** Headers ****/
#ifdef WIN32
   #include "windows.h"
#endif

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "ta_test_priv.h"
#include "ta_utility.h"
#include "ta_memory.h"


/**** External functions declarations. ****/
/* None */

/**** External variables declarations. ****/
extern int nbProfiledCall;
extern double timeInProfiledCall;
extern double worstProfiledCall;
extern int insufficientClockPrecision;

/**** Global variables definitions.    ****/

/* Global temporary buffers used while testing. */
#define RESV_PATTERN_MEMGUARD_1   (2.4789205E-150)
#define RESV_PATTERN_MEMGUARD_2   (4.2302468E-165)

#define RESV_PATTERN_PREFIX       (9.1349043E-200)
#define RESV_PATTERN_SUFFIX       (8.1489031E-158)
#define RESV_PATTERN_IMPROBABLE   (-2.849284E-199)

#define RESV_PATTERN_PREFIX_INT     (TA_INTEGER_DEFAULT)
#define RESV_PATTERN_SUFFIX_INT     (TA_INTEGER_DEFAULT)
#define RESV_PATTERN_IMPROBABLE_INT (TA_INTEGER_DEFAULT)

#define TA_BUF_PREFIX 100
#define TA_BUF_SUFFIX 100
#define TA_BUF_SIZE   (TA_BUF_PREFIX+MAX_NB_TEST_ELEMENT+TA_BUF_SUFFIX)

#define TA_NB_OUT 3
#define TA_NB_IN  1
#define TA_NB_OUT_IN (TA_NB_OUT+TA_NB_IN)

TA_Real memoryGuard1 = RESV_PATTERN_MEMGUARD_1; /* Magic number to detect problem. */
TA_Real buf[NB_GLOBAL_BUFFER][TA_NB_OUT_IN][TA_BUF_SIZE];     /* The global buffers. */
TA_Real memoryGuard2 = RESV_PATTERN_MEMGUARD_2; /* Magic number to detect problem. */

#define NB_TOTAL_ELEMENTS (sizeof(buf)/sizeof(TA_Real))

TestBuffer gBuffer[5]; /* See initGlobalBuffer. */

/**** Local declarations.              ****/
/* None */

/**** Local functions declarations.    ****/
static ErrorNumber doRangeTestFixSize( RangeTestFunction testFunction,
                                       void *opaqueData,
                                       TA_Integer refOutBeg,
                                       TA_Integer refOutNbElement,
                                       TA_Integer refLookback,
                                       const TA_Real    *refBuffer,
                                       const TA_Integer *refBufferInt,
                                       TA_FuncUnstId unstId,
                                       TA_Integer fixSize,
                                       unsigned int outputNb,
                                       unsigned int integerTolerance );

static int dataWithinReasonableRange( TA_Real val1, TA_Real val2,
                                      unsigned int outputPosition,
                                      TA_FuncUnstId unstId,
                                      unsigned int integerTolerance );

static ErrorNumber doRangeTestForOneOutput( RangeTestFunction testFunction,
                                            TA_FuncUnstId unstId,
                                            void *opaqueData,
                                            unsigned int outputNb,
                                            unsigned int integerTolerance );

static TA_RetCode CallTestFunction( RangeTestFunction testFunction,
                                    TA_Integer    startIdx,
                                    TA_Integer    endIdx,
                                    TA_Real      *outputBuffer,
                                    TA_Integer   *outputBufferInt,
                                    TA_Integer   *outBegIdx,
                                    TA_Integer   *outNbElement,
                                    TA_Integer   *lookback,
                                    void         *opaqueData,
                                    unsigned int  outputNb,
                                    unsigned int *isOutputInteger );

/**** Local variables definitions.     ****/
/* None */

/**** Global functions definitions.   ****/
static int ta_g_val = 0;
static const char *ta_g_wheel = "-\\|/";
void showFeedback()
{
   if( ta_g_wheel[ta_g_val] == '\0' )
      ta_g_val = 0; 
   putchar('\b');
   putchar(ta_g_wheel[ta_g_val]);
   fflush(stdout);
   ta_g_val++;
}

void hideFeedback()
{
   putchar('\b');
   fflush(stdout);
   ta_g_val = 0;
}

ErrorNumber allocLib()
{
   TA_RetCode retCode;

   /* Initialize the library. */
   retCode = TA_Initialize();
   if( retCode != TA_SUCCESS )
   {
      printf( "TA_Initialize failed [%d]\n", retCode );
      return TA_TESTUTIL_INIT_FAILED;
   }

   return TA_TEST_PASS;
}

ErrorNumber freeLib()
{
   TA_RetCode retCode;

   /* For testing purpose */
   /* TA_FATAL_RET( "Test again", 100, 200, 0 ); */

   retCode = TA_Shutdown();
   if( retCode != TA_SUCCESS )
   {
      printf( "TA_Shutdown failed [%d]\n", retCode );
      return TA_TESTUTIL_SHUTDOWN_FAILED;
   }

   return TA_TEST_PASS;
}

void reportError( const char *str, TA_RetCode retCode )
{
   TA_RetCodeInfo retCodeInfo;

   TA_SetRetCodeInfo( retCode, &retCodeInfo );

   printf( "%s,%d==%s\n", str, retCode, retCodeInfo.enumStr );
   printf( "[%s]\n", retCodeInfo.infoStr );
}

/* Need to be called only once. */
void initGlobalBuffer( void )
{
   gBuffer[0].in   = &buf[0][0][TA_BUF_PREFIX];
   gBuffer[0].out0 = &buf[0][1][TA_BUF_PREFIX];
   gBuffer[0].out1 = &buf[0][2][TA_BUF_PREFIX];
   gBuffer[0].out2 = &buf[0][3][TA_BUF_PREFIX];
   
   gBuffer[1].in   = &buf[1][0][TA_BUF_PREFIX];
   gBuffer[1].out0 = &buf[1][1][TA_BUF_PREFIX];
   gBuffer[1].out1 = &buf[1][2][TA_BUF_PREFIX];
   gBuffer[1].out2 = &buf[1][3][TA_BUF_PREFIX];

   gBuffer[2].in   = &buf[2][0][TA_BUF_PREFIX];
   gBuffer[2].out0 = &buf[2][1][TA_BUF_PREFIX];
   gBuffer[2].out1 = &buf[2][2][TA_BUF_PREFIX];
   gBuffer[2].out2 = &buf[2][3][TA_BUF_PREFIX];

   gBuffer[3].in   = &buf[3][0][TA_BUF_PREFIX];
   gBuffer[3].out0 = &buf[3][1][TA_BUF_PREFIX];
   gBuffer[3].out1 = &buf[3][2][TA_BUF_PREFIX];
   gBuffer[3].out2 = &buf[3][3][TA_BUF_PREFIX];
   
   gBuffer[4].in   = &buf[4][0][TA_BUF_PREFIX];
   gBuffer[4].out0 = &buf[4][1][TA_BUF_PREFIX];
   gBuffer[4].out1 = &buf[4][2][TA_BUF_PREFIX];
   gBuffer[4].out2 = &buf[4][3][TA_BUF_PREFIX];
}

/* Will set some values in the buffers allowing
 * to detect later if the function is writing
 * out-of-bound (and to make sure the
 * function is writing exactly the number
 * of values it pretends to do).
 */
void clearAllBuffers( void )
{
   unsigned int i,j,k;

   for( i=0; i < NB_GLOBAL_BUFFER; i++ )
   {
      for( j=0; j < TA_NB_OUT_IN; j++ )
      {
         for( k=0; k < TA_BUF_PREFIX; k++ )
            buf[i][j][k] = RESV_PATTERN_PREFIX;
         for( ; k < TA_BUF_SIZE; k++ )
            buf[i][j][k] = RESV_PATTERN_SUFFIX;
      }
   }
}

void setInputBuffer( unsigned int i, const TA_Real *data, unsigned int nbElement )
{
   unsigned int j;
   for( j=0; j < nbElement; j++ )
      buf[i][0][j+TA_BUF_PREFIX] = data[j];
}

void setInputBufferValue( unsigned int i, const TA_Real data, unsigned int nbElement )
{
   unsigned int j;
   for( j=0; j < nbElement; j++ )
      buf[i][0][j+TA_BUF_PREFIX] = data;

}

/* Check that a buffer (within a TestBuffer) is not containing
 * NAN (or any reserved "impossible" value) within the specified
 * range (it also checks that all out-of-bound values are untouch).
 *
 * Return 1 on success.
 */
ErrorNumber checkForNAN( const TA_Real *buffer,
                         unsigned int nbElement )
{
   unsigned int i;
   unsigned int idx;

   const TA_Real *theBuffer;
   theBuffer = buffer - TA_BUF_PREFIX;

   /* Check that the prefix are all still untouch. */
   for( idx=0; idx < TA_BUF_PREFIX; idx++ )
   {
      if( theBuffer[idx] != RESV_PATTERN_PREFIX )
      {
         printf( "Fail: Out of range writing in prefix buffer (%d,%f)\n", idx, theBuffer[idx] );
         return TA_TEST_TFRR_OVERLAP_OR_NAN_0;
      }
   }

   if( nbElement > MAX_NB_TEST_ELEMENT )
   {
       printf( "Fail: outNbElement is out of range 0 (%d)\n", nbElement );
       return TA_TEST_TFRR_NB_ELEMENT_OUT_OF_RANGE;
   }

   /* Check that no NAN (or reserved "impossible" value) exist
    * in the specified range.
    */
   for( i=0; i < nbElement; i++,idx++ )
   {
      /* TODO Add back some nan/inf checking
      if( trio_isnan(theBuffer[idx]) )
      {
         printf( "Fail: Not a number find within the data (%d,%f)\n", i, theBuffer[idx] );
         return TA_TEST_TFRR_OVERLAP_OR_NAN_1;
      }

      if( trio_isinf(theBuffer[idx]) )
      {
         printf( "Fail: Not a number find within the data (%d,%f)\n", i, theBuffer[idx] );
         return TA_TEST_TFRR_OVERLAP_OR_NAN_2;
      }*/

      if( theBuffer[idx] == RESV_PATTERN_PREFIX )
      {
         printf( "Fail: Not a number find within the data (%d,%f)\n", i, theBuffer[idx] );
         return TA_TEST_TFRR_OVERLAP_OR_NAN_3;
      }

      if( theBuffer[idx] == RESV_PATTERN_SUFFIX )
      {
         printf( "Fail: Not a number find within the data (%d,%f)\n", i, theBuffer[idx] );
         return TA_TEST_TFRR_OVERLAP_OR_NAN_4;
      }
   }

   /* Make sure that the remaining of the buffer is untouch. */
   for( ; idx < TA_BUF_SIZE; idx++ )
   {
      if( theBuffer[idx] != RESV_PATTERN_SUFFIX )
      {
         printf( "Fail: Out of range writing in suffix buffer (%d,%f)\n", idx, theBuffer[idx] );
         return TA_TEST_TFRR_OVERLAP_OR_NAN_5;
      }

      idx++;
   }

   /* Make sure the global memory guard are untouch. */
   if( memoryGuard1 != RESV_PATTERN_MEMGUARD_1 )
   {
      printf( "Fail: MemoryGuard1 have been modified (%f,%f)\n", memoryGuard1, RESV_PATTERN_MEMGUARD_1 );
      return TA_TEST_TFRR_OVERLAP_OR_NAN_6;
   }

   if( memoryGuard2 != RESV_PATTERN_MEMGUARD_2 )
   {
      printf( "Fail: MemoryGuard2 have been modified (%f,%f)\n", memoryGuard2, RESV_PATTERN_MEMGUARD_2 );
      return TA_TEST_TFRR_OVERLAP_OR_NAN_7;
   }

   /* Everything looks good! */
   return TA_TEST_PASS;
}

/* Return 1 on success */
ErrorNumber checkSameContent( TA_Real *buffer1,
                              TA_Real *buffer2 )
{
   const TA_Real *theBuffer1;
   const TA_Real *theBuffer2;

   unsigned int i;

   theBuffer1 = buffer1 - TA_BUF_PREFIX;
   theBuffer2 = buffer2 - TA_BUF_PREFIX;

   for( i=0; i < TA_BUF_SIZE; i++ )
   {
        /* TODO Add back nan/inf checking
          (!trio_isnan(theBuffer1[i])) &&
          (!trio_isinf(theBuffer1[i])) &&
         */

      if( (theBuffer1[i] != RESV_PATTERN_SUFFIX) &&
          (theBuffer1[i] != RESV_PATTERN_PREFIX) )
      {
         
         if(!TA_REAL_EQ( theBuffer1[i], theBuffer2[i], 0.000001))
         {
            printf( "Fail: Large difference found between two value expected identical (%f,%f,%d)\n",
                     theBuffer1[i], theBuffer2[i], i );
            return TA_TEST_TFRR_CHECK_SAME_CONTENT;
         }
      }
   }

   return TA_TEST_PASS;
}

ErrorNumber checkDataSame( const TA_Real *data,
                           const TA_Real *originalInput,
                           unsigned int nbElement )
{
   unsigned int i;
   ErrorNumber errNb;

   errNb = checkForNAN( data, nbElement );

   if( errNb != TA_TEST_PASS )
       return errNb;

   if( nbElement > MAX_NB_TEST_ELEMENT )
   {
       printf( "Fail: outNbElement is out of range 1 (%d)\n", nbElement );
       return TA_TEST_TFRR_NB_ELEMENT_OUT_OF_RANGE;
   }

   for( i=0; i < nbElement; i++ )
   {
      if( originalInput[i] != data[i] )
      {
         printf( "Fail: Data was wrongly modified (%f,%f,%d)\n",
                 originalInput[i],
                 data[i], i );
         return TA_TEST_TFRR_INPUT_HAS_BEEN_MODIFIED;
      }
   }

   return TA_TEST_PASS;
}

ErrorNumber checkExpectedValue( const TA_Real *data,
                                TA_RetCode retCode, TA_RetCode expectedRetCode,
                                unsigned int outBegIdx, unsigned int expectedBegIdx,
                                unsigned int outNbElement, unsigned int expectedNbElement,
                                TA_Real oneOfTheExpectedOutReal,
                                unsigned int oneOfTheExpectedOutRealIndex )
{    
   if( retCode != expectedRetCode )
   {
      printf( "Fail: RetCode %d different than expected %d\n", retCode, expectedRetCode );
      return TA_TESTUTIL_TFRR_BAD_RETCODE;
   }

   if( retCode != TA_SUCCESS )
   {
      /* An error did occured, but it
       * was expected. No need to go
       * further.
       */      
      return TA_TEST_PASS; 
   }

   if( outNbElement > MAX_NB_TEST_ELEMENT )
   {
       printf( "Fail: outNbElement is out of range 2 (%d)\n", outNbElement );
       return TA_TEST_TFRR_NB_ELEMENT_OUT_OF_RANGE;
   }


   /* Make sure the range of output does not contains NAN. */
   /* TODO Add back nan/inf checking
   for( i=0; i < outNbElement; i++ )
   {
      if( trio_isnan(data[i]) )
      {
         printf( "Fail: Not a number find within the data (%d,%f)\n", i, data[i] );
         return TA_TEST_TFRR_OVERLAP_OR_NAN_3;
      }
   }*/

   /* Verify that the expected output is there. */

   if( outNbElement != expectedNbElement )
   {
      printf( "Fail: outNbElement expected %d but got %d\n",
              expectedNbElement, outNbElement );
      return TA_TESTUTIL_TFRR_BAD_OUTNBELEMENT;
   }

   if( expectedNbElement > 0 )
   {
      if( !TA_REAL_EQ( oneOfTheExpectedOutReal, data[oneOfTheExpectedOutRealIndex], 0.01 ) )
      {
         printf( "Fail: For index %d, Expected value = %f but calculate value is %f\n",
                 oneOfTheExpectedOutRealIndex,
                 oneOfTheExpectedOutReal,
                 data[oneOfTheExpectedOutRealIndex] );
         return TA_TESTUTIL_TFRR_BAD_CALCULATION;
      }
   
      if( expectedBegIdx != outBegIdx )
      {
         printf( "Fail: outBegIdx expected %d but got %d\n", expectedBegIdx, outBegIdx );
         return TA_TESTUTIL_TFRR_BAD_BEGIDX;
      }
   }

   /* Succeed. */
   return TA_TEST_PASS;
}


ErrorNumber doRangeTest( RangeTestFunction testFunction,
                         TA_FuncUnstId unstId,
                         void *opaqueData,
                         unsigned int nbOutput,
                         unsigned int integerTolerance )
{
   unsigned int outputNb;
   ErrorNumber errNb;

   /* Test all the outputs individually. */
   for( outputNb=0; outputNb < nbOutput; outputNb++ )
   {
      errNb = doRangeTestForOneOutput( testFunction,
                                       unstId,
                                       opaqueData,
                                       outputNb,
                                       integerTolerance );
      if( errNb != TA_TEST_PASS )
      {
         printf( "Failed: For output #%d of %d\n", outputNb+1, nbOutput );
         return errNb;
      }
   }

   return TA_TEST_PASS;
}

void printRetCode( TA_RetCode retCode )
{
   TA_RetCodeInfo retCodeInfo;

   TA_SetRetCodeInfo( retCode, &retCodeInfo );
   printf( "\nFailed: ErrorCode %d=%s:[%s]\n", retCode,
           retCodeInfo.enumStr,
           retCodeInfo.infoStr );
}



/**** Local functions definitions.     ****/
static ErrorNumber doRangeTestForOneOutput( RangeTestFunction testFunction,
                                            TA_FuncUnstId unstId,
                                            void *opaqueData,
                                            unsigned int outputNb,
                                            unsigned int integerTolerance )
{
   TA_RetCode retCode;
   TA_Integer refOutBeg, refOutNbElement, refLookback;
   TA_Integer fixSize;
   TA_Real *refBuffer;
   TA_Integer *refBufferInt;
   ErrorNumber errNb;
   TA_Integer unstablePeriod, temp;
   unsigned int outputIsInteger;

   showFeedback();

   /* Caculate the whole range. This is going
    * to be the reference for all subsequent test.
    */
   refBuffer = (TA_Real *)TA_Malloc( MAX_RANGE_SIZE * sizeof( TA_Real ) );

   if( !refBuffer )
      return TA_TESTUTIL_DRT_ALLOC_ERR;

   refBufferInt = (TA_Integer *)TA_Malloc( MAX_RANGE_SIZE * sizeof( TA_Integer ) );

   if( !refBufferInt )
   {
      TA_Free( refBuffer );
      return TA_TESTUTIL_DRT_ALLOC_ERR;
   }
  
   if( unstId != TA_FUNC_UNST_NONE )
   {
      /* Caller wish to test for a range of unstable
       * period values. But the reference is calculated
       * on the whole range by keeping that unstable period
       * to zero.
       */
      TA_SetUnstablePeriod( unstId, 0 );
   }

   outputIsInteger = 0;   
   retCode = CallTestFunction( testFunction, 0, MAX_RANGE_END, refBuffer, refBufferInt,
                           &refOutBeg, &refOutNbElement, &refLookback,
                           opaqueData, outputNb, &outputIsInteger );

   if( retCode != TA_SUCCESS )
   {
      printf( "Fail: doRangeTest whole range failed (%d)\n", retCode );
      TA_Free( refBuffer );
      TA_Free( refBufferInt );
      return TA_TESTUTIL_DRT_REF_FAILED;
   }

   /* When calculating for the whole range, the lookback and the
    * refOutBeg are supppose to be equal.
    */
   if( refLookback != refOutBeg )
   {
      printf( "Fail: doRangeTest refLookback != refOutBeg (%d != %d)\n", refLookback, refOutBeg );
      TA_Free( refBuffer );
      TA_Free( refBufferInt );
      return TA_TESTUTIL_DRT_LOOKBACK_INCORRECT;
   }
   
   temp = MAX_RANGE_SIZE-refLookback;
   if( temp != refOutNbElement )
   {
      printf( "Fail: doRangeTest either refOutNbElement or refLookback bad (%d,%d)\n", temp, refOutNbElement );
      TA_Free( refBuffer );
      TA_Free( refBufferInt );
      return TA_TESTUTIL_DRT_REF_OUTPUT_INCORRECT;
   }

   /* Calculate each value ONE by ONE and make sure it is identical
    * to the reference.
    *
    * Then repeat the test but calculate TWO by TWO and so on...
    */
   for( fixSize=1; fixSize <= MAX_RANGE_SIZE; fixSize++ )
   {
      /* When a function has an unstable period, verify some
       * unstable period between 0 and MAX_RANGE_SIZE.
       */
      if( unstId == TA_FUNC_UNST_NONE )
      {
         errNb = doRangeTestFixSize( testFunction, opaqueData,
                                     refOutBeg, refOutNbElement, refLookback,
                                     refBuffer, refBufferInt,
                                     unstId, fixSize, outputNb, integerTolerance );
         if( errNb != TA_TEST_PASS)
         {
            TA_Free( refBuffer );
            TA_Free( refBufferInt );
            return errNb;
         }
      }
      else
      {         
         for( unstablePeriod=0; unstablePeriod <= MAX_RANGE_SIZE; unstablePeriod++ )
         {
            TA_SetUnstablePeriod( unstId, unstablePeriod );

            errNb = doRangeTestFixSize( testFunction, opaqueData,
                                        refOutBeg, refOutNbElement, refLookback,
                                        refBuffer, refBufferInt,
                                        unstId, fixSize, outputNb, integerTolerance );
            if( errNb != TA_TEST_PASS)
            {
               printf( "Fail: Using unstable period %d\n", unstablePeriod );
               TA_Free( refBuffer );
               TA_Free( refBufferInt );
               return errNb;
            }

            /* Randomly skip the test of some unstable period (limit case are
             * always tested though).
             */
            if( (unstablePeriod > 5) && (unstablePeriod < 240) )
            {
               /* Randomly skips from 0 to 239 tests. Never
                * make unstablePeriod exceed 240.
                */
               temp = (rand() % 240);
               unstablePeriod += temp;
               if( unstablePeriod > 240 )
                  unstablePeriod = 240;
            }
         }

         /* Because the tests with an unstable period are very intensive
          * and kinda repetitive, skip the test of some fixSize (limit 
          * case are always tested though).
          */
         if( (fixSize > 5) && (fixSize < 240) )
         {
            /* Randomly skips from 0 to 239 tests. Never
             * make fixSize exceed 240.
             */
            temp = (rand() % 239);
            fixSize += temp;
            if( fixSize > 240 )
               fixSize = 240;
         }
      }
   }

   TA_Free( refBuffer );
   TA_Free( refBufferInt );
   return TA_TEST_PASS;
}

static ErrorNumber doRangeTestFixSize( RangeTestFunction testFunction,
                                       void *opaqueData,
                                       TA_Integer refOutBeg,
                                       TA_Integer refOutNbElement,
                                       TA_Integer refLookback,
                                       const TA_Real *refBuffer,
                                       const TA_Integer *refBufferInt,
                                       TA_FuncUnstId unstId,
                                       TA_Integer fixSize,
                                       unsigned int outputNb,
                                       unsigned int integerTolerance )
{
   TA_RetCode retCode;
   TA_Real *outputBuffer;
   TA_Real val1, val2;
   TA_Integer i, temp;
   TA_Integer outputBegIdx, outputNbElement, lookback;
   TA_Integer startIdx, endIdx, relativeIdx, outputSizeByOptimalLogic;
   TA_Integer *outputBufferInt;
   unsigned int outputIsInteger;

   (void)refLookback;

   /* Allocate the output buffer (+prefix and suffix memory guard). */
   outputBuffer = (TA_Real *)TA_Malloc( (fixSize+2) * sizeof( TA_Real ) );

   if( !outputBuffer )
      return TA_TESTUTIL_DRT_ALLOC_ERR;

   outputBufferInt = (TA_Integer *)TA_Malloc( (fixSize+2) * sizeof( TA_Integer ) );

   if( !refBufferInt )
   {
      TA_Free( outputBuffer );
      return TA_TESTUTIL_DRT_ALLOC_ERR;
   }

   outputBuffer[0] = RESV_PATTERN_PREFIX;
   outputBuffer[fixSize+1] = RESV_PATTERN_SUFFIX;

   outputBufferInt[0] = RESV_PATTERN_PREFIX_INT;
   outputBufferInt[fixSize+1] = RESV_PATTERN_SUFFIX_INT;

   /* Initialize the outputs with improbable values. */
   for( i=1; i <= fixSize; i++ )
   {
      outputBuffer[i] = RESV_PATTERN_IMPROBABLE;
      outputBufferInt[i] = RESV_PATTERN_IMPROBABLE_INT;
   }

   /* Test for a large number of possible startIdx */
   for( startIdx=0; startIdx <= (MAX_RANGE_SIZE-fixSize); startIdx++ )
   {
      /* Call the TA function. */
      endIdx = startIdx+fixSize-1;
      retCode = CallTestFunction( testFunction, startIdx, endIdx,
                              &outputBuffer[1], &outputBufferInt[1],
                              &outputBegIdx, &outputNbElement, &lookback,
                              opaqueData, outputNb, &outputIsInteger );
      
      if( retCode != TA_SUCCESS ) 
      {
          /* No call shall never fail here. When the range
           * is "out-of-range" the function shall still return
           * TA_SUCCESS with the outNbElement equal to zero.
           */
         printf( "Fail: doRangeTestFixSize testFunction return error=(%d) (%d,%d)\n", retCode, fixSize, startIdx );
         TA_Free( outputBuffer );
         TA_Free( outputBufferInt );
         return TA_TESTUTIL_DRT_RETCODE_ERR;
      }
      else
      {
         /* Possible startIdx gap of the output shall be always the
          * same regardless of the range.
          */
         if( outputNbElement == 0 )
         {
            /* Trap cases where there is no output. */
            if( (startIdx > lookback) || (endIdx > lookback) )
            {
               /* Whenever startIdx is greater than lookback, some data 
                * shall be return. Same idea with endIdx.
                * 
                * Note:
                *  some output will never start at the startIdx, particularly
                *  when a TA function have multiple output. Usually, the first output
                *  will be between startIdx/endIdx and other outputs may have a "gap"
                *  from the startIdx.
                *
                * Example:
                *    Stochastic %K is between startIdx/endIdx, but %D output will
                *    have less data because it is a MA of %K. A gap will then
                *    exist for the %D output.
                */
               printf( "Fail: doRangeTestFixSize data missing (%d,%d,%d)\n", startIdx, endIdx, lookback );
                                                                                
               TA_Free( outputBuffer );
               TA_Free( outputBufferInt );
               return TA_TESTUTIL_DRT_MISSING_DATA;
            }
         }
         else
         {
            /* Some output was returned. Are the returned index correct? */
            if( (outputBegIdx < startIdx) || (outputBegIdx > endIdx) || (outputBegIdx < refOutBeg))
            {
               printf( "Fail: doRangeTestFixSize bad outBegIdx\n" );
               printf( "Fail: doRangeTestFixSize (%d,%d,%d,%d,%d)\n", startIdx, endIdx, outputBegIdx, outputNbElement, fixSize );
               printf( "Fail: doRangeTestFixSize refOutBeg,refOutNbElement (%d,%d)\n", refOutBeg, refOutNbElement );
               TA_Free( outputBuffer );
               TA_Free( outputBufferInt );
               return TA_TESTUTIL_DRT_BAD_OUTBEGIDX;
            }

            if( (outputNbElement > fixSize) || (outputNbElement > refOutNbElement) )
            {
               printf( "Fail: doRangeTestFixSize Incorrect outputNbElement\n" );
               printf( "Fail: doRangeTestFixSize (%d,%d,%d,%d,%d)\n", startIdx, endIdx, outputBegIdx, outputNbElement, fixSize );
               printf( "Fail: doRangeTestFixSize refOutBeg,refOutNbElement (%d,%d)\n", refOutBeg, refOutNbElement );
               TA_Free(  outputBuffer );
               return TA_TESTUTIL_DRT_BAD_OUTNBLEMENT;
            }

            /* Is the calculated lookback too high? */
            if( outputBegIdx < lookback )
            {
               printf( "Fail: doRangeTestFixSize Lookback calculation too high? (%d)\n", lookback );
               printf( "Fail: doRangeTestFixSize (%d,%d,%d,%d,%d)\n", startIdx, endIdx, outputBegIdx, outputNbElement, fixSize );
               printf( "Fail: doRangeTestFixSize refOutBeg,refOutNbElement (%d,%d)\n", refOutBeg, refOutNbElement );
               TA_Free( outputBuffer );
               TA_Free( outputBufferInt );
               return TA_TESTUTIL_DRT_LOOKBACK_TOO_HIGH;
            }

            /* Is the output identical to the reference? */
            relativeIdx = outputBegIdx-refOutBeg;
            for( i=0; i < outputNbElement; i++ )
            {
               if( outputIsInteger )
               {
                  if( outputBufferInt[1+i] != refBufferInt[relativeIdx+i] )
                  {
                     printf( "Fail: doRangeTestFixSize diff data for idx=%d (%d,%d)\n", i, 
                              outputBufferInt[1+i], refBufferInt[relativeIdx+i] );
                     printf( "Fail: doRangeTestFixSize (%d,%d,%d,%d,%d)\n", startIdx, endIdx, outputBegIdx, outputNbElement, fixSize );
                     printf( "Fail: doRangeTestFixSize refOutBeg,refOutNbElement (%d,%d)\n", refOutBeg, refOutNbElement );
                     TA_Free( outputBuffer );
                     TA_Free( outputBufferInt );
                     return TA_TESTUTIL_DRT_DATA_DIFF_INT;
                  }
               }
               else
               {
                  val1 = outputBuffer[1+i];
                  val2 = refBuffer[relativeIdx+i];
                  if( !dataWithinReasonableRange( val1, val2, i, unstId, integerTolerance ) )
                  {
                     printf( "Fail: doRangeTestFixSize diff data for idx=%d (%e,%e)\n", i, val1, val2 );
                     printf( "Fail: doRangeTestFixSize (%d,%d,%d,%d,%d)\n", startIdx, endIdx, outputBegIdx, outputNbElement, fixSize );
                     printf( "Fail: doRangeTestFixSize refOutBeg,refOutNbElement (%d,%d)\n", refOutBeg, refOutNbElement );
                     if( val1 != 0.0 )
                        printf( "Fail: Diff %g %%\n", ((val2-val1)/val1)*100.0 );
                     TA_Free( outputBuffer );
                     TA_Free( outputBufferInt );
                     return TA_TESTUTIL_DRT_DATA_DIFF;
                  }
               }

               /* Randomly skip the verification of some value. Limit
                * cases are always checked though.
                */
               if( outputNbElement > 30 )
               {
                  temp = outputNbElement-20;
                  if( (i > 20) && (i < temp) )
                  {
                     /* Randomly skips from 0 to 200 verification. 
                      * Never make it skip the last 20 values.
                      */
                     i += (rand() % 200);
                     if( i > temp )
                        i = temp;
                  }
               }
            }

            /* Verify out-of-bound writing in the output buffer. */
            outputSizeByOptimalLogic = max(lookback,startIdx);
            if( outputSizeByOptimalLogic > endIdx )
               outputSizeByOptimalLogic = 0;
            else
               outputSizeByOptimalLogic = endIdx-outputSizeByOptimalLogic+1;

            if( (fixSize != outputNbElement) && (outputBuffer[1+outputSizeByOptimalLogic] != RESV_PATTERN_IMPROBABLE) )
            {
               printf( "Fail: doRangeTestFixSize out-of-bound output (%e)\n", outputBuffer[1+outputSizeByOptimalLogic] );
               printf( "Fail: doRangeTestFixSize (%d,%d,%d,%d,%d)\n", startIdx, endIdx, outputBegIdx, outputNbElement, fixSize );
               printf( "Fail: doRangeTestFixSize refOutBeg,refOutNbElement (%d,%d)\n", refOutBeg, refOutNbElement );
               TA_Free( outputBuffer );
               TA_Free( outputBufferInt );
               return TA_TESTUTIL_DRT_OUT_OF_BOUND_OUT;
            }

            if( (fixSize != outputNbElement) && (outputBufferInt[1+outputSizeByOptimalLogic] != RESV_PATTERN_IMPROBABLE_INT) )
            {
               printf( "Fail: doRangeTestFixSize out-of-bound output  (%d)\n", outputBufferInt[1+outputSizeByOptimalLogic] );
               printf( "Fail: doRangeTestFixSize (%d,%d,%d,%d,%d)\n", startIdx, endIdx, outputBegIdx, outputNbElement, fixSize );
               printf( "Fail: doRangeTestFixSize refOutBeg,refOutNbElement (%d,%d)\n", refOutBeg, refOutNbElement );
               TA_Free( outputBuffer );
               TA_Free( outputBufferInt );
               return TA_TESTUTIL_DRT_OUT_OF_BOUND_OUT_INT;
            }

            /* Verify that the memory guard were preserved. */
            if( outputBuffer[0] != RESV_PATTERN_PREFIX )
            {
               printf( "Fail: doRangeTestFixSize bad RESV_PATTERN_PREFIX (%e)\n", outputBuffer[0] );
               printf( "Fail: doRangeTestFixSize (%d,%d,%d,%d,%d)\n", startIdx, endIdx, outputBegIdx, outputNbElement, fixSize );
               printf( "Fail: doRangeTestFixSize refOutBeg,refOutNbElement (%d,%d)\n", refOutBeg, refOutNbElement );
               TA_Free( outputBuffer );
               TA_Free( outputBufferInt );
               return TA_TESTUTIL_DRT_BAD_PREFIX;
            }

            if( outputBufferInt[0] != RESV_PATTERN_PREFIX_INT )
            {
               printf( "Fail: doRangeTestFixSize bad RESV_PATTERN_PREFIX_INT (%d)\n", outputBufferInt[0] );
               printf( "Fail: doRangeTestFixSize (%d,%d,%d,%d,%d)\n", startIdx, endIdx, outputBegIdx, outputNbElement, fixSize );
               printf( "Fail: doRangeTestFixSize refOutBeg,refOutNbElement (%d,%d)\n", refOutBeg, refOutNbElement );
               TA_Free( outputBuffer );
               TA_Free( outputBufferInt );
               return TA_TESTUTIL_DRT_BAD_PREFIX;
            }

            if( outputBuffer[fixSize+1] != RESV_PATTERN_SUFFIX )
            {
               printf( "Fail: doRangeTestFixSize bad RESV_PATTERN_SUFFIX (%e)\n", outputBuffer[fixSize+1] );
               printf( "Fail: doRangeTestFixSize (%d,%d,%d,%d,%d)\n", startIdx, endIdx, outputBegIdx, outputNbElement, fixSize );
               printf( "Fail: doRangeTestFixSize refOutBeg,refOutNbElement (%d,%d)\n", refOutBeg, refOutNbElement );
               TA_Free( outputBuffer );
               TA_Free( outputBufferInt );
               return TA_TESTUTIL_DRT_BAD_SUFFIX;
            }

            if( outputBufferInt[fixSize+1] != RESV_PATTERN_SUFFIX_INT )
            {
               printf( "Fail: doRangeTestFixSize bad RESV_PATTERN_SUFFIX_INT (%d)\n", outputBufferInt[fixSize+1] );
               printf( "Fail: doRangeTestFixSize (%d,%d,%d,%d,%d)\n", startIdx, endIdx, outputBegIdx, outputNbElement, fixSize );
               printf( "Fail: doRangeTestFixSize refOutBeg,refOutNbElement (%d,%d)\n", refOutBeg, refOutNbElement );
               TA_Free( outputBuffer );
               TA_Free( outputBufferInt );
               return TA_TESTUTIL_DRT_BAD_SUFFIX;
            }

            /* Clean-up for next test. */
            if( outputIsInteger )
            {
               for( i=1; i <= fixSize; i++ )
                  outputBufferInt[i] = RESV_PATTERN_IMPROBABLE_INT;
            }
            else
            {
               for( i=1; i <= fixSize; i++ )
                  outputBuffer[i] = RESV_PATTERN_IMPROBABLE;
            }
         }

         /* Skip some startIdx at random. Limit case are still 
          * tested though.
          */
         if( (startIdx > 30) && ((startIdx+100) <= (MAX_RANGE_SIZE-fixSize)) )             
         {
            /* Randomly skips from 40 to 100 tests. */
            temp = (rand() % 100)+40;
            startIdx += temp;
         }
      }

      /* Loop and move forward for the next startIdx to test. */
   }

   TA_Free( outputBuffer );
   TA_Free( outputBufferInt );
   return TA_TEST_PASS;
}

/* This function compares two value.
 * The value is determined to be equal
 * if it is within a certain error range.
 */
static int dataWithinReasonableRange( TA_Real val1, TA_Real val2,
                                      unsigned int outputPosition,
                                      TA_FuncUnstId unstId,
                                      unsigned int integerTolerance )
{
   TA_Real difference, tolerance, temp;
   unsigned int val1_int, val2_int, tempInt, periodToIgnore;

   if( integerTolerance == TA_DO_NOT_COMPARE )
      return 1; /* Don't compare, says that everything is fine */

   /* If the function does not have an unstable period,
    * the compared value shall be identical.
    *
    * Because the algo may vary slightly allow for
    * a small epsilon error because of the nature
    * of floating point operations.
    */
   if( unstId == TA_FUNC_UNST_NONE )
      return TA_REAL_EQ( val1, val2, 0.000000001 );

   /* In the context of the TA functions, all value
    * below 0.00001 are considered equal to zero and
    * are considered to be equal within a reasonable range.
    * (the percentage difference might be large, but
    *  unsignificant at that level, so no tolerance
    *  check is being done).
    */
    if( (val1 < 0.00001) && (val2 < 0.00001) )
      return 1;

   /* When the function is unstable, the comparison
    * tolerate at first a large difference.
    *
    * As the number of "outputPosition" is higher
    * the tolerance is reduced.
    *
    * In the same way, as the unstable period
    * increase, the tolerance is reduced (that's
    * what the unstable period is for... reducing
    * difference).
    *
    * When dealing with an unstable period, the
    * first 100 values are ignored.
    *
    * Following 100, the tolerance is 
    * progressively reduced as follow:
    *
    *   1   == 0.5/1   == 50  %
    *   2   == 0.5/2   == 25  %
    *   ...
    *   100 == 0.5/100 == 0.005 %
    *   ...
    *
    * Overall, the following is a fair estimation:
    *  When using a unstable period of 200, you
    *  can expect the output to not vary more
    *  than 0.005 %
    *
    * The logic is sligthly different if the 
    * output are rounded integer, but it is
    * the same idea.
    *
    * The following describe the special meaning of
    * the integerTolerance:
    *
    * Value 10      -> A tolerance of 1/10  is used.
    *
    * Value 100     -> A tolerance of 1/100 is used.
    *
    * Value 1000    -> A tolerance of 1/1000 is used.
    * 
    * Value 360     -> Useful when the output are
    *                  degrees. In that case, a fix
    *                  tolerance of 1 degree is used.
    *
    * Value TA_DO_NOT_COMPARE -> 
    *                  Indicate that NO COMPARISON take
    *                  place. This is useful for functions
    *                  that cannot be compare when changing
    *                  the range (like the accumulative
    *                  algorithm used for TA_AD and TA_ADOSC).
    */


   /* Some functions requires a longer unstable period.
    * These are trap here.
    */
   switch( unstId )
   {
   case TA_FUNC_UNST_T3:
      periodToIgnore = 200;
      break;
   default:
      periodToIgnore = 100;
      break;
   }

   if( integerTolerance == 1000 )
   {
      /* Check for no difference of more
       * than 1/1000
       */
      if( val1 > val2 )
         difference = (val1-val2);
      else
         difference = (val2-val1);

      difference *= 1000.0;

      temp = outputPosition+TA_GetUnstablePeriod(unstId)+1;
      if( temp <= periodToIgnore )
      {
         /* Pretend it is fine. */
         return 1;
      }
      else if( (int)difference > 1 )
      {
         printf( "\nFail: Value diffferent by more than 1/1000 (%f)\n", difference );
         return 0;
      }
   }
   else if( integerTolerance == 100 )
   {
      /* Check for no difference of more
       * than 1/1000
       */
      if( val1 > val2 )
         difference = (val1-val2);
      else
         difference = (val2-val1);

      difference *= 100.0;

      temp = outputPosition+TA_GetUnstablePeriod(unstId)+1;
      if( temp <= periodToIgnore )
      {
         /* Pretend it is fine. */
         return 1;
      }
      else if( (int)difference > 1 )
      {
         printf( "\nFail: Value diffferent by more than 1/100 (%f)\n", difference );
         return 0;
      }
   }
   else if( integerTolerance == 10 )
   {
      /* Check for no difference of more
       * than 1/1000
       */
      if( val1 > val2 )
         difference = (val1-val2);
      else
         difference = (val2-val1);

      difference *= 10.0;

      temp = outputPosition+TA_GetUnstablePeriod(unstId)+1;
      if( temp <= periodToIgnore )
      {
         /* Pretend it is fine. */
         return 1;
      }
      else if( (int)difference > 1 )
      {
         printf( "\nFail: Value diffferent by more than 1/10 (%f)\n", difference );
         return 0;
      }
   }
   else if( integerTolerance == 360 )
   {
      /* Check for no difference of no more
       * than 10% when the value is higher than
       * 1 degree.
       *
       * Difference of less than 1 degree are not significant.
       */
      val1_int = (unsigned int)val1;
      val2_int = (unsigned int)val2;
      if( val1_int > val2_int )
         tempInt = val1_int - val2_int;
      else
         tempInt = val2_int - val1_int;

      if( val1 > val2 )
         difference = (val1-val2)/val1;
      else
         difference = (val2-val1)/val2;

      temp = outputPosition+TA_GetUnstablePeriod(unstId)+1;
      if( temp <= periodToIgnore )
      {
         /* Pretend it is fine. */
         return 1;
      }
      else if( (tempInt > 1) && (difference > 0.10) )
      {
         printf( "\nFail: Value diffferent by more than 10 percent over 1 degree (%d)\n", tempInt );
         return 0;
      }       
   }
   else if( integerTolerance )
   {
      /* Check that the integer part of the value
       * is not different more than the specified
       * integerTolerance.
       */
      val1_int = (unsigned int)val1;
      val2_int = (unsigned int)val2;
      if( val1_int > val2_int )
         tempInt = val1_int - val2_int;
      else
         tempInt = val2_int - val1_int;

      temp = outputPosition+TA_GetUnstablePeriod(unstId)+1;
      if( temp <= periodToIgnore )
      {
         /* Pretend it is fine. */
         return 1;
      }
      else if( temp < 100 )
      {
         if( tempInt >= 3*integerTolerance )
         {
            printf( "\nFail: Value out of 3*tolerance range (%d,%d)\n", tempInt, integerTolerance );
            return 0; /* Value considered different */
         }
      }
      else if( temp < 150 )
      {
         if( tempInt >= 2*integerTolerance )
         {
            printf( "\nFail: Value out of 2*tolerance range (%d,%d)\n", tempInt, integerTolerance );
            return 0; /* Value considered different */
         }
      }
      else if( temp < 200 )
      {
         if( tempInt >= integerTolerance )
         {
            printf( "\nFail: Value out of tolerance range (%d,%d)\n", tempInt, integerTolerance );
            return 0; /* Value considered different */
         }
      }
      else if( tempInt >= 1 )
      {
         printf( "\nFail: Value not equal (difference is %d)\n", tempInt );
         return 0; /* Value considered different */
      } 
   }
   else
   {
      if( val1 > val2 )
         difference = (val1-val2)/val1;
      else
         difference = (val2-val1)/val2;
     
      temp = outputPosition+TA_GetUnstablePeriod(unstId)+1;
      if( temp <= periodToIgnore )
      {
         /* Pretend it is fine. */
         return 1;
      }
      else
      {
         temp -= periodToIgnore;
         tolerance = 0.5/temp;
      }
 
      if( difference > tolerance )
      {
         printf( "\nFail: Value out of tolerance range (%g,%g)\n", difference, tolerance );
         return 0; /* Out of tolerance... values are not equal. */
      }
   }

   return 1; /* Value equal within tolerance. */
}

static TA_RetCode CallTestFunction( RangeTestFunction testFunction,
                                    TA_Integer    startIdx,
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
   /* Call the function and do profiling. */
   TA_RetCode retCode;
   double clockDelta;

#ifdef WIN32
   LARGE_INTEGER startClock;
   LARGE_INTEGER endClock;
#else
   clock_t startClock;
   clock_t endClock;
#endif

#ifdef WIN32
   QueryPerformanceCounter(&startClock);
#else
   startClock = clock();
#endif
	retCode = testFunction( startIdx,
                  endIdx,
                  outputBuffer,
                  outputBufferInt,
                  outBegIdx,
                  outNbElement,
                  lookback,
                  opaqueData,
                  outputNb,
                  isOutputInteger );   

	/* Profile only functions producing at least 20 values. */
	if( *outNbElement < 20 )
	{
		return retCode;
	}

#ifdef WIN32
   QueryPerformanceCounter(&endClock);
   clockDelta = (double)((__int64)endClock.QuadPart - (__int64) startClock.QuadPart);
#else
   endClock = clock();
   clockDelta = (double)(endClock - startClock);
#endif

   if( clockDelta <= 0 )
   {
	   insufficientClockPrecision = 1;	   
   }
   else
   {	   
      if( clockDelta > worstProfiledCall )
         worstProfiledCall = clockDelta;
      timeInProfiledCall += clockDelta;
      nbProfiledCall++;
   }
   
   return retCode;
}
