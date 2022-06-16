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


/* TA_ParamHolderPriv is the private implementation of a TA_ParamHolder. */

/* Definition in this header shall be used only internaly by the
 * ta_abstract module.
 * End-user of the TA-LIB shall never attempt to access these
 * structure directly.
 */

#ifndef TA_FRAME_PRIV_H
#define TA_FRAME_PRIV_H

#ifndef TA_ABSTRACT_H
   #include "ta_abstract.h"
#endif

#ifndef TA_MAGIC_NB_H
   #include "ta_magic_nb.h"
#endif

typedef struct
{
   const TA_Real      *open;
   const TA_Real      *high;
   const TA_Real      *low;
   const TA_Real      *close;
   const TA_Real      *volume;
   const TA_Real      *openInterest;
} TA_PricePtrs;

typedef struct
{
   union TA_ParamHolderInputData
   {
      const TA_Real      *inReal;
      const TA_Integer   *inInteger;
      TA_PricePtrs        inPrice;
   } data;

   const TA_InputParameterInfo *inputInfo;

} TA_ParamHolderInput;

typedef struct
{
   union TA_ParamHolderOptInData
   {
      TA_Integer optInInteger;
      TA_Real    optInReal;
   } data;

   const TA_OptInputParameterInfo *optInputInfo;

} TA_ParamHolderOptInput;

typedef struct
{
   union TA_ParamHolderOutputData
   {
      TA_Real        *outReal;
      TA_Integer     *outInteger;
   } data;

   const TA_OutputParameterInfo *outputInfo;
} TA_ParamHolderOutput;

typedef struct
{
   /* Magic number is used to detect internal error. */
   unsigned int magicNumber;

   TA_ParamHolderInput    *in;
   TA_ParamHolderOptInput *optIn;
   TA_ParamHolderOutput   *out;

   /* Indicate which parameter have been initialized.
    * The LSB (Less Significant Bit) is the first parameter
    * and a bit equal to '1' indicate that the parameter is
    * not initialized.
    */
   unsigned int inBitmap;
   unsigned int outBitmap;

   const TA_FuncInfo *funcInfo;
} TA_ParamHolderPriv;

typedef TA_RetCode (*TA_FrameFunction)( const TA_ParamHolderPriv *params,
                                        TA_Integer  startIdx,
                                        TA_Integer  endIdx,
                                        TA_Integer *outBegIdx,
                                        TA_Integer *outNbElement );

typedef unsigned int (*TA_FrameLookback)( const TA_ParamHolderPriv *params );

#endif
