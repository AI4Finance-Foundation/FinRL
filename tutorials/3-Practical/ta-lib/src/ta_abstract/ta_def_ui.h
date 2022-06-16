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

#ifndef TA_DEF_UI_H
#define TA_DEF_UI_H

/* Offer pre-defined user interface constant.
 *
 * This allows to avoid to duplicate static data for
 * common user interface elements.
 */

#ifndef TA_ABSTRACT_H
   #include "ta_abstract.h"
#endif

#ifndef TA_FRAME_PRIV_H
   #include "ta_frame_priv.h"
#endif

#if !defined(TA_GEN_CODE) && !defined( TA_FRAME_H )
   #include "ta_frame.h"
#endif

typedef enum
{
  /* If you modify this enum, make sure you update ta_def_ui.c */
  TA_GroupId_MathOperators,
  TA_GroupId_MathTransform,
  TA_GroupId_OverlapStudies,
  TA_GroupId_VolatilityIndicators,
  TA_GroupId_MomentumIndicators,
  TA_GroupId_CycleIndicators,
  TA_GroupId_VolumeIndicators,
  TA_GroupId_PatternRecognition,
  TA_GroupId_Statistic,
  TA_GroupId_PriceTransform,
  TA_NB_GROUP_ID
} TA_GroupId;

extern const char TA_GroupId_MathOperatorsString[];
extern const char TA_GroupId_MathTransformString[];
extern const char TA_GroupId_OverlapStudiesString[];
extern const char TA_GroupId_VolatilityIndicatorsString[];
extern const char TA_GroupId_MomentumIndicatorsString[];
extern const char TA_GroupId_CycleIndicatorsString[];
extern const char TA_GroupId_VolumeIndicatorsString[];
extern const char TA_GroupId_PatternRecognitionString[];
extern const char TA_GroupId_StatisticString[];
extern const char TA_GroupId_PriceTransformString[];

extern const char *TA_GroupString[TA_NB_GROUP_ID];

/* Inputs */
extern const TA_InputParameterInfo TA_DEF_UI_Input_Real;
extern const TA_InputParameterInfo TA_DEF_UI_Input_Real0;
extern const TA_InputParameterInfo TA_DEF_UI_Input_Real1;
extern const TA_InputParameterInfo TA_DEF_UI_Input_Integer;
extern const TA_InputParameterInfo TA_DEF_UI_Input_Price_OHLCV;
extern const TA_InputParameterInfo TA_DEF_UI_Input_Price_HLCV;
extern const TA_InputParameterInfo TA_DEF_UI_Input_Price_OHLC;
extern const TA_InputParameterInfo TA_DEF_UI_Input_Price_HLC;
extern const TA_InputParameterInfo TA_DEF_UI_Input_Price_HL;
extern const TA_InputParameterInfo TA_DEF_UI_Input_Price_CV;
extern const TA_InputParameterInfo TA_DEF_UI_Input_Price_V;
extern const TA_InputParameterInfo TA_DEF_UI_Input_Periods;


/* Outputs. */
extern const TA_OutputParameterInfo TA_DEF_UI_Output_Real;
extern const TA_OutputParameterInfo TA_DEF_UI_Output_Integer;
extern const TA_OutputParameterInfo TA_DEF_UI_Output_Lines;

/* Optional Inputs. */
extern const TA_OptInputParameterInfo TA_DEF_UI_TimePeriod_30;
extern const TA_OptInputParameterInfo TA_DEF_UI_TimePeriod_14;
extern const TA_OptInputParameterInfo TA_DEF_UI_TimePeriod_10;
extern const TA_OptInputParameterInfo TA_DEF_UI_TimePeriod_5;

extern const TA_OptInputParameterInfo TA_DEF_UI_TimePeriod_30_MINIMUM2;
extern const TA_OptInputParameterInfo TA_DEF_UI_TimePeriod_21_MINIMUM2;
extern const TA_OptInputParameterInfo TA_DEF_UI_TimePeriod_14_MINIMUM2;
extern const TA_OptInputParameterInfo TA_DEF_UI_TimePeriod_14_MINIMUM5;
extern const TA_OptInputParameterInfo TA_DEF_UI_TimePeriod_10_MINIMUM2;
extern const TA_OptInputParameterInfo TA_DEF_UI_TimePeriod_5_MINIMUM2;

extern const TA_OptInputParameterInfo TA_DEF_UI_VerticalShift;
extern const TA_OptInputParameterInfo TA_DEF_UI_HorizontalShift;

extern const TA_OptInputParameterInfo TA_DEF_UI_MA_Method;
extern const TA_OptInputParameterInfo TA_DEF_UI_Fast_Period;
extern const TA_OptInputParameterInfo TA_DEF_UI_Slow_Period;

extern const TA_OptInputParameterInfo TA_DEF_UI_NbDeviation;

extern const TA_OptInputParameterInfo TA_DEF_UI_Penetration_30;
extern const TA_OptInputParameterInfo TA_DEF_UI_Penetration_50;

extern const TA_OptInputParameterInfo TA_DEF_UI_MinPeriod;
extern const TA_OptInputParameterInfo TA_DEF_UI_MaxPeriod;

/* Re-usable ranges. */
extern const TA_IntegerRange TA_DEF_TimePeriod_Positive;
extern const TA_IntegerRange TA_DEF_TimePeriod_Positive_Minimum2;
extern const TA_IntegerRange TA_DEF_TimePeriod_Positive_Minimum5;

extern const TA_RealRange    TA_DEF_VerticalShiftPercent;
extern const TA_IntegerRange TA_DEF_HorizontalShiftPeriod;
extern const TA_RealRange    TA_DEF_NbDeviation;
extern const TA_RealRange    TA_DEF_ZeroToOne;
extern const TA_RealRange    TA_DEF_RealPositive;

/* Useful to build your own TA_DEF_UI with the list of
 * implemented Moving Average type.
 */
extern const TA_IntegerList TA_MA_TypeList;

/* An internal structure for coordinating all these const info.
 * One TA_FuncDef instance will exist for each TA function.
 */
typedef struct
{
   /* Magic number is used to detect internal error. */
   const unsigned int magicNumber;

   /* The function will belong to this group. */
   const TA_GroupId groupId;

   /* Some more info. */
   const TA_FuncInfo * const funcInfo;

   /* Description of the parameters. */
   const TA_InputParameterInfo    * const input;
   const TA_OptInputParameterInfo * const optInput;
   const TA_OutputParameterInfo   * const output;

   /* Entry point of the TA function and its lookback function. */
   const TA_FrameFunction function;
   const TA_FrameLookback lookback;
} TA_FuncDef;

/* The following MACROs are helpers being used in
 * the tables\table<a..z>.c files.
 */
#if !defined( TA_GEN_CODE )
   /* This definition is used when compiling the end-user library. */
   #define DEF_FUNCTION( name, \
                         groupId, \
                         hint, \
                         camelCaseName, \
                         flags ) \
   \
   TA_FuncInfo TA_INFO_##name; \
   \
   const TA_FuncDef TA_DEF_##name = \
   { \
      TA_FUNC_DEF_MAGIC_NB, \
      groupId, \
      &TA_INFO_##name, \
      (const TA_InputParameterInfo    * const)&TA_##name##_Inputs[0],    \
      (const TA_OptInputParameterInfo * const)&TA_##name##_OptInputs[0], \
      (const TA_OutputParameterInfo   * const)&TA_##name##_Outputs[0],   \
      TA_##name##_FramePP, \
      TA_##name##_FramePPLB \
   }; \
   TA_FuncInfo TA_INFO_##name = \
   { \
      (const char * const)#name, \
      (const char * const)groupId##String, \
      (const char * const)hint, \
      (const char * const)camelCaseName, \
      (const int)flags, \
      (sizeof(TA_##name##_Inputs)   / sizeof(TA_InputParameterInfo *))   - 1, \
      (sizeof(TA_##name##_OptInputs)/ sizeof(TA_OptInputParameterInfo *))- 1, \
      (sizeof(TA_##name##_Outputs)  / sizeof(TA_OutputParameterInfo *))  - 1, \
      (const TA_FuncHandle * const)&TA_DEF_##name \
   };
#else
   /* This definition is used only when compiling for gencode.
    * Because some pointers may not be defined before gencode
    * is run at least once, some value are set to NULL.
    */
   #define DEF_FUNCTION( name, \
                         groupId, \
                         hint, \
                         camelCaseName, \
                         flags ) \
   \
   TA_FuncInfo TA_INFO_##name; \
   \
   const TA_FuncDef TA_DEF_##name = \
   { \
      TA_FUNC_DEF_MAGIC_NB, \
      groupId, \
      &TA_INFO_##name, \
      (const TA_InputParameterInfo    * const)&TA_##name##_Inputs[0],    \
      (const TA_OptInputParameterInfo * const)&TA_##name##_OptInputs[0], \
      (const TA_OutputParameterInfo   * const)&TA_##name##_Outputs[0],   \
      NULL, \
      NULL \
   }; \
   TA_FuncInfo TA_INFO_##name = \
   { \
      (const char * const)#name, \
      (const char * const)groupId##String, \
      (const char * const)hint, \
      (const char * const)camelCaseName, \
      (const int)flags, \
      (sizeof(TA_##name##_Inputs)   / sizeof(TA_InputParameterInfo *))   - 1, \
      (sizeof(TA_##name##_OptInputs)/ sizeof(TA_OptInputParameterInfo *))- 1, \
      (sizeof(TA_##name##_Outputs)  / sizeof(TA_OutputParameterInfo *))  - 1, \
      (const TA_FuncHandle * const)&TA_DEF_##name \
   };
#endif

#define ADD_TO_TABLE(name) &TA_DEF_##name

/* Utility Macro to quickly define Math operator operating on one array
 * with no optional parameter.
 */
#define DEF_MATH_UNARY_OPERATOR(NAME,HINT,CAMELCASENAME) \
	static const TA_InputParameterInfo    *TA_##NAME##_Inputs[]    = \
{ \
  &TA_DEF_UI_Input_Real, \
  NULL \
}; \
static const TA_OutputParameterInfo   *TA_##NAME##_Outputs[]   = \
{ \
  &TA_DEF_UI_Output_Real, \
  NULL \
}; \
	static const TA_OptInputParameterInfo *TA_##NAME##_OptInputs[] = { NULL }; \
DEF_FUNCTION( NAME, /* name */ \
              TA_GroupId_MathTransform, /* groupId */ \
              HINT, /* hint */ \
              CAMELCASENAME,  /* CamelCase name */ \
              0        /* flags */ \
             );

#define DEF_MATH_BINARY_OPERATOR(NAME,HINT,CAMELCASENAME) \
	static const TA_InputParameterInfo    *TA_##NAME##_Inputs[]    = \
{ \
  &TA_DEF_UI_Input_Real0, \
  &TA_DEF_UI_Input_Real1, \
  NULL \
}; \
static const TA_OutputParameterInfo   *TA_##NAME##_Outputs[]   = \
{ \
  &TA_DEF_UI_Output_Real, \
  NULL \
}; \
	static const TA_OptInputParameterInfo *TA_##NAME##_OptInputs[] = { NULL }; \
DEF_FUNCTION( NAME, /* name */ \
              TA_GroupId_MathOperators, /* groupId */ \
              HINT, /* hint */ \
              CAMELCASENAME,  /* CamelCase name */ \
              0        /* flags */ \
             );

#endif

