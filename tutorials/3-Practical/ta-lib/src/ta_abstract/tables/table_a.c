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

/*********************************************************************
 * This file contains only TA functions starting with the letter 'A' *
 *********************************************************************/
#include <stddef.h>
#include "ta_abstract.h"
#include "ta_def_ui.h"

/* Follow the 3 steps defined below for adding a new TA Function to this
 * file.
 */

/****************************************************************************
 * Step 1 - Define here the interface to your TA functions with
 *          the macro DEF_FUNCTION.
 *
 ****************************************************************************/

/* ACOS BEGIN */
DEF_MATH_UNARY_OPERATOR( ACOS, "Vector Trigonometric ACos", "Acos" )
/* ACOS END */

/* AD BEGIN */
static const TA_InputParameterInfo    *TA_AD_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_HLCV,
  NULL
};

static const TA_OutputParameterInfo   *TA_AD_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_AD_OptInputs[] =
{
  NULL
};

DEF_FUNCTION( AD,                         /* name */
              TA_GroupId_VolumeIndicators,   /* groupId */
              "Chaikin A/D Line", /* hint */
              "Ad",                         /* CamelCase name */
              0                             /* flags */
             );
/* AD END */

/* ADD BEGIN */
DEF_MATH_BINARY_OPERATOR( ADD, "Vector Arithmetic Add", "Add" )
/* ADD END */

/* ADOSC BEGIN */
static const TA_OptInputParameterInfo TA_DEF_UI_FastADOSC_Period =
{
   TA_OptInput_IntegerRange, /* type */
   "optInFastPeriod",        /* paramName */
   0,                        /* flags */

   "Fast Period",            /* displayName */
   (const void *)&TA_DEF_TimePeriod_Positive_Minimum2, /* dataSet */
   3, /* defaultValue */
   "Number of period for the fast MA", /* hint */

   NULL /* CamelCase name */
};

static const TA_OptInputParameterInfo TA_DEF_UI_SlowADOSC_Period =
{
   TA_OptInput_IntegerRange, /* type */
   "optInSlowPeriod",        /* paramName */
   0,                        /* flags */

   "Slow Period",            /* displayName */
   (const void *)&TA_DEF_TimePeriod_Positive_Minimum2, /* dataSet */
   10, /* defaultValue */
   "Number of period for the slow MA", /* hint */

   NULL /* CamelCase name */
};

static const TA_InputParameterInfo    *TA_ADOSC_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_HLCV,
  NULL
};

static const TA_OutputParameterInfo   *TA_ADOSC_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_ADOSC_OptInputs[] =
{  
  &TA_DEF_UI_FastADOSC_Period,
  &TA_DEF_UI_SlowADOSC_Period,
  NULL
};

DEF_FUNCTION( ADOSC,                         /* name */
              TA_GroupId_VolumeIndicators,   /* groupId */
              "Chaikin A/D Oscillator", /* hint */
              "AdOsc",                  /* CamelCase name */
              0                         /* flags */
             );
/* ADOSC END */

/* ADX BEGIN */
static const TA_InputParameterInfo    *TA_ADX_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_HLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_ADX_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_ADX_OptInputs[] =
{ &TA_DEF_UI_TimePeriod_14_MINIMUM2,
  NULL
};

DEF_FUNCTION( ADX,                          /* name */
              TA_GroupId_MomentumIndicators,   /* groupId */
              "Average Directional Movement Index", /* hint */
              "Adx",                         /* CamelCase name */
              TA_FUNC_FLG_UNST_PER          /* flags */
             );
/* ADX END */

/* ADXR BEGIN */
static const TA_InputParameterInfo    *TA_ADXR_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_HLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_ADXR_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_ADXR_OptInputs[] =
{ &TA_DEF_UI_TimePeriod_14_MINIMUM2,
  NULL
};

DEF_FUNCTION( ADXR,                         /* name */
              TA_GroupId_MomentumIndicators,   /* groupId */
              "Average Directional Movement Index Rating", /* hint */
			  "Adxr",                      /* CamelCase name */
              TA_FUNC_FLG_UNST_PER          /* flags */
             );
/* ADXR END */

/* APO BEGIN */
static const TA_InputParameterInfo *TA_APO_Inputs[] =
{
  &TA_DEF_UI_Input_Real,
  NULL
};

static const TA_OutputParameterInfo   *TA_APO_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_APO_OptInputs[] =
{ &TA_DEF_UI_Fast_Period,
  &TA_DEF_UI_Slow_Period,
  &TA_DEF_UI_MA_Method,
  NULL
};

DEF_FUNCTION( APO,                         /* name */
              TA_GroupId_MomentumIndicators,  /* groupId */
              "Absolute Price Oscillator", /* hint */
              "Apo",                       /* CamelCase name */
              0                            /* flags */
             );
/* APO END */

/* AROON BEGIN */
const TA_OutputParameterInfo TA_DEF_UI_Output_Real_AroonUp =
                               { TA_Output_Real, "outAroonDown", TA_OUT_DASH_LINE };

const TA_OutputParameterInfo TA_DEF_UI_Output_Real_AroonDown =
                                { TA_Output_Real, "outAroonUp", TA_OUT_LINE };

static const TA_InputParameterInfo    *TA_AROON_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_HL,
  NULL
};

static const TA_OutputParameterInfo   *TA_AROON_Outputs[]   =
{
  &TA_DEF_UI_Output_Real_AroonUp,
  &TA_DEF_UI_Output_Real_AroonDown,
  NULL
};

static const TA_OptInputParameterInfo *TA_AROON_OptInputs[] = 
{ 
  &TA_DEF_UI_TimePeriod_14_MINIMUM2,
  NULL
};

DEF_FUNCTION( AROON,                          /* name */
              TA_GroupId_MomentumIndicators,  /* groupId */
              "Aroon",                        /* hint */
              "Aroon",                        /* CamelCase name */
              0                               /* flags */              
             );

/* AROON END */

/* AROONOSC BEGIN */
static const TA_InputParameterInfo    *TA_AROONOSC_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_HL,
  NULL
};

static const TA_OutputParameterInfo   *TA_AROONOSC_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_AROONOSC_OptInputs[] = 
{ 
  &TA_DEF_UI_TimePeriod_14_MINIMUM2,
  NULL
};

DEF_FUNCTION( AROONOSC,                       /* name */
              TA_GroupId_MomentumIndicators,  /* groupId */
              "Aroon Oscillator",             /* hint */
			  "AroonOsc",                     /* CamelCase name */
              0                               /* flags */
             );

/* AROONOSC END */

/* ASIN BEGIN */
DEF_MATH_UNARY_OPERATOR( ASIN, "Vector Trigonometric ASin", "Asin" )
/* ASIN END */

/* ATAN BEGIN */
DEF_MATH_UNARY_OPERATOR( ATAN, "Vector Trigonometric ATan", "Atan" )
/* ATAN END */

/* ATR BEGIN */
static const TA_InputParameterInfo    *TA_ATR_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_HLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_ATR_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_ATR_OptInputs[] =
{ &TA_DEF_UI_TimePeriod_14,
  NULL
};

DEF_FUNCTION( ATR,                        /* name */
              TA_GroupId_VolatilityIndicators, /* groupId */
              "Average True Range",       /* hint */
              "Atr",                      /* CamelCase name */
              TA_FUNC_FLG_UNST_PER        /* flags */
             );
/* ATR END */

/* AVGPRICE BEGIN */
static const TA_InputParameterInfo    *TA_AVGPRICE_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_AVGPRICE_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_AVGPRICE_OptInputs[] = { NULL };

DEF_FUNCTION( AVGPRICE,                   /* name */
              TA_GroupId_PriceTransform,  /* groupId */
              "Average Price",            /* hint */
              "AvgPrice",                 /* CamelCase name */
              TA_FUNC_FLG_OVERLAP         /* flags */
             );
/* AVGPRICE END */

/****************************************************************************
 * Step 2 - Add your TA function to the table.
 *          Keep in alphabetical order. Must be NULL terminated.
 ****************************************************************************/
const TA_FuncDef *TA_DEF_TableA[] =
{
   ADD_TO_TABLE(ACOS),
   ADD_TO_TABLE(AD),
   ADD_TO_TABLE(ADD),
   ADD_TO_TABLE(ADOSC),
   ADD_TO_TABLE(ADX),
   ADD_TO_TABLE(ADXR),
   ADD_TO_TABLE(APO),
   ADD_TO_TABLE(AROON),
   ADD_TO_TABLE(AROONOSC),
   ADD_TO_TABLE(ASIN),
   ADD_TO_TABLE(ATAN),
   ADD_TO_TABLE(ATR),
   ADD_TO_TABLE(AVGPRICE),
   NULL
};


/* Do not modify the following line. */
const unsigned int TA_DEF_TableASize =
              ((sizeof(TA_DEF_TableA)/sizeof(TA_FuncDef *))-1);


/****************************************************************************
 * Step 3 - Make sure "gen_code" is executed for generating all other
 *          source files derived from this one.
 *          You can then re-compile the library as usual and you are done!
 ****************************************************************************/
