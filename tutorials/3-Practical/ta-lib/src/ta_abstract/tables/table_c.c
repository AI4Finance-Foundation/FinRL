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
 * This file contains only TA functions starting with the letter 'C' *
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

/* CCI BEGIN */
static const TA_InputParameterInfo    *TA_CCI_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_HLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CCI_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_CCI_OptInputs[] =
{ &TA_DEF_UI_TimePeriod_14_MINIMUM2,
  NULL
};

DEF_FUNCTION( CCI,                           /* name */
              TA_GroupId_MomentumIndicators, /* groupId */
              "Commodity Channel Index",     /* hint */
              "Cci",                         /* CamelCase name */
              0                              /* flags */             );

/* CCI END */

/* CDL2CROWS BEGIN */
static const TA_InputParameterInfo    *TA_CDL2CROWS_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDL2CROWS_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDL2CROWS_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDL2CROWS,                      /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Two Crows",                    /* hint */
              "Cdl2Crows",                    /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
			 );

/* CDL2CROWS END */

/* CDL3BLACKCROWS BEGIN */
static const TA_InputParameterInfo    *TA_CDL3BLACKCROWS_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDL3BLACKCROWS_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDL3BLACKCROWS_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDL3BLACKCROWS,                 /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Three Black Crows",            /* hint */
              "Cdl3BlackCrows",               /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
			);

/* CDL3BLACKCROWS END */

/* CDL3INSIDE BEGIN */
static const TA_InputParameterInfo    *TA_CDL3INSIDE_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDL3INSIDE_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDL3INSIDE_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDL3INSIDE,                     /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Three Inside Up/Down",         /* hint */
              "Cdl3Inside",                   /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
			);

/* CDL3INSIDE END */

/* CDL3LINESTRIKE BEGIN */
static const TA_InputParameterInfo    *TA_CDL3LINESTRIKE_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDL3LINESTRIKE_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDL3LINESTRIKE_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDL3LINESTRIKE,                 /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Three-Line Strike ",           /* hint */
              "Cdl3LineStrike",               /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */              
            );

/* CDL3LINESTRIKE END */

/* CDL3OUTSIDE BEGIN */
static const TA_InputParameterInfo    *TA_CDL3OUTSIDE_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDL3OUTSIDE_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDL3OUTSIDE_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDL3OUTSIDE,                    /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Three Outside Up/Down",        /* hint */
              "Cdl3Outside",                  /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
			);

/* CDL3OUTSIDE END */

/* CDL3STARSINSOUTH BEGIN */
static const TA_InputParameterInfo    *TA_CDL3STARSINSOUTH_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDL3STARSINSOUTH_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDL3STARSINSOUTH_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDL3STARSINSOUTH,               /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Three Stars In The South",     /* hint */
              "Cdl3StarsInSouth",             /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */              
            );

/* CDL3STARSINSOUTH END */

/* CDL3WHITESOLDIERS BEGIN */
static const TA_InputParameterInfo    *TA_CDL3WHITESOLDIERS_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDL3WHITESOLDIERS_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDL3WHITESOLDIERS_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDL3WHITESOLDIERS,              /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Three Advancing White Soldiers", /* hint */
              "Cdl3WhiteSoldiers",            /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
			);

/* CDL3WHITESOLDIERS END */

/* CDLABANDONEDBABY BEGIN */
static const TA_InputParameterInfo    *TA_CDLABANDONEDBABY_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLABANDONEDBABY_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLABANDONEDBABY_OptInputs[] =
{ 
  &TA_DEF_UI_Penetration_30,
  NULL
};

DEF_FUNCTION( CDLABANDONEDBABY,               /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Abandoned Baby",               /* hint */
              "CdlAbandonedBaby",             /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
		    );

/* CDLABANDONEDBABY END */

/* CDLADVANCEBLOCK BEGIN */
static const TA_InputParameterInfo    *TA_CDLADVANCEBLOCK_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLADVANCEBLOCK_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLADVANCEBLOCK_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLADVANCEBLOCK,                /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Advance Block",                /* hint */
              "CdlAdvanceBlock",              /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
			);

/* CDLADVANCEBLOCK END */

/* CDLBELTHOLD BEGIN */
static const TA_InputParameterInfo    *TA_CDLBELTHOLD_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLBELTHOLD_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLBELTHOLD_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLBELTHOLD,                    /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Belt-hold",                    /* hint */
              "CdlBeltHold",                  /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
			);

/* CDLBELTHOLD END */

/* CDLBREAKAWAY BEGIN */
static const TA_InputParameterInfo    *TA_CDLBREAKAWAY_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLBREAKAWAY_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLBREAKAWAY_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLBREAKAWAY,                   /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Breakaway",                    /* hint */
              "CdlBreakaway",                 /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
			);

/* CDLBREAKAWAY END */

/* CDLCLOSINGMARUBOZU BEGIN */
static const TA_InputParameterInfo    *TA_CDLCLOSINGMARUBOZU_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLCLOSINGMARUBOZU_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLCLOSINGMARUBOZU_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLCLOSINGMARUBOZU,             /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Closing Marubozu",             /* hint */
              "CdlClosingMarubozu",           /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
            );

/* CDLCLOSINGMARUBOZU END */

/* CDLCONCEALBABYSWALL BEGIN */
static const TA_InputParameterInfo    *TA_CDLCONCEALBABYSWALL_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLCONCEALBABYSWALL_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLCONCEALBABYSWALL_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLCONCEALBABYSWALL,            /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Concealing Baby Swallow",      /* hint */
			  "CdlConcealBabysWall",          /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLCONCEALBABYSWALL END */

/* CDLCOUNTERATTACK BEGIN */
static const TA_InputParameterInfo    *TA_CDLCOUNTERATTACK_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLCOUNTERATTACK_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLCOUNTERATTACK_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLCOUNTERATTACK,               /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Counterattack",                /* hint */
              "CdlCounterAttack",             /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLCOUNTERATTACK END */

/* CDLDARKCLOUDCOVER BEGIN */
static const TA_InputParameterInfo    *TA_CDLDARKCLOUDCOVER_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLDARKCLOUDCOVER_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLDARKCLOUDCOVER_OptInputs[] =
{ 
  &TA_DEF_UI_Penetration_50,
  NULL
};

DEF_FUNCTION( CDLDARKCLOUDCOVER,              /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Dark Cloud Cover",             /* hint */
			  "CdlDarkCloudCover",            /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLDARKCLOUDCOVER END */

/* CDLDOJI BEGIN */
static const TA_InputParameterInfo    *TA_CDLDOJI_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLDOJI_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLDOJI_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLDOJI,                        /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Doji",                         /* hint */
              "CdlDoji",                      /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLDOJI END */

/* CDLDOJISTAR BEGIN */
static const TA_InputParameterInfo    *TA_CDLDOJISTAR_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLDOJISTAR_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLDOJISTAR_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLDOJISTAR,                    /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Doji Star",                    /* hint */
              "CdlDojiStar",                  /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLDOJISTAR END */

/* CDLDRAGONFLYDOJI BEGIN */
static const TA_InputParameterInfo    *TA_CDLDRAGONFLYDOJI_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLDRAGONFLYDOJI_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLDRAGONFLYDOJI_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLDRAGONFLYDOJI,               /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Dragonfly Doji",               /* hint */
              "CdlDragonflyDoji",             /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLDRAGONFLYDOJI END */

/* CDLENGULFING BEGIN */
static const TA_InputParameterInfo    *TA_CDLENGULFING_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLENGULFING_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLENGULFING_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLENGULFING,                   /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Engulfing Pattern",            /* hint */
              "CdlEngulfing",                 /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLENGULFING END */

/* CDLEVENINGDOJISTAR BEGIN */
static const TA_InputParameterInfo    *TA_CDLEVENINGDOJISTAR_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLEVENINGDOJISTAR_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLEVENINGDOJISTAR_OptInputs[] =
{ 
  &TA_DEF_UI_Penetration_30,
  NULL
};

DEF_FUNCTION( CDLEVENINGDOJISTAR,             /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Evening Doji Star",            /* hint */
              "CdlEveningDojiStar",           /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLEVENINGDOJISTAR END */

/* CDLEVENINGSTAR BEGIN */
static const TA_InputParameterInfo    *TA_CDLEVENINGSTAR_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLEVENINGSTAR_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLEVENINGSTAR_OptInputs[] =
{ 
  &TA_DEF_UI_Penetration_30,
  NULL
};

DEF_FUNCTION( CDLEVENINGSTAR,                 /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Evening Star",                 /* hint */
              "CdlEveningStar",               /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLEVENINGSTAR END */

/* CDLEVENINGDOJISTAR END */

/* CDLGAPSIDESIDEWHITE BEGIN */
static const TA_InputParameterInfo    *TA_CDLGAPSIDESIDEWHITE_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLGAPSIDESIDEWHITE_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLGAPSIDESIDEWHITE_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLGAPSIDESIDEWHITE,            /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Up/Down-gap side-by-side white lines",   /* hint */
			  "CdlGapSideSideWhite",         /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK        /* flags */
             );

/* CDLGAPSIDESIDEWHITE END */

/* CDLGRAVESTONEDOJI BEGIN */
static const TA_InputParameterInfo    *TA_CDLGRAVESTONEDOJI_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLGRAVESTONEDOJI_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLGRAVESTONEDOJI_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLGRAVESTONEDOJI,              /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Gravestone Doji",              /* hint */
              "CdlGravestoneDoji",            /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLGRAVESTONEDOJI END */

/* CDLHAMMER BEGIN */
static const TA_InputParameterInfo    *TA_CDLHAMMER_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLHAMMER_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLHAMMER_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLHAMMER,                      /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Hammer",                       /* hint */
              "CdlHammer",                    /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLHAMMER END */

/* CDLHANGINGMAN BEGIN */
static const TA_InputParameterInfo    *TA_CDLHANGINGMAN_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLHANGINGMAN_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLHANGINGMAN_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLHANGINGMAN,                  /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Hanging Man",                  /* hint */
              "CdlHangingMan",                /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLHANGINGMAN END */

/* CDLHARAMI BEGIN */
static const TA_InputParameterInfo    *TA_CDLHARAMI_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLHARAMI_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLHARAMI_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLHARAMI,                      /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Harami Pattern",               /* hint */
              "CdlHarami",                    /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLHARAMI END */

/* CDLHARAMICROSS BEGIN */
static const TA_InputParameterInfo    *TA_CDLHARAMICROSS_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLHARAMICROSS_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLHARAMICROSS_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLHARAMICROSS,                 /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Harami Cross Pattern",         /* hint */
              "CdlHaramiCross",               /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLHARAMICROSS END */

/* CDLHIGHWAVE BEGIN */
static const TA_InputParameterInfo    *TA_CDLHIGHWAVE_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLHIGHWAVE_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLHIGHWAVE_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLHIGHWAVE,                    /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "High-Wave Candle",             /* hint */
              "CdlHignWave",                  /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLHIGHWAVE END */

/* CDLHIKKAKE BEGIN */
static const TA_InputParameterInfo    *TA_CDLHIKKAKE_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLHIKKAKE_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLHIKKAKE_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLHIKKAKE,                     /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Hikkake Pattern",              /* hint */
              "CdlHikkake",                   /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLHIKKAKE END */

/* CDLHIKKAKEMOD BEGIN */
static const TA_InputParameterInfo    *TA_CDLHIKKAKEMOD_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLHIKKAKEMOD_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLHIKKAKEMOD_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLHIKKAKEMOD,                  /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Modified Hikkake Pattern",     /* hint */
              "CdlHikkakeMod",                /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLHIKKAKEMOD END */

/* CDLHOMINGPIGEON BEGIN */
static const TA_InputParameterInfo    *TA_CDLHOMINGPIGEON_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLHOMINGPIGEON_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLHOMINGPIGEON_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLHOMINGPIGEON,                /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Homing Pigeon",                /* hint */
              "CdlHomingPigeon",              /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLHOMINGPIGEON END */

/* CDLIDENTICAL3CROWS BEGIN */
static const TA_InputParameterInfo    *TA_CDLIDENTICAL3CROWS_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLIDENTICAL3CROWS_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLIDENTICAL3CROWS_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLIDENTICAL3CROWS,             /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Identical Three Crows",        /* hint */
              "CdlIdentical3Crows",           /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLIDENTICAL3CROWS END */

/* CDLINNECK BEGIN */
static const TA_InputParameterInfo    *TA_CDLINNECK_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLINNECK_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLINNECK_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLINNECK,                      /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "In-Neck Pattern",              /* hint */
              "CdlInNeck",                    /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLINNECK END */

/* CDLINVERTEDHAMMER BEGIN */
static const TA_InputParameterInfo    *TA_CDLINVERTEDHAMMER_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLINVERTEDHAMMER_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLINVERTEDHAMMER_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLINVERTEDHAMMER,              /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Inverted Hammer",              /* hint */
              "CdlInvertedHammer",            /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLINVERTEDHAMMER END */

/* CDLKICKING BEGIN */
static const TA_InputParameterInfo    *TA_CDLKICKING_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLKICKING_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLKICKING_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLKICKING,                     /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Kicking",                      /* hint */
              "CdlKicking",                   /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLKICKING END */

/* CDLKICKINGBYLENGTH BEGIN */
static const TA_InputParameterInfo    *TA_CDLKICKINGBYLENGTH_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLKICKINGBYLENGTH_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLKICKINGBYLENGTH_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLKICKINGBYLENGTH,             /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Kicking - bull/bear determined by the longer marubozu",              /* hint */
              "CdlKickingByLength",           /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLKICKINGBYLENGTH END */

/* CDLLADDERBOTTOM BEGIN */
static const TA_InputParameterInfo    *TA_CDLLADDERBOTTOM_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLLADDERBOTTOM_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLLADDERBOTTOM_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLLADDERBOTTOM,                /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Ladder Bottom",                /* hint */
              "CdlLadderBottom",              /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLLADDERBOTTOM END */

/* CDLLONGLEGGEDDOJI BEGIN */
static const TA_InputParameterInfo    *TA_CDLLONGLEGGEDDOJI_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLLONGLEGGEDDOJI_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLLONGLEGGEDDOJI_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLLONGLEGGEDDOJI,              /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Long Legged Doji",             /* hint */
              "CdlLongLeggedDoji",            /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLLONGLEGGEDDOJI END */

/* CDLLONGLINE BEGIN */
static const TA_InputParameterInfo    *TA_CDLLONGLINE_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLLONGLINE_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLLONGLINE_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLLONGLINE,                    /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Long Line Candle",             /* hint */
              "CdlLongLine",                  /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLLONGLINE END */

/* CDLMARUBOZU BEGIN */
static const TA_InputParameterInfo    *TA_CDLMARUBOZU_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLMARUBOZU_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLMARUBOZU_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLMARUBOZU,                    /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Marubozu",                     /* hint */
              "CdlMarubozu",                 /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK        /* flags */
             );

/* CDLMARUBOZU END */

/* CDLMATCHINGLOW BEGIN */
static const TA_InputParameterInfo    *TA_CDLMATCHINGLOW_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLMATCHINGLOW_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLMATCHINGLOW_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLMATCHINGLOW,                 /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Matching Low",                 /* hint */
			  "CdlMatchingLow",               /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLMATCHINGLOW END */

/* CDLMATHOLD BEGIN */
static const TA_InputParameterInfo    *TA_CDLMATHOLD_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLMATHOLD_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLMATHOLD_OptInputs[] =
{ 
  &TA_DEF_UI_Penetration_50,
  NULL
};

DEF_FUNCTION( CDLMATHOLD,                     /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Mat Hold",                     /* hint */
              "CdlMatHold",                   /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK        /* flags */
             );

/* CDLMATHOLD END */

/* CDLMORNINGDOJISTAR BEGIN */
static const TA_InputParameterInfo    *TA_CDLMORNINGDOJISTAR_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLMORNINGDOJISTAR_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLMORNINGDOJISTAR_OptInputs[] =
{ 
  &TA_DEF_UI_Penetration_30,
  NULL
};

DEF_FUNCTION( CDLMORNINGDOJISTAR,             /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Morning Doji Star",            /* hint */
              "CdlMorningDojiStar",           /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLMORNINGDOJISTAR END */

/* CDLMORNINGSTAR BEGIN */
static const TA_InputParameterInfo    *TA_CDLMORNINGSTAR_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLMORNINGSTAR_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLMORNINGSTAR_OptInputs[] =
{ 
  &TA_DEF_UI_Penetration_30,
  NULL
};

DEF_FUNCTION( CDLMORNINGSTAR,                 /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Morning Star",                 /* hint */
              "CdlMorningStar",               /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLMORNINGSTAR END */

/* CDLONNECK BEGIN */
static const TA_InputParameterInfo    *TA_CDLONNECK_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLONNECK_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLONNECK_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLONNECK,                      /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "On-Neck Pattern",              /* hint */
			  "CdlOnNeck",                    /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLONNECK END */

/* CDLPIERCING BEGIN */
static const TA_InputParameterInfo    *TA_CDLPIERCING_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLPIERCING_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLPIERCING_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLPIERCING,                    /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Piercing Pattern",             /* hint */
              "CdlPiercing",                  /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLPIERCING END */

/* CDLRICKSHAWMAN BEGIN */
static const TA_InputParameterInfo    *TA_CDLRICKSHAWMAN_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLRICKSHAWMAN_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLRICKSHAWMAN_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLRICKSHAWMAN,                 /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Rickshaw Man",                 /* hint */
              "CdlRickshawMan",               /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLRICKSHAWMAN END */

/* CDLRISEFALL3METHODS BEGIN */
static const TA_InputParameterInfo    *TA_CDLRISEFALL3METHODS_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLRISEFALL3METHODS_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLRISEFALL3METHODS_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLRISEFALL3METHODS,            /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Rising/Falling Three Methods", /* hint */
              "CdlRiseFall3Methods",          /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLRISEFALL3METHODS END */

/* CDLSEPARATINGLINES BEGIN */
static const TA_InputParameterInfo    *TA_CDLSEPARATINGLINES_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLSEPARATINGLINES_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLSEPARATINGLINES_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLSEPARATINGLINES,             /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Separating Lines",             /* hint */
              "CdlSeperatingLines",           /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLSEPARATINGLINES END */

/* CDLSHOOTINGSTAR BEGIN */
static const TA_InputParameterInfo    *TA_CDLSHOOTINGSTAR_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLSHOOTINGSTAR_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLSHOOTINGSTAR_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLSHOOTINGSTAR,                /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Shooting Star",                /* hint */
              "CdlShootingStar",              /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLSHOOTINGSTAR END */

/* CDLSHORTLINE BEGIN */
static const TA_InputParameterInfo    *TA_CDLSHORTLINE_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLSHORTLINE_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLSHORTLINE_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLSHORTLINE,                   /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Short Line Candle",            /* hint */
			  "CdlShortLine",                 /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLSHORTLINE END */

/* CDLSPINNINGTOP BEGIN */
static const TA_InputParameterInfo    *TA_CDLSPINNINGTOP_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLSPINNINGTOP_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLSPINNINGTOP_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLSPINNINGTOP,                 /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Spinning Top",                 /* hint */
              "CdlSpinningTop",               /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLSPINNINGTOP END */

/* CDLSTALLEDPATTERN BEGIN */
static const TA_InputParameterInfo    *TA_CDLSTALLEDPATTERN_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLSTALLEDPATTERN_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLSTALLEDPATTERN_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLSTALLEDPATTERN,              /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Stalled Pattern",              /* hint */
              "CdlStalledPattern",            /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLSTALLEDPATTERN END */

/* CDLSTICKSANDWICH BEGIN */
static const TA_InputParameterInfo    *TA_CDLSTICKSANDWICH_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLSTICKSANDWICH_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLSTICKSANDWICH_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLSTICKSANDWICH,               /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Stick Sandwich",               /* hint */
			  "CdlStickSandwhich",            /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLSTICKSANDWICH END */

/* CDLTAKURI BEGIN */
static const TA_InputParameterInfo    *TA_CDLTAKURI_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLTAKURI_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLTAKURI_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLTAKURI,                      /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Takuri (Dragonfly Doji with very long lower shadow)",    /* hint */
              "CdlTakuri",                   /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK        /* flags */
             );

/* CDLTAKURI END */

/* CDLTASUKIGAP BEGIN */
static const TA_InputParameterInfo    *TA_CDLTASUKIGAP_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLTASUKIGAP_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLTASUKIGAP_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLTASUKIGAP,                   /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Tasuki Gap",                   /* hint */
              "CdlTasukiGap",                 /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLTASUKIGAP END */

/* CDLTHRUSTING BEGIN */
static const TA_InputParameterInfo    *TA_CDLTHRUSTING_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLTHRUSTING_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLTHRUSTING_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLTHRUSTING,                   /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Thrusting Pattern",            /* hint */
              "CdlThrusting",                 /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLTHRUSTING END */

/* CDLTRISTAR BEGIN */
static const TA_InputParameterInfo    *TA_CDLTRISTAR_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLTRISTAR_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLTRISTAR_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLTRISTAR,                     /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Tristar Pattern",              /* hint */
              "CdlTristar",                   /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLTRISTAR END */

/* CDLUNIQUE3RIVER BEGIN */
static const TA_InputParameterInfo    *TA_CDLUNIQUE3RIVER_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLUNIQUE3RIVER_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLUNIQUE3RIVER_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLUNIQUE3RIVER,                /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Unique 3 River",               /* hint */
              "CdlUnique3River",              /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLUNIQUE3RIVER END */

/* CDLUPSIDEGAP2CROWS BEGIN */
static const TA_InputParameterInfo    *TA_CDLUPSIDEGAP2CROWS_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLUPSIDEGAP2CROWS_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLUPSIDEGAP2CROWS_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLUPSIDEGAP2CROWS,             /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Upside Gap Two Crows",         /* hint */
              "CdlUpsideGap2Crows",           /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK         /* flags */
             );

/* CDLUPSIDEGAP2CROWS END */

/* CDLXSIDEGAP3METHODS BEGIN */
static const TA_InputParameterInfo    *TA_CDLXSIDEGAP3METHODS_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_CDLXSIDEGAP3METHODS_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_CDLXSIDEGAP3METHODS_OptInputs[] =
{ 
  NULL
};

DEF_FUNCTION( CDLXSIDEGAP3METHODS,            /* name */
              TA_GroupId_PatternRecognition,  /* groupId */
              "Upside/Downside Gap Three Methods",    /* hint */
              "CdlXSideGap3Methods",         /* CamelCase name */
              TA_FUNC_FLG_CANDLESTICK        /* flags */
             );

/* CDLXSIDEGAP3METHODS END */

/* CEIL BEGIN */
DEF_MATH_UNARY_OPERATOR( CEIL, "Vector Ceil", "Ceil" )
/* CEIL END */

/* CMO BEGIN */
static const TA_InputParameterInfo    *TA_CMO_Inputs[]    =
{
  &TA_DEF_UI_Input_Real,
  NULL
};

static const TA_OutputParameterInfo   *TA_CMO_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_CMO_OptInputs[] =
{
  &TA_DEF_UI_TimePeriod_14_MINIMUM2,
  NULL
};

DEF_FUNCTION( CMO,                        /* name */
              TA_GroupId_MomentumIndicators,  /* groupId */
              "Chande Momentum Oscillator",  /* hint */
              "Cmo",                      /* CamelCase name */
              TA_FUNC_FLG_UNST_PER       /* flags */
             );
/* CMO END */

/* CORREL BEGIN */
static const TA_InputParameterInfo    *TA_CORREL_Inputs[]    =
{
  &TA_DEF_UI_Input_Real0,
  &TA_DEF_UI_Input_Real1,
  NULL
};

static const TA_OutputParameterInfo   *TA_CORREL_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_CORREL_OptInputs[] =
{ &TA_DEF_UI_TimePeriod_30,
  NULL
};

DEF_FUNCTION( CORREL,                      /* name */
              TA_GroupId_Statistic,     /* groupId */
              "Pearson's Correlation Coefficient (r)", /* hint */
              "Correl",                /* CamelCase name */
              0                        /* flags */
             );
/* CORREL END */

/* COS BEGIN */
DEF_MATH_UNARY_OPERATOR( COS, "Vector Trigonometric Cos", "Cos" )
/* COS END */

/* COSH BEGIN */
DEF_MATH_UNARY_OPERATOR( COSH, "Vector Trigonometric Cosh", "Cosh" )
/* COSH END */

/****************************************************************************
 * Step 2 - Add your TA function to the table.
 *          Keep in alphabetical order. Must be NULL terminated.
 ****************************************************************************/
const TA_FuncDef *TA_DEF_TableC[] =
{
   ADD_TO_TABLE(CCI),
   ADD_TO_TABLE(CDL2CROWS),
   ADD_TO_TABLE(CDL3BLACKCROWS),
   ADD_TO_TABLE(CDL3INSIDE),
   ADD_TO_TABLE(CDL3LINESTRIKE),
   ADD_TO_TABLE(CDL3OUTSIDE),
   ADD_TO_TABLE(CDL3STARSINSOUTH),
   ADD_TO_TABLE(CDL3WHITESOLDIERS),
   ADD_TO_TABLE(CDLABANDONEDBABY),
   ADD_TO_TABLE(CDLADVANCEBLOCK),
   ADD_TO_TABLE(CDLBELTHOLD),
   ADD_TO_TABLE(CDLBREAKAWAY),
   ADD_TO_TABLE(CDLCLOSINGMARUBOZU),
   ADD_TO_TABLE(CDLCONCEALBABYSWALL),
   ADD_TO_TABLE(CDLCOUNTERATTACK),
   ADD_TO_TABLE(CDLDARKCLOUDCOVER),
   ADD_TO_TABLE(CDLDOJI),
   ADD_TO_TABLE(CDLDOJISTAR),
   ADD_TO_TABLE(CDLDRAGONFLYDOJI),
   ADD_TO_TABLE(CDLENGULFING),
   ADD_TO_TABLE(CDLEVENINGDOJISTAR),
   ADD_TO_TABLE(CDLEVENINGSTAR),
   ADD_TO_TABLE(CDLGAPSIDESIDEWHITE),
   ADD_TO_TABLE(CDLGRAVESTONEDOJI),
   ADD_TO_TABLE(CDLHAMMER),
   ADD_TO_TABLE(CDLHANGINGMAN),
   ADD_TO_TABLE(CDLHARAMI),
   ADD_TO_TABLE(CDLHARAMICROSS),
   ADD_TO_TABLE(CDLHIGHWAVE),
   ADD_TO_TABLE(CDLHIKKAKE),
   ADD_TO_TABLE(CDLHIKKAKEMOD),
   ADD_TO_TABLE(CDLHOMINGPIGEON),
   ADD_TO_TABLE(CDLIDENTICAL3CROWS),
   ADD_TO_TABLE(CDLINNECK),
   ADD_TO_TABLE(CDLINVERTEDHAMMER),
   ADD_TO_TABLE(CDLKICKING),
   ADD_TO_TABLE(CDLKICKINGBYLENGTH),
   ADD_TO_TABLE(CDLLADDERBOTTOM),
   ADD_TO_TABLE(CDLLONGLEGGEDDOJI),
   ADD_TO_TABLE(CDLLONGLINE),
   ADD_TO_TABLE(CDLMARUBOZU),
   ADD_TO_TABLE(CDLMATCHINGLOW),
   ADD_TO_TABLE(CDLMATHOLD),
   ADD_TO_TABLE(CDLMORNINGDOJISTAR),
   ADD_TO_TABLE(CDLMORNINGSTAR),
   ADD_TO_TABLE(CDLONNECK),
   ADD_TO_TABLE(CDLPIERCING),
   ADD_TO_TABLE(CDLRICKSHAWMAN),
   ADD_TO_TABLE(CDLRISEFALL3METHODS),
   ADD_TO_TABLE(CDLSEPARATINGLINES),
   ADD_TO_TABLE(CDLSHOOTINGSTAR),
   ADD_TO_TABLE(CDLSHORTLINE),
   ADD_TO_TABLE(CDLSPINNINGTOP),  
   ADD_TO_TABLE(CDLSTALLEDPATTERN),  
   ADD_TO_TABLE(CDLSTICKSANDWICH),  
   ADD_TO_TABLE(CDLTAKURI),
   ADD_TO_TABLE(CDLTASUKIGAP),
   ADD_TO_TABLE(CDLTHRUSTING),
   ADD_TO_TABLE(CDLTRISTAR),
   ADD_TO_TABLE(CDLUNIQUE3RIVER), 
   ADD_TO_TABLE(CDLUPSIDEGAP2CROWS), 
   ADD_TO_TABLE(CDLXSIDEGAP3METHODS), 
   ADD_TO_TABLE(CEIL),
   ADD_TO_TABLE(CMO),
   ADD_TO_TABLE(CORREL),
   ADD_TO_TABLE(COS),
   ADD_TO_TABLE(COSH),
   NULL
};


/* Do not modify the following line. */
const unsigned int TA_DEF_TableCSize =
              ((sizeof(TA_DEF_TableC)/sizeof(TA_FuncDef *))-1);


/****************************************************************************
 * Step 3 - Make sure "gen_code" is executed for generating all other
 *          source files derived from this one.
 *          You can then re-compile the library as usual and you are done!
 ****************************************************************************/
