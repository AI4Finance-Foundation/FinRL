#ifndef TA_GLOBAL_H
#define TA_GLOBAL_H

#ifndef TA_COMMON_H
   #include "ta_common.h"
#endif

#ifndef TA_FUNC_H
   #include "ta_func.h"
#endif

/* TA_CandleSetting is the one setting struct */
typedef struct {
    TA_CandleSettingType    settingType;
    TA_RangeType            rangeType;
    int                     avgPeriod;
    double                  factor;
} TA_CandleSetting;

/* This interface is used exclusively INTERNALY to the TA-LIB.
 * There is nothing for the end-user here ;->
 */

/* Provides functionality for managing global ressource
 * throughout the TA-LIB.
 *
 * Since not all module are used/link in the application,
 * the ta_common simply provides the mechanism for the module
 * to optionnaly "register" its initialization/shutdown
 * function.
 *
 * A function shall access its global variable by calling 
 * TA_GetGlobal. This function will appropriatly call the 
 * initialization function if its global are not yet initialized.
 * 
 * The call of the init and shutdown function are guaranteed
 * to be multithread protected. It is also guarantee that
 * these function will always get called in alternance (in
 * other word, following an initialization only a shutdown
 * can get called).
 */

typedef enum
{
   /* Module will be shutdown in the order specified here. */
   	
   TA_ABSTRACTION_GLOBAL_ID,
   TA_FUNC_GLOBAL_ID,
   TA_MEMORY_GLOBAL_ID, /* Must be last.        */
   TA_NB_GLOBAL_ID
} TA_GlobalModuleId;

typedef TA_RetCode (*TA_GlobalInitFunc)    ( void **globalToAlloc );
typedef TA_RetCode (*TA_GlobalShutdownFunc)( void *globalAllocated );

typedef struct
{
   const TA_GlobalModuleId     id;
   const TA_GlobalInitFunc     init;
   const TA_GlobalShutdownFunc shutdown;
} TA_GlobalControl;

TA_RetCode TA_GetGlobal( const TA_GlobalControl * const control,
                         void **global );

/* Occasionaly, code tracing must be disable.
 * Example:
 *  - The memory module needs to know if the tracing is
 *    still enabled or not when freeing memory on shutdown.
 *  - We do not want to recursively trace while the tracing
 *    function themselves gets called ;->
 */
int  TA_IsTraceEnabled( void );
void TA_TraceEnable   ( void );
void TA_TraceDisable  ( void );

/* If enabled by the user, use a local drive
 * for configuration and/or temporary file.
 * TA-LIB must NEVER assume such local drive 
 * is available.
 */
const char *TA_GetLocalCachePath( void );

typedef struct
{
  unsigned int initialize;
  const TA_GlobalControl * control;
  void *global;
} TA_ModuleControl;

/* This is the hidden implementation of TA_Libc. */
typedef struct
{
   unsigned int magicNb; /* Unique identifier of this object. */
   TA_ModuleControl moduleControl[TA_NB_GLOBAL_ID];

   unsigned int traceEnabled;
   unsigned int stdioEnabled;
   FILE *stdioFile;

   const char *localCachePath;

   /* For handling the compatibility with other software */
   TA_Compatibility compatibility;

   /* For handling the unstable period of some TA function. */
   unsigned int unstablePeriod[TA_FUNC_UNST_ALL];

   /* For handling the candlestick global settings */
   TA_CandleSetting candleSettings[TA_AllCandleSettings];

} TA_LibcPriv;

/* The following global is used all over the place 
 * and is the entry point for all other globals.
 */
extern TA_LibcPriv *TA_Globals;

#endif
