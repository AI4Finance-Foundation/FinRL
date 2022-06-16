
noinst_LTLIBRARIES = libta_common.la
AM_CPPFLAGS = -I../ta_common/

libta_common_la_SOURCES = ta_global.c \
	ta_retcode.c \
	ta_version.c 

libta_common_la_LDFLAGS = -version-info $(TALIB_LIBRARY_VERSION)

libta_commondir=$(includedir)/ta-lib/
libta_common_HEADERS = ../../include/ta_defs.h \
	../../include/ta_libc.h \
	../../include/ta_common.h

