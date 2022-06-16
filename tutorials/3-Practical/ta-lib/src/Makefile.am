AUTOMAKE_OPTIONS = foreign 1.4
SUBDIRS = ta_abstract ta_common ta_func

lib_LTLIBRARIES = libta_lib.la

libta_lib_la_SOURCES = 

libta_lib_la_LIBADD = \
  ta_abstract/libta_abstract.la \
  ta_func/libta_func.la \
  ta_common/libta_common.la

libta_lib_la_LDFLAGS = -version-info $(TALIB_LIBRARY_VERSION)

