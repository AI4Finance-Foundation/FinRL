
# ta_regrest is used for testing and should
# not be installed.
noinst_PROGRAMS = gen_code

gen_code_SOURCES = gen_code.c

gen_code_CPPFLAGS = -I../../ta_common
gen_code_LDFLAGS = -L../../ta_common -L../../ta_abstract -L../../ta_func
gen_code_LDADD = -lta_common -lta_abstract_gc -lta_func -lm

all-local:
	$(MAKE) $(AM_MAKEFLAGS) gen_code
	cp gen_code ../../../bin
