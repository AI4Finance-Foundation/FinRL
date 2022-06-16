#!/bin/sh
echo aclocal
aclocal || exit
echo autoheader
autoheader || exit
echo libtoolize --copy --force
libtoolize --copy --force || exit
echo automake -a -c
automake -a -c || exit
echo autoconf
autoconf || exit
