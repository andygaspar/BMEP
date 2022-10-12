g++ -D_REENTRANT -c -fPIC -fopenmp  -O3 -fPIC -Xpreprocessor  -I${XPRESSDIR}/include/ -L${XPRESSDIR}/lib -lxprs -lxprb main_test.cpp -o main.o
g++ -D_REENTRANT -shared -fopenmp -Xpreprocessor  -I${XPRESSDIR}/include/ -L${XPRESSDIR}/lib  -Wl,-soname,bridge.so -o bridge.so main.o -shared -lxprs -lxprb

rm -f main.o