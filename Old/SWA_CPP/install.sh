g++  -c -fPIC swa.cc -o swa.o
g++ -shared  -Wl,-soname,bridge_swa.so -o bridge_swa.so swa.o
rm -f Solvers/SWA/bridge_swa.o Solvers/SWA/swa.o