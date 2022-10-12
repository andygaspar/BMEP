#include <mutex>
#include <algorithm>
#include <queue>
#include <array>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <vector>
#include <chrono>
#include <thread>
#include <random>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include "xprb_cpp.h"
#include "xprs.h"

/* NAMESPACES */
using namespace std;
using std::scientific;
using std::fixed;
using namespace ::dashoptimization;

/* DEFINITIONS & CONSTANTS */
#define CURRENTVERSION 	   "7.9 - Golden Master"
#define CURRENTVERSIONDATE "November 27 2020"

#define MIN(X, Y) (((X) <= (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) >= (Y)) ? (X) : (Y))

#define MAX_INT    0x7FFFFFFF					//int infinity
#define MAX_DOUBLE 0x1.FFFFFFFFFFFFFP+1023		//double infinity
#define INF 	   MAX_DOUBLE					//infinity - it is set to the double one
// #define MAX_TIME   3600						//maximum running time to be considered 3h
#define MAX_TIME   -1							//no max running time
#define PRECISION  16                           //output precision
#define MAX_TOLL   0.0000000001                 //10^-10


struct EDGE{int i; int j;};
struct OPTIMUM_STRUCT{
	double tree_len;								 //Length of the best so far tree
	vector <struct EDGE> tree; 						 //Best so far tree encoded as a list of edges
	long long OverallNumber_of_Primal_Solutions;	 //...
	vector <vector <struct EDGE> > primal_bound_list;
};

struct NODE{
	vector <struct EDGE> partial_tree;
	int start_edge;
	int end_edge;
	int taxon;
	bool empty;
};


struct STATS{
	bool      outoftime;		/* Check if the algorithm is running out of time   */
	long long nodecounter;		/*        		  B&B Node Counter				 */
	long long int_sol_counter;	/*     		 Integer solution counter    		 */
	long long LPCounter;      	/*  			   LP calls executed 				 */
	long long DualCounter;      /*  			   Dual calls executed 				 */
	double    PardiRootLB;		/*  		    Pardi's root lower bound 			 */
	double    LPRootLB;			/*  			 LP root lower bound 				 */
	double    RootLB;			/* Root LB: LP or Pardi depending on bound type 	 */
	double    RootUB;			/*  			   Root upper bound 				 */
	double    GAP;		  		/*			  	  Root Gap     				   	 */
	double 	  start_time;       /*  start time (sec) of the global search			 */
	double    end_time;         /*  	end time (sec) of the global search			 */
	bool	  core_runnnig;     /*     Specifies if the core is idle or running    */
};

mutex m; 											 // Global locker for mutual exclusive access to data
ifstream in;

class GlobalVars{
public:
	char separator;
	int nameWidth;
	int numWidth;
	int numWidth2;
	double TOLL;
	double EPS;
	OPTIMUM_STRUCT Optimum;  		  					 //Global Optimum
	queue<NODE> Queue;
	int 	  boundtype;			/*  1: Pardi's one; 2: LP + Lagrangiano; */
	int n_taxa;
	double**  dist;			/*  Distance Matrix Pointer */
	STATS    *StatsArray; 	 	/*  Statistics array, one entry per core. By default, core 0 stores RootLB and UB */
	double   *precomputedpow;       /*  precomputed powers 2^-k  */
	double   *precomputedpowVH;     /*  precomputed powers 2^-k/n  */
	double 	 *beta;	   	    /*  Pardi's beta_k vector	*/
	int 	  verbose;            /*  Control how much output to write in the console */
	long 	  OverallCoreNumber;	/*  Overall number of cores available in the machine */

	bool 	  output_for_experiment;
	bool 	  VinhHaeseler;
	int       order=0;
	string    namefile;

	GlobalVars() {
	separator    = ' ';
	nameWidth     = 30;
	numWidth      = 25;
	numWidth2     = 12;
	TOLL=1;
	EPS=0.5;
	boundtype=2;			/*  1: Pardi's one; 2: LP + Lagrangiano; */
	n_taxa = 9;
	dist=NULL;			/*  Distance Matrix Pointer */
	StatsArray=NULL; 	 	/*  Statistics array, one entry per core. By default, core 0 stores RootLB and UB */
	beta=NULL;	   	    /*  Pardi's beta_k vector	*/
	verbose=1;            /*  Control how much output to write in the console */
	output_for_experiment=false;
	VinhHaeseler = false;
    order=0;										 // Global locker for mutual exclusive access to data
	precomputedpow = NULL;       /*  precomputed powers 2^-k  */
	precomputedpowVH = NULL;     /*  precomputed powers 2^-k/n  */
	beta = NULL;	   	    /*  Pardi's beta_k vector	*/
	verbose =0;            /*  Control how much output to write in the console */
	OverallCoreNumber = 16;	/*  Overall number of cores available in the machine */
    namefile = " ";
	};

	GlobalVars(double* d, int n){
		separator    = ' ';
		nameWidth     = 30;
		numWidth      = 25;
		numWidth2     = 12;
		TOLL=1;
		EPS=0.5;
		boundtype=2;			/*  1: Pardi's one; 2: LP + Lagrangiano; */
		StatsArray=NULL; 	 	/*  Statistics array, one entry per core. By default, core 0 stores RootLB and UB */
		beta=NULL;	   	    /*  Pardi's beta_k vector	*/
		verbose=1;            /*  Control how much output to write in the console */
		output_for_experiment=false;
		VinhHaeseler = false;
		order=0;										 // Global locker for mutual exclusive access to data
		precomputedpow = NULL;       /*  precomputed powers 2^-k  */
		precomputedpowVH = NULL;     /*  precomputed powers 2^-k/n  */
		beta = NULL;	   	    /*  Pardi's beta_k vector	*/
		verbose =0;            /*  Control how much output to write in the console */
		OverallCoreNumber = 16;	/*  Overall number of cores available in the machine */
		namefile = " ";
		n_taxa = n;

		dist=new double*[n_taxa+1];
		for(register unsigned int i=0; i<=n_taxa;i++) dist[i] = new double[n_taxa+1];
		for (int i=0; i< n_taxa + 1; i++) dist[0][i] = 0;
		for (int i=1; i< n_taxa + 1; i++) {
			dist[i][0] = 0;
			for (int j=1; j< n_taxa + 1; j++) dist[i][j] = d[(i-1)*n_taxa + j -1];
		}
    };
	~GlobalVars() {}

	bool Out_of_Time(){if(MAX_TIME ==-1) return false; else return (omp_get_wtime()-StatsArray[0].start_time > MAX_TIME) ? true : false;}
	bool ThereAreIdleCores(int cores){lock_guard<mutex> lg(m); for( unsigned int i=0; i<cores; i++) if(StatsArray[i].core_runnnig==false) return true; return false;}
	bool AllIdleCores(int cores){lock_guard<mutex> lg(m); for( unsigned int i=0; i<cores; i++) if(StatsArray[i].core_runnnig==true) return false; return true;}
};