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

//These constants are just used for formatting console output text
const char separator    = ' ';  
const int nameWidth     = 30;
const int numWidth      = 25;
const int numWidth2     = 12;
double TOLL=1;
double EPS=0.5;

/* Global variables */
mutex m; 											 // Global locker for mutual exclusive access to data

struct EDGE{int i; int j;};
struct OPTIMUM_STRUCT{
	double tree_len;								 //Length of the best so far tree 
	vector <struct EDGE> tree; 						 //Best so far tree encoded as a list of edges
	long long OverallNumber_of_Primal_Solutions;	 //...
	vector <vector <struct EDGE> > primal_bound_list;
};

OPTIMUM_STRUCT Optimum;  		  					 //Global Optimum

struct NODE{
	vector <struct EDGE> partial_tree;   
	int start_edge;
	int end_edge;
	int taxon;
	bool empty;
};

queue<NODE> Queue; 

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

int 	  boundtype=2;			/*  1: Pardi's one; 2: LP + Lagrangiano; */
double  **dist=NULL;			/*  Distance Matrix Pointer */ 	
STATS    *StatsArray=NULL; 	 	/*  Statistics array, one entry per core. By default, core 0 stores RootLB and UB */

double   *precomputedpow;       /*  precomputed powers 2^-k  */
double   *precomputedpowVH;     /*  precomputed powers 2^-k/n  */
double 	 *beta=NULL;	   	    /*  Pardi's beta_k vector	*/	 
int 	  verbose=1;            /*  Control how much output to write in the console */
long 	  OverallCoreNumber;	/*  Overall number of cores available in the machine */

bool 	  output_for_experiment=false;
bool 	  VinhHaeseler = false;
int       order=0;
string    namefile;

bool Out_of_Time(){if(MAX_TIME ==-1) return false; else return (omp_get_wtime()-StatsArray[0].start_time > MAX_TIME) ? true : false;}
bool ThereAreIdleCores(int cores){lock_guard<mutex> lg(m); for( unsigned int i=0; i<cores; i++) if(StatsArray[i].core_runnnig==false) return true; return false;}
bool AllIdleCores(int cores){lock_guard<mutex> lg(m); for( unsigned int i=0; i<cores; i++) if(StatsArray[i].core_runnnig==true) return false; return true;}

/* My Classes */
#include "InputHandler.cpp"
#include "OutputHandler.cpp"
#include "Solver.cpp"
InputHandler *IH;

void Precompute_Pardi_Beta_Array( unsigned int n){
  for( unsigned int k=4; k<=n; k++){ 
    beta[k]=INF;
    for( unsigned int i=1; i<=k-2; i++)
   	  for( unsigned int j=i+1; j<=k-1; j++)
	 	if (beta[k] > dist[i][k]+dist[j][k]-dist[i][j]) beta[k] = dist[i][k]+dist[j][k]-dist[i][j];
  }	
}

void GlobalSearch(int n, int core){
	Solver *CoreSolver=new Solver(core,n);
	while(true){
		NODE node=CoreSolver->RetrieveNode_and_Delete_from_Queue();
		if(node.empty==false){
			StatsArray[core].core_runnnig=true;
			CoreSolver->SetNode_and_Run(&node);
			StatsArray[core].core_runnnig=false;

		}

		if(Queue.empty()==true && AllIdleCores(OverallCoreNumber)==true) break;
	}
	delete CoreSolver;
}

int* make_adj_mat(vector <struct EDGE>tree){
    int n = (int)tree.size();
    cout<<"n "<<n<<endl;
    for(int i=0; i< n; i++) cout<<tree[i].i<<' '<<tree[i].j<<endl;
    int n_mat = n + 1;
    int* adj_mat = new int[n_mat * n_mat];
    for(int i = 0; i < n_mat; i++) {for(int j=0; j<n_mat; j++) adj_mat[i*n_mat +j] = 0;}
    for(register unsigned int k=0; k<n; k++){
        adj_mat[(tree[k].i - 1)*n_mat + tree[k].j - 1] = 1;
        adj_mat[(tree[k].j - 1)*n_mat + tree[k].i - 1] = 1;

    }
    return adj_mat;
	}

int* Initializer_and_Terminator(double* d, int taxa, bool rescaling, int Max_Cores_To_USE){
	int n;

	#pragma omp master
	{
		n=taxa; 
		
		//IH->ReadDistanceMatrix(datafile,namefile,taxa,dist,n,rescaling);
		dist = IH -> set_d(d, taxa);
				for (int i=0; i< n+1; i++) {
			for (int j=0; j < n+1; j++) {
				cout<<dist[i][j]<<" ";
			}
			cout<<endl;
		}
		precomputedpow=new double[n+1]; 
		precomputedpowVH=new double[n+1]; 
		for( unsigned int i=0; i<=n; i++){precomputedpow[i]=pow(2.0,i); precomputedpowVH[i]=pow(2.0,(double)i/n);} //1.2446
		beta=new double[n+1]; 
		Precompute_Pardi_Beta_Array(n);	
		
		IH->MachineEpsilon();
		// TOLL=MAX_TOLL;
		cout<<"Setting tolerance to "<<TOLL<<endl;	
		IH->StabilityChecker(n,dist);
		//cout<<"Running entropic analysis..."<<endl;
		IH->EntropyAnalysis(n,dist,rescaling);
		cout<<"Entropy analysis completed."<<endl;
	}

	OverallCoreNumber=MIN(omp_get_num_procs(),Max_Cores_To_USE);
	#pragma omp master
	{	
		cout<<"Available computing cores: \x1b[92m"<<OverallCoreNumber<<"\x1b[0m"<<endl;
		StatsArray=new STATS[OverallCoreNumber]; 
		Optimum.tree_len=MAX_DOUBLE;
		Optimum.tree.clear();
		Optimum.OverallNumber_of_Primal_Solutions=0;
	}

	#pragma omp parallel for 
	for( unsigned int i=0; i<OverallCoreNumber; i++){
		StatsArray[i].outoftime=false;
		StatsArray[i].nodecounter=0;
		StatsArray[i].int_sol_counter=0;
		StatsArray[i].LPCounter=0;
		StatsArray[i].DualCounter=0;
		StatsArray[i].RootLB=-MAX_DOUBLE;
		StatsArray[i].RootUB=MAX_DOUBLE;
		StatsArray[i].start_time=0;
		StatsArray[i].end_time=0;
		StatsArray[i].core_runnnig=false;
	} 
	

	stringstream ss; ss<<Optimum.tree_len; string sss=ss.str(); int nDigits = MAX(sss.size(),25); 
 	std::cout << std::fixed << std::setprecision(PRECISION)<<endl;
 	Solver *Core0Solver;  /* temp solver  */
	int* adj_mat;
	#pragma omp master 
	{
		StatsArray[0].start_time = omp_get_wtime();
		Core0Solver=new Solver(0,n);
		Core0Solver->PrimalBoundSwALP();
		if(boundtype==1){
			cout<<"Computing lower bound via Pardi's combinatorial approach."<<endl;
			StatsArray[0].RootLB=StatsArray[0].PardiRootLB;
		} 		
		else{
			cout<<"Computing lower bound via linear programming."<<endl; 
			StatsArray[0].RootLB=StatsArray[0].LPRootLB;
		}
		StatsArray[0].GAP = abs(Optimum.tree_len-StatsArray[0].RootLB)/Optimum.tree_len*100;
		delete Core0Solver;
	}
	// TOLL=0.000000001;
	if(StatsArray[0].GAP > TOLL){
		#pragma omp master 
		{
			// cout<<"\nStarting parallel global search on ";
			auto start = std::chrono::system_clock::now();
			std::time_t start_date = std::chrono::system_clock::to_time_t(start);

		}	
		#pragma omp barrier

		#pragma omp parallel shared(Optimum, StatsArray, dist, precomputedpow, beta, verbose, OverallCoreNumber) num_threads(OverallCoreNumber)
		{
			int core = omp_get_thread_num(); 
			GlobalSearch(n,core); //Running parallel Search;
		}
	}
	//Let's wait for every core to finish

	#pragma omp barrier
	#pragma omp master
	{ 
		StatsArray[0].end_time = omp_get_wtime();
		auto start = std::chrono::system_clock::now();
		std::time_t start_date = std::chrono::system_clock::to_time_t(start);

		long long OverallNodes =0; 
		for( unsigned int i=0; i<OverallCoreNumber; i++) OverallNodes+= StatsArray[i].nodecounter; 

		Optimum.primal_bound_list.push_back(Optimum.tree);

        adj_mat = make_adj_mat(Optimum.tree);


		//Only Master frees memory	
 		delete[] beta;
		delete[] precomputedpow;
		delete[] precomputedpowVH;
		delete[] StatsArray;
		if(dist!=NULL){for( unsigned int i=1;i<=n;i++) delete [] dist[i]; delete [] dist;}		
		delete IH;		
	}	
	return adj_mat;
}

int* InputParser(double* d, int n_taxa){
	int taxa_to_consider=n_taxa;
	bool rescaling=false;
	int cores_to_use=16;
	boundtype=2;
	IH=new InputHandler();
	return  Initializer_and_Terminator(d,taxa_to_consider,rescaling,cores_to_use);
}


/*Stat rosa pristina nomine 
  Nomina nuda tenemus */

extern "C" {

    int* run(double* d, int n_taxa, bool log) // float (*f)(Vector<float>))
    {
         cout<<log<<"++++++++++++++++++++++++++++++++++++++++"<<endl;

        if(log==false){
            cout.setstate(std::ios_base::failbit);
            XPRB::setMsgLevel(0);
            }
		int* adj_mat = InputParser(d, n_taxa);
		cout.clear();
		return adj_mat;
	}

}