#include "global_vars.cpp"

class Solver{
private:
	/* Engine variables */
	int Core;
	int n;
	vector <struct EDGE> tree;
	int **Tau;
	int **M;
	int *mem;
	double len;
	/* **************** */
	XPRBprob problem;							/*          XPRESS Problem			 */
	XPRBvar ***path;							/*            XPRESS vars			 */
	XPRBctr *Kraft,**Unicity,Third;				/*      XPRESS Core Constraints		 */
	// int ***DistTaxon;							/* Topological distances when taxon is inserted */
	double **DualValuesAtLeaf;                  /* This Matrix store at row i the dual values (lambda and mu) relative to taxon i */
	void LoadProblem();
	void Explore(int, int, int);
	void DFS(int);
	void Add(int, int);
	void Restore(int);
	void ComputeTopologicalMatrix(int,bool);
	double LagrangianDualLowerBound(int);
	double FastLinearProgrammingLowerBound(int);
	double PardiLowerBound(int);
	bool Pruned(int, int);
	bool Push(int, int, int);
	void UpdateOptimum();
	void UpdateOptimum2(bool);
	void SetStartingTree();
	void PrintTree();
	// void PrintM();
	void NNISwap(int, int, int, int);
	void NNI(int,bool);

public:
    GlobalVars* vars;
	NODE RetrieveNode_and_Delete_from_Queue();
	void SetNode_and_Run(NODE *node);
	void PrimalBoundSwALP();
	void NJtree();
	// void SetStartingQueue(int taxon,int edge_cardinality);
	Solver(){}
	Solver(int CoreNum,int taxa, GlobalVars* v){
	    vars = v;
		Core=CoreNum;
		n=taxa;
		Tau=new int*[n+1];  for(register unsigned int i=0; i<=n;  i++)   Tau[i]=new int[n+1];
	    M=new int*[n-1];  for(register unsigned int i=0; i<=n-2;  i++) M[i]=new int[7];
	    mem=new int[n-1];
	    path=new XPRBvar**[n+1]; for(register unsigned int i=1; i<=n-1; i++) {path[i]=new  XPRBvar*[n+1]; for(register unsigned int j=i+1; j<=n; j++) path[i][j]=new  XPRBvar[n];}
	    Kraft=new XPRBctr[n+1];
	    Unicity=new XPRBctr*[n+1]; for(register unsigned int i=1; i<=n; i++) Unicity[i]=new XPRBctr[n+1];
	    DualValuesAtLeaf=new double*[n+1];
	    for(register unsigned int i=1; i<=n; i++) DualValuesAtLeaf[i]=new double[n+1];
	    // DistTaxon=new int**[n+1];
	    // for(register unsigned int i=0; i<=n; i++){DistTaxon[i]=new int*[n+1]; for(register unsigned int j=0; j<=n;  j++) DistTaxon[i][j]=new int[n+1];}
		LoadProblem();
		SetStartingTree();
		ComputeTopologicalMatrix(3,true);
	}
	~Solver(){
		for(register unsigned int i=0; i<=n-1; i++){
		for(register unsigned int j=0; j<=n; j++){
			for(register unsigned int k=2; k<=n-1; k++){
			    if(path[i][j][k].getSol() > 0.1) cout<<k<<" ";
			}
		}
		cout<<endl;
	}
	  //tree.clear();
	  for(register unsigned int i=0; i<=n; i++)    delete [] Tau[i]; delete [] Tau;
	  for(register unsigned int i=0; i<=n-2; i++)  delete [] M[i]; delete [] M;
	  delete [] mem;
	  for(register unsigned int i=1; i<=n-1; i++) {for(register unsigned int j=i+1; j<=n; j++) delete [] path[i][j];  delete [] path[i];}  delete [] path;
	  delete[] Kraft;
	  for(register unsigned int i=1; i<=n; i++) delete[] Unicity[i]; delete[] Unicity;
	  for(register unsigned int i=1; i<=n; i++) delete[] DualValuesAtLeaf[i]; delete[] DualValuesAtLeaf;
	  // for(register unsigned int i=0; i<=n; i++){for(register unsigned int j=0; j<=n; j++) delete [] DistTaxon[i][j]; delete [] DistTaxon[i];} delete [] DistTaxon;
	 }
};

void Solver::PrintTree(){
	cout<<"Core: "<<Core<<" printing tree..."<<endl;
	for(register unsigned int i=0; i<(int)tree.size(); i++) cout<<tree[i].i<<"\t"<<tree[i].j<<endl;
}


void Solver::SetStartingTree(){
	struct EDGE x;
	x.i=1; x.j=n+1; tree.push_back(x);
	x.i=2; x.j=n+1; tree.push_back(x);
	x.i=3; x.j=n+1; tree.push_back(x);
}

void Solver::Add(int taxon, int e){
	struct EDGE edge1, edge2, edge3;
	edge1.i=taxon;      edge1.j=n+taxon-2;
	edge2.i=tree[e].i;  edge2.j=n+taxon-2;
	edge3.i=tree[e].j;  edge3.j=n+taxon-2;

	tree.erase(tree.begin()+e);
	tree.push_back(edge1);
	tree.push_back(edge2);
	tree.push_back(edge3);
}

void Solver::Restore(int e){
	struct EDGE edge;
	long tsize=tree.size();
	edge.i=tree[tsize-2].i;
	edge.j=tree[tsize-1].i;

	tree.pop_back();
	tree.pop_back();
	tree.pop_back();
	tree.insert(tree.begin()+e,edge);
}


void Solver::ComputeTopologicalMatrix(int taxon, bool flag){
	//M[i][1],M[i][2],M[i][3] store the nodes adjacent to the internal node i+n
	int tsize=tree.size();
	if(flag){
		for(register unsigned int i=1; i<=n-2;i++) mem[i]=1;
		for(register unsigned int e=0; e<tsize; e++){
			int temp1=tree[e].i; int temp2=tree[e].j;
			if (temp1 > n) {M[temp1-n][mem[temp1-n]]=temp2; mem[temp1-n]++;}
			if (temp2 > n) {M[temp2-n][mem[temp2-n]]=temp1; mem[temp2-n]++;}
		}
	}
	register int CurrentNode,VisitedNode;
	len=0;
	Tau[0][0] = 0; // Modified 2/6/2009
	for(register unsigned int j=0; j<tsize; j++){
		if(tree[j].i<=taxon){
			int i = tree[j].i;
			int father=tree[j].j;

			Tau[i][i]=0; // Modified 2/6/2009 - rimodificato il 29/9/2020
			Tau[i][0]=0; // modificato il 29/9/2020

			M[father-n][4]=1; M[father-n][5]=i; M[father-n][6]=0;
			CurrentNode=father;
			while(true){
				if (M[CurrentNode-n][6]<3){
					M[CurrentNode-n][6]++; VisitedNode=M[CurrentNode-n][M[CurrentNode-n][6]];
					if (VisitedNode != M[CurrentNode-n][5]){
						if (VisitedNode > n){
							M[VisitedNode-n][4]=M[CurrentNode-n][4]+1;
							M[VisitedNode-n][5]=CurrentNode;
							M[VisitedNode-n][6]=0;
							CurrentNode=VisitedNode;
						}
						else{
							 // Modified 2/6/2009 - rimodificato il 29/9/2020
							 	Tau[i][VisitedNode]=M[CurrentNode-n][4]+1;
							 	if(Tau[i][VisitedNode]>Tau[i][0]){
							 		Tau[i][0] = Tau[i][VisitedNode];
							 		if (Tau[i][0] > Tau[0][0]) Tau[0][0] = Tau[i][0];
							 	}
							 // End Modified 2/6/2009 - 29/9/2020
							 if(vars->VinhHaeseler) len+=vars->dist[i][VisitedNode]/vars->precomputedpowVH[Tau[i][VisitedNode]];              //pow(2.0,(double)Tau[i][VisitedNode]/n);
							 else len+=vars->dist[i][VisitedNode]*vars->precomputedpow[n-Tau[i][VisitedNode]];

						}
					}
				}
				else{
					if(CurrentNode == father) break;
					else CurrentNode=M[CurrentNode-n][5];
				}
			}
      	}
    }
}

void Solver::LoadProblem(){
	problem.setMsgLevel(0);
 	XPRSsetintcontrol(problem.getXPRSprob(), XPRS_PRESOLVE,    0);
    XPRSsetintcontrol(problem.getXPRSprob(), XPRS_CUTSTRATEGY, 0);
	XPRSsetintcontrol(problem.getXPRSprob(), XPRS_HEURSTRATEGY,0);
	XPRSsetintcontrol(problem.getXPRSprob(), XPRS_THREADS,  1);
	XPRSsetintcontrol(problem.getXPRSprob(), XPRS_LPTHREADS,1);
	XPRSsetintcontrol(problem.getXPRSprob(), XPRS_BARTHREADS,1);
	//Path variables only for i<j
	for(register unsigned int i=1; i<=n-1; i++){
		for(register unsigned int j=i+1; j<=n; j++){
			for(register unsigned int k=2; k<=n-1; k++){
				path[i][j][k]=problem.newVar("path",XPRB_BV);
			}
		}
	}
	//Obj function
	XPRBlinExp Obj;
	for(register unsigned int i=1; i<=n-1; i++){
		for(register unsigned int j=i+1; j<=n; j++){
			for(register unsigned int k=2; k<=n-1; k++){
				double coeff;
				if(vars->VinhHaeseler) coeff =  2*vars->dist[i][j]/vars->precomputedpowVH[k];           //pow(2.0,(double)k/n);
				else coeff = 2*vars->dist[i][j]*vars->precomputedpow[n-k];
				Obj+=path[i][j][k]*coeff;
			}
		}
	}

	int mat_tau_opt[49] {0, 6, 2, 6, 3, 4, 5, 6, 0, 6, 2, 5, 4, 3, 2, 6, 0, 6, 3, 4, 5, 6, 2, 6, 0, 5, 4, 3, 3, 5, 3, 5, 0, 3, 4, 4, 4, 4, 4, 3, 0, 3, 5, 3, 5, 3, 4, 3, 0};
	problem.setObj(Obj);
	//Convexity constraint
	for(register unsigned int i=1; i<=n; i++){
		for(register unsigned int j=i+1; j<=n; j++){
			XPRBlinExp C2;
			for(register unsigned int k=2; k<=n-1; k++)  C2+=path[i][j][k];

			for(register unsigned int k=2; k<=n-1; k++) {
			    if(mat_tau_opt[n*i + j] == k) {problem.newCtr(path[i][j][k] == 1); /*cout<<i<<" "<<j<<" "<<k<<endl;*/}
			    else problem.newCtr(path[i][j][k] == 0);

			}
			Unicity[i][j]=problem.newCtr(C2 == 1);
		}
		for(register unsigned int j=1; j<=i-1; j++){
			XPRBlinExp C2;
			for(register unsigned int k=2; k<=n-1; k++)  C2+=path[j][i][k];
			Unicity[j][i]=problem.newCtr(C2 == 1);
		}
	}

	//Kraft equality
	for(register unsigned int i=1; i<=n; i++){
		XPRBlinExp C1;
		for(register unsigned int j=1; j<=i-1; j++) for(register unsigned int k=2; k<=n-1; k++) C1+=path[j][i][k]*vars->precomputedpow[n-k];
		for(register unsigned int j=i+1; j<=n; j++) for(register unsigned int k=2; k<=n-1; k++) C1+=path[i][j][k]*vars->precomputedpow[n-k];
		Kraft[i]=problem.newCtr(C1 == vars->precomputedpow[n-1]);
	}

	//Third constraint
	XPRBlinExp TC3;
	for(register unsigned int i=1; i<=n-1; i++){
		for(register unsigned int j=i+1; j<=n; j++){
			for(register unsigned int k=2; k<=n-1; k++){
			   double coeff = 2*k*vars->precomputedpow[n-k];
			   TC3+=path[i][j][k]*coeff;
			}
		}
	}
	Third=problem.newCtr(TC3 == (2*n-3)*vars->precomputedpow[n]);
}



double Solver::PardiLowerBound(int taxon){
	double acc=0; for(register unsigned int k=taxon+1; k<=n; k++) acc+=vars->beta[k];
	acc=vars->precomputedpow[n-1]*acc + len;
	return acc;
}

double Solver::LagrangianDualLowerBound(int taxon){
	double LB1 = DualValuesAtLeaf[taxon][0]*(2*n-3)*vars->precomputedpow[n];
	for(register unsigned int i=1; i<=n; i++) LB1 += DualValuesAtLeaf[taxon][i]*vars->precomputedpow[n-1];

	double temp,temp1;
	int temp2;
	for(register unsigned int i=1; i<=taxon-1; i++)
		for(register unsigned int j=i+1; j<=taxon; j++) {
			temp = 2*vars->dist[i][j]-DualValuesAtLeaf[taxon][i]-DualValuesAtLeaf[taxon][j];
			temp1 = (temp - 2*DualValuesAtLeaf[taxon][0]*Tau[i][j])*vars->precomputedpow[n-Tau[i][j]];
			temp2 = MIN(n-taxon+Tau[i][j],n-1);
			for(register unsigned int k = Tau[i][j]+1; k <= temp2; k++)
				if ((temp - 2*DualValuesAtLeaf[taxon][0]*k)*vars->precomputedpow[n-k] < temp1) temp1 = (temp - 2*DualValuesAtLeaf[taxon][0]*k)*vars->precomputedpow[n-k];
			LB1 += temp1;
		}

	for(register unsigned int i=1; i<=taxon-1; i++){
		for(register unsigned int j=taxon+1; j<=n; j++){
			temp = 2*vars->dist[i][j]-DualValuesAtLeaf[taxon][i]-DualValuesAtLeaf[taxon][j];
			temp1 = (temp - 2*DualValuesAtLeaf[taxon][0]*2)*vars->precomputedpow[n-2];
			temp2 = MIN(n-taxon+Tau[i][0],n-1);	// MODIFIED 2/6/2009
			for(register unsigned int k = 3; k <= temp2; k++)  // MODIFIED 2/6/2009
				if ((temp - 2*DualValuesAtLeaf[taxon][0]*k)*vars->precomputedpow[n-k] < temp1) temp1 = (temp - 2*DualValuesAtLeaf[taxon][0]*k)*vars->precomputedpow[n-k];
			LB1 += temp1;
		}
	}
	for(register unsigned int i=taxon; i<=n-1; i++){
		for(register unsigned int j=i+1; j<=n; j++){
			temp = 2*vars->dist[i][j]-DualValuesAtLeaf[taxon][i]-DualValuesAtLeaf[taxon][j];
			temp1 = (temp - 2*DualValuesAtLeaf[taxon][0]*2)*vars->precomputedpow[n-2];
			temp2 = MIN(n-taxon+Tau[0][0],n-1); // MODIFIED 2/6/2009
			for(register unsigned int k = 3; k <= n-1; k++) // MODIFIED 2/6/2009
				if ((temp - 2*DualValuesAtLeaf[taxon][0]*k)*vars->precomputedpow[n-k] < temp1) temp1 = (temp - 2*DualValuesAtLeaf[taxon][0]*k)*vars->precomputedpow[n-k];
			LB1 += temp1;
		}
	}
	return LB1;
}


double Solver::FastLinearProgrammingLowerBound(int taxon){
	// Eq 13
	for(register unsigned int i=1; i<=taxon-1; i++){
		for(register unsigned int j=i+1; j<=taxon; j++){
			for(register unsigned int k=2; k<=Tau[i][j]-1; k++) path[i][j][k].setUB(0.0);
			for(register unsigned int k=(n-taxon)+Tau[i][j]+1; k<=n-1; k++) path[i][j][k].setUB(0.0);
		}
	}

	// Eq 14
	for(register unsigned int i=1; i<=taxon; i++){
		for(register unsigned int j=taxon+1; j<=n; j++){
			for(register unsigned int k=(n-taxon)+Tau[taxon][0]+1; k<=n-1; k++) path[i][j][k].setUB(0.0);
		}
	}

	// Eq 15
	for(register unsigned int i=taxon+1; i<=n-1; i++){
		for(register unsigned int j=i+1; j<=n;   j++){
			for(register unsigned int k=(n-taxon)+Tau[0][0]+1; k<=n-1; k++) path[i][j][k].setUB(0.0);
		}
	}

	problem.minim("ld");
	double LB=problem.getObjVal();
	DualValuesAtLeaf[taxon][0]=Third.getDual();
	for(register unsigned int i=1; i <= n; i++) DualValuesAtLeaf[taxon][i]=Kraft[i].getDual();

	// Eq 13
	for(register unsigned int i=1; i<=taxon-1; i++){
		for(register unsigned int j=i+1; j<=taxon; j++){
			for(register unsigned int k=2; k<=Tau[i][j]-1; k++) path[i][j][k].setUB(1.0);
			for(register unsigned int k=(n-taxon)+Tau[i][j]+1; k<=n-1; k++) path[i][j][k].setUB(1.0);
		}
	}

	// Eq 14
	for(register unsigned int i=1; i<=taxon; i++){
		for(register unsigned int j=taxon+1; j<=n; j++){
			for(register unsigned int k=(n-taxon)+Tau[taxon][0]+1; k<=n-1; k++) path[i][j][k].setUB(1.0);
		}
	}

	// Eq 15
	for(register unsigned int i=taxon+1; i<=n-1; i++){
		for(register unsigned int j=i+1; j<=n;   j++){
			for(register unsigned int k=(n-taxon)+Tau[0][0]+1; k<=n-1; k++) path[i][j][k].setUB(1.0);
		}
	}

	return LB;
}

bool Solver::Pruned(int taxon, int selector){
	double LowerBound=0;
	switch(selector){
		case 1: LowerBound=PardiLowerBound(taxon); if(LowerBound > vars->Optimum.tree_len) return true; break;
		case 2: if(!vars->VinhHaeseler){
					vars->StatsArray[Core].DualCounter++;
					LowerBound=LagrangianDualLowerBound(taxon);
					if(LowerBound > vars->Optimum.tree_len) return true;
				}
				vars->StatsArray[Core].LPCounter++;
				LowerBound=FastLinearProgrammingLowerBound(taxon);
				if(LowerBound > vars->Optimum.tree_len) return true;
				else return false;
				break;
    }
	return false;
}

void Solver::Explore(int start, int end, int taxon){
	for(register unsigned int e=start; e<end; e++){
	 	Add(taxon,e);
	 	ComputeTopologicalMatrix(taxon,true);
		vars->StatsArray[Core].nodecounter++;
		DFS(taxon);
		Restore(e);
	}
}

void Solver::DFS(int taxon){
	if(vars->Out_of_Time()){vars->StatsArray[Core].outoftime=true; return;}
	ComputeTopologicalMatrix(taxon,true);
	if(taxon==n){
		vars->StatsArray[Core].int_sol_counter++;
		#pragma omp critical
		{
			if(len < vars->Optimum.tree_len){
				UpdateOptimum();
			}
		}
		NNI(n,true);
	    return;
	}
    else{
	   	if(Pruned(taxon,vars->boundtype)==true) return;
		  	else{
			  	if(vars->ThereAreIdleCores(vars->OverallCoreNumber) || vars->Queue.size()<vars->OverallCoreNumber){
			  		//we try to insert the node in the queue
			  		int size=(int)tree.size();
			  		int middle = floor(size/2);
			  		bool inserted_properly=Push(middle+1,size,taxon);
			  		if(inserted_properly==true) Explore(0,middle+1,taxon+1);
			  		else Explore(0,(int)tree.size(),taxon+1);
			  	}
			  	else{
					Explore(0,(int)tree.size(),taxon+1);
				}
		 	}
	}
}

void Solver::UpdateOptimum(){
	lock_guard<mutex> lg(m); //blocking access to the optimum data...
	vars->Optimum.tree_len=len;
	vars->Optimum.primal_bound_list.push_back(vars->Optimum.tree);
	vars->Optimum.OverallNumber_of_Primal_Solutions++;
	vars->Optimum.tree.clear();
	struct EDGE edge;
	for(register unsigned int e=0; e<(int)tree.size(); e++){
		edge.i=tree[e].i;
		edge.j=tree[e].j;
		vars->Optimum.tree.push_back(edge);
	}
	cout<<"* "
			<<setw(vars->numWidth2)<< setfill(vars->separator)<<vars->Optimum.OverallNumber_of_Primal_Solutions
			<<setw(vars->numWidth+PRECISION) << setfill(vars->separator)<<std::setprecision(PRECISION)<<vars->Optimum.tree_len<<std::setprecision(PRECISION)
			<<setw(vars->numWidth+PRECISION) << setfill(vars->separator)<<std::scientific<<abs(vars->Optimum.tree_len-vars->StatsArray[0].RootLB)/vars->Optimum.tree_len*100
			<<setprecision(8)
			<<setw(vars->numWidth) << setfill(vars->separator)<<std::fixed<<omp_get_wtime() - vars->StatsArray[0].start_time
			<<setw(vars->numWidth) << setfill(vars->separator)<<vars->Queue.size()
		    <<setw(vars->numWidth) << setfill(vars->separator)<<"no"
			<<endl;
}

void Solver::UpdateOptimum2(bool print_line){
	lock_guard<mutex> lg(m); //blocking access to the optimum data...
	vars->Optimum.tree_len=len;
	vars->Optimum.primal_bound_list.push_back(vars->Optimum.tree);
	vars->Optimum.OverallNumber_of_Primal_Solutions++;
	vars->Optimum.tree.clear();
	EDGE edge;
	register unsigned int tsize=tree.size();
	for(register unsigned int e=0; e<tsize; e++){
		edge.i=tree[e].i;
		edge.j=tree[e].j;
		vars->Optimum.tree.push_back(edge);
	}
	if(print_line){
		cout<<"* "
			<<setw(vars->numWidth2)<< setfill(vars->separator)<<vars->Optimum.OverallNumber_of_Primal_Solutions
			<<setw(vars->numWidth+PRECISION) << setfill(vars->separator)<<std::setprecision(PRECISION)<<vars->Optimum.tree_len<<std::setprecision(PRECISION)
			<<setw(vars->numWidth+PRECISION) << setfill(vars->separator)<<std::scientific<<abs(vars->Optimum.tree_len-vars->StatsArray[0].RootLB)/vars->Optimum.tree_len*100
			<<setprecision(8)
			<<setw(vars->numWidth) << setfill(vars->separator)<<std::fixed<<omp_get_wtime() - vars->StatsArray[0].start_time
			<<setw(vars->numWidth) << setfill(vars->separator)<<vars->Queue.size()
			<<setw(vars->numWidth) << setfill(vars->separator)<<"yes"
			<<endl;
	}
}


/*NNI BLOCK*/
void Solver::NNISwap(int node1, int node2, int node3, int node4){
	// cout<<"swapping nodes "<<node1<<" and "<<node2<<" with "<<node3<<" and "<<node4<<endl;
	register unsigned int tsize=tree.size();
	for(register unsigned int e=0; e<tsize; e++){
		if(tree[e].i==node1 && tree[e].j==node2) {tree[e].j=node4; if(tree[e].j < tree[e].i){int temp = tree[e].j; tree[e].j=tree[e].i; tree[e].i=temp;} break;}
		else if(tree[e].i==node2 && tree[e].j==node1) {tree[e].i=node4; if(tree[e].j < tree[e].i){int temp = tree[e].j; tree[e].j=tree[e].i; tree[e].i=temp;} break;}
	}
	for(register unsigned int e=0; e<tsize; e++){
		if(tree[e].i==node3 && tree[e].j==node4) {tree[e].j=node2; if(tree[e].j < tree[e].i){int temp = tree[e].j; tree[e].j=tree[e].i; tree[e].i=temp;} break;}
		else if(tree[e].i==node4 && tree[e].j==node3) {tree[e].i=node2; if(tree[e].j < tree[e].i){int temp = tree[e].j; tree[e].j=tree[e].i; tree[e].i=temp;} break;}
	}
}


void Solver::NNI(int taxon,bool flag){
	// cout<<"In NNI..."<<endl;
	register unsigned int tsize=tree.size();
	vector<EDGE> temp_tree;
	EDGE f; for(register unsigned int e=0; e<tsize; e++){f.i=tree[e].i; f.j=tree[e].j; temp_tree.push_back(f);}
	for(register unsigned int e=0; e<tsize; e++){
		//if e is internal
		if(tree[e].i >n && tree[e].j>n){
			register unsigned int node[5];
			register unsigned int c=1;
			for(register unsigned int k=1; k<=3; k++) if(M[tree[e].i-n][k]!=tree[e].j) {node[c]=M[tree[e].i-n][k]; c++;}
			for(register unsigned int k=1; k<=3; k++) if(M[tree[e].j-n][k]!=tree[e].i) {node[c]=M[tree[e].j-n][k]; c++;}
			NNISwap(tree[e].i,node[2],tree[e].j,node[3]);
			ComputeTopologicalMatrix(taxon,true);
			if(vars->Optimum.tree_len > len){UpdateOptimum2(flag);}
			tree.clear(); for(register unsigned int s=0; s<tsize; s++){f.i=temp_tree[s].i; f.j=temp_tree[s].j; tree.push_back(f);}
			NNISwap(tree[e].i,node[2],tree[e].j,node[4]);
			ComputeTopologicalMatrix(taxon,true);
			if(vars->Optimum.tree_len > len){UpdateOptimum2(flag);}
			tree.clear(); for(register unsigned int s=0; s<tsize; s++){f.i=temp_tree[s].i; f.j=temp_tree[s].j; tree.push_back(f);}
			ComputeTopologicalMatrix(taxon,true);
		}
	}
}

/*END NNI BLOCK*/

void Solver::NJtree(){
	cout<<"Running the NJ algorithm...";
	double **NJdist;
    NJdist=new double*[2*n-1];
    for(int i=0; i<=2*n-2; i++) NJdist[i]=new double[2*n-1];
	NJdist[0][n]=n;NJdist[0][n+1]=0;
	for(register unsigned int i=1;i<=n-1;i++){
		NJdist[0][i]=i;
		for(register unsigned int j=i+1;j<=n;j++){
			NJdist[i][j]=vars->dist[i][j];
			NJdist[j][i]=vars->dist[j][i];
		}
	}
	tree.clear();

	struct EDGE e; for(register unsigned int k=1;k<=n;k++){e.i=k;e.j=n+1;tree.push_back(e);}

	// cout<<"printing star tree..."<<endl;
	// for(int e=0; e<tree.size(); e++) cout<<e<<": "<<tree[e].i<<" "<<tree[e].j<<endl;

	struct MINPOS{double value; int a; int i; int b; int j;	int idx;};
	MINPOS min;
	for(register unsigned int k=4; k<=n; k++){
		min.value = INF;
		for(register unsigned int i=1; i<=n-k+3; i++){
			for(register unsigned int j=i+1; j<=n-k+4; j++){
 				if (NJdist[(int)NJdist[0][i]][(int)NJdist[0][j]] < min.value){
					min.value=NJdist[(int)NJdist[0][i]][(int)NJdist[0][j]];
					min.a=NJdist[0][i]; min.i=i;
					min.b=NJdist[0][j]; min.j=j;
				}
			}
		}
		for(register unsigned int e=0; e<(int)tree.size(); e++)
			if ((tree[e].i == min.a && tree[e].j == n+1) || (tree[e].i == n+1 && tree[e].j == min.a)) {min.idx = e; break;}
		for(register unsigned int e=0; e<(int)tree.size(); e++)
			if ((tree[e].i == min.b && tree[e].j == n+1) || (tree[e].i == n+1 && tree[e].j == min.b)) {tree.erase(tree.begin()+e); break;}
		// Add(min.b,n+k-2,min.idx);

		struct EDGE edge1, edge2, edge3;
		edge1.i=min.b;      edge1.j=n+k-2;
		edge2.i=tree[min.idx].i;  edge2.j=n+k-2;
		edge3.i=tree[min.idx].j;  edge3.j=n+k-2;

		tree.erase(tree.begin()+min.idx);
		tree.push_back(edge1);
		tree.push_back(edge2);
		tree.push_back(edge3);

		// cout<<"printing current tree..."<<endl;
		// for(int e=0; e<tree.size(); e++) cout<<e<<": "<<tree[e].i<<" "<<tree[e].j<<endl;

		for(register unsigned int i=min.i; i<=n-k+4; i++) NJdist[0][i]=NJdist[0][i+1];
		for(register unsigned int i=min.j-1; i<=n-k+4; i++) NJdist[0][i]=NJdist[0][i+1];
		NJdist[0][n-k+3]=n+k-2;
		for(register unsigned int i=1; i<=n-k+2; i++){
			NJdist[(int)NJdist[0][i]][n+k-2]=0.5*(NJdist[min.a][(int)NJdist[0][i]]+NJdist[min.b][(int)NJdist[0][i]]-min.value);
			NJdist[n+k-2][(int)NJdist[0][i]]=NJdist[(int)NJdist[0][i]][n+k-2];
		}
	}
	ComputeTopologicalMatrix(n,true);
	cout<<"done!"<<endl;
	cout<<"BME length of the NJ-tree: "<<"\x1b[92m"<<len<<"\x1b[0m"<<endl;
	if(vars->Optimum.tree_len > len){
		UpdateOptimum2(false);
		cout<<"+The NJ-tree has a length shorter than the SWA's one. Updating the primal bound..."<<endl;
	}
    for(int i=0; i<=2*n-2; i++) delete [] NJdist[i]; delete [] NJdist;
}



void Solver::PrimalBoundSwALP(){
	cout<<"Computing root relaxation..."<<endl;

	len=INF; tree.clear();
	SetStartingTree();
	ComputeTopologicalMatrix(3,true);
	vars->StatsArray[0].PardiRootLB=PardiLowerBound(3);       //Setting Pardi's root LB
	problem.setMsgLevel(0);
	XPRSsetintcontrol(problem.getXPRSprob(), XPRS_THREADS,  vars->OverallCoreNumber);
	XPRSsetintcontrol(problem.getXPRSprob(), XPRS_LPTHREADS, vars->OverallCoreNumber);
	XPRSsetintcontrol(problem.getXPRSprob(), XPRS_BARTHREADS, vars->OverallCoreNumber);
	problem.minim("ld");
	XPRSsetintcontrol(problem.getXPRSprob(), XPRS_THREADS,   1);
	XPRSsetintcontrol(problem.getXPRSprob(), XPRS_LPTHREADS, 1);
	XPRSsetintcontrol(problem.getXPRSprob(), XPRS_BARTHREADS,1);
	problem.setMsgLevel(0);
	cout<<"done."<<endl;
	vars->StatsArray[0].LPRootLB=problem.getObjVal(); //Setting LP root LB
	cout<<endl<<endl;
	cout<<"Computing some primal bounds for the instance..."<<endl;
	cout<<"Running the SWA algorithm...";
	len=INF; tree.clear();
	SetStartingTree();
	for(register unsigned int k=4; k<=n; k++){
		double min=INF;
		register unsigned int pos=0;
		register unsigned int tsize=tree.size();
		for(register unsigned int e=0; e<tsize; e++){
			Add(k,e);
		    ComputeTopologicalMatrix(k,true);
		    double LB=PardiLowerBound(k);
			if(min > LB){pos=e; min=LB;}
			Restore(e);
	 	}
        Add(k,pos);
        cout<<setprecision(PRECISION);
        cout<<"Adding taxon "<<k<<" in edge "<<pos<<". Current length: "<<len<<endl;
	}
	ComputeTopologicalMatrix(n,true);
	cout<<"done!\nValue of SWA tree: "<<"\x1b[92m"<<len<<"\x1b[0m"<<endl;
	vars->StatsArray[0].RootUB=len;
	vars->Optimum.tree_len=len;
	vars->Optimum.OverallNumber_of_Primal_Solutions++;
	vars->Optimum.tree.clear();
	EDGE edge;
	register unsigned int tsize=tree.size();
	for(register unsigned int e=0; e<tsize; e++){
		edge.i=tree[e].i;
		edge.j=tree[e].j;
		vars->Optimum.tree.push_back(edge);
	}
	vars->Optimum.primal_bound_list.push_back(vars->Optimum.tree);
	double current_val=len;

	NJtree();

	current_val = vars->Optimum.tree_len;
	while(true){
		NNI(n,false);
		if(current_val != vars->Optimum.tree_len){
			cout<<"+New NNI local optimum: "<<"\x1b[92m"<<vars->Optimum.tree_len<<"\x1b[0m"<<endl;
			// Optimum.primal_bound_list.push_back(Optimum.tree);
			current_val=vars->Optimum.tree_len;
		}
		else break;
	}
	cout<<"Overall number of primal bound updates: "<<vars->Optimum.OverallNumber_of_Primal_Solutions<<endl;
	tree.clear();
	SetStartingTree();
	Push(0,3,3);
}



bool Solver::Push(int start, int end, int taxon){
    NODE node;
	//Recopying partial tree
    node.partial_tree.clear();
	EDGE edge;
	for(register unsigned int e=0; e<(int)tree.size(); e++){
		edge.i=tree[e].i;
		edge.j=tree[e].j;
		node.partial_tree.push_back(edge);
	}
	//Storing taxon, start edge and end edge
	node.start_edge=start;
    node.end_edge=end;
    node.taxon=taxon;
    node.empty=false;
    bool inserted_properly=false;
    #pragma omp critical
    {
		if(vars->Queue.size()<vars->OverallCoreNumber){
			vars->Queue.push(node);
			inserted_properly=true;
		}
	}
	return inserted_properly;
}


NODE Solver::RetrieveNode_and_Delete_from_Queue() {
	NODE node;
	node.empty=true;
	node.partial_tree.clear();
    #pragma omp critical
    {
      if(!vars->Queue.empty()){
   		node = vars->Queue.front();
   		vars->Queue.pop();
 	   }
 	}
 	return node;
}


void Solver::SetNode_and_Run(NODE *node){
	int taxon = node->taxon;
	tree.clear();
	//Reset of the decoding matrices.
	for(register unsigned int i=0; i<=n; i++)    for(register unsigned int j=0; j<=n; j++) Tau[i][j]=0;
	for(register unsigned int i=0; i<=n-2;  i++) for(register unsigned int j=0; j<7;  j++) M[i][j]=0;
	for(register unsigned int i=0; i<n-1; i++) mem[i]=0;

	struct EDGE edge;
	for(register unsigned int e=0; e<(int)node->partial_tree.size(); e++){
		edge.i=node->partial_tree[e].i;
		edge.j=node->partial_tree[e].j;
		tree.push_back(edge);
	}

	//clearing the tree of node
	node->partial_tree.clear();
	//Decoding the tree
	ComputeTopologicalMatrix(taxon,true);
	//Resetting all previous bounds...
	for(register unsigned int i=1; i<=n-1; i++){
		for(register unsigned int j=i+1; j<=n; j++){
			for(register unsigned int k=2; k<=n-1; k++) path[i][j][k].setUB(1.0);
		}
	}
	//setting bounds on path variables...Eq. 13
	for(register unsigned int i=1; i<=taxon-1; i++){
		for(register unsigned int j=i+1; j<=taxon; j++){
			for(register unsigned int k=2; k<=Tau[i][j]-1; k++)	path[i][j][k].setUB(0.0);
			for(register unsigned int k=(n-taxon)+Tau[i][j]+1; k<=n-1; k++) path[i][j][k].setUB(0.0);
		}
	}

	//Eq. 14
	//Tau[taxon][0] contains the max tau_{taxon,j} over all j in the partial phylogeny, j!=taxon
	for(register unsigned int i=1; i<=taxon; i++){
		for(register unsigned int j=taxon+1; j<=n; j++){
			for(register unsigned int k=(n-taxon)+Tau[i][0]+1; k<=n-1; k++) path[i][j][k].setUB(0.0);
		}
	}

	//Eq. 15
	//Tau[0][0] contains the max tau_ij over all i,j in the partial phylogeny
	for(register unsigned int i=taxon+1; i<=n-1; i++){
		for(register unsigned int j=i+1; j<=n;   j++){
			for(register unsigned int k=(n-taxon)+Tau[0][0]+1; k<=n-1; k++)	path[i][j][k].setUB(0.0);
		}
	}
	//Compute the LP relaxation for this tree and if it is smaller than the current best so far...
	if(FastLinearProgrammingLowerBound(taxon) < vars->Optimum.tree_len)
    //Explore
	Explore(node->start_edge,node->end_edge, node->taxon+1);
}
/*Stat rosa pristina nomine
  Nomina nuda tenemus */