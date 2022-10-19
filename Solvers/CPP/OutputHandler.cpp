class OutputHandler{
private:
	ofstream out;
	int **M;
	int **Tau;  
	int *mem;
	int n;
public:
	OutputHandler(int taxa){
		n=taxa;
		M=new int*[n-1];  for(register unsigned int i=0; i<=n-2;  i++) M[i]=new int[7]; 
		Tau=new int*[n+1];  for(register unsigned int i=0; i<=n;  i++)   Tau[i]=new int[n+1]; 
	    mem=new int[n-1];
	}
	~OutputHandler(){
	  for(register unsigned int i=0; i<=n-2; i++)  delete [] M[i]; delete [] M; 
	  for(register unsigned int i=0; i<=n-2; i++)  delete [] Tau[i]; delete [] Tau; 
	  delete [] mem; 
	}
	int** get_tau(){return Tau;}



	void ComputeAdjacents(vector <struct EDGE>tree){
		int tsize=tree.size();
		for(register unsigned int i=1; i<=n-2;i++) mem[i]=1;
		for(register unsigned int e=0; e<tsize; e++){
			int temp1=tree[e].i; int temp2=tree[e].j;	
			if (temp1 > n) {M[temp1-n][mem[temp1-n]]=temp2; mem[temp1-n]++;}
			if (temp2 > n) {M[temp2-n][mem[temp2-n]]=temp1; mem[temp2-n]++;}
		}
	}	

	void RecursiveTraversal(int node, int father,int depth){
		for(int i=0; i<=depth; i++) out<<"  ";
		out<<"\"name\" : "<<node<<","<<endl;
		for(int i=0; i<=depth; i++) out<<"  ";
		out<<"\"parent\" : "; father == -1 ? out<<"\"null\"" : out<<father;
		if(node <= n){out<<endl; return;}
		else{
			out<<","<<endl;
			for(int i=0; i<=depth; i++) out<<"  ";
			out<<"\"children\" : ["<<endl;
			for(int i=1; i<=3; i++){
				if(M[node-n][i]!=father){	
					for(int i=0; i<=depth; i++) out<<"  ";
					out<<"  {"<<endl;
					RecursiveTraversal(M[node-n][i], node,depth+1);
					for(int i=0; i<=depth; i++) out<<"  ";
					out<<"  }"; 
					if(i<=2) out<<","<<endl; 
					else out<<endl;
				}	
			}	
			for(int i=0; i<=depth; i++) out<<"  ";
			out<<"]"<<endl;
		}
	}


	double ComputeTopologicalMatrix(vector <struct EDGE>tree, int taxon){

		int tsize=tree.size();
		for(int i=0; i<=n; i++) for(int j=0; j<=n; j++) Tau[i][j]=0;
		cout<<"sssss"<<tree.size()<<endl;
		for(register unsigned int i=1; i<=n-2;i++) mem[i]=1;
		for(register unsigned int e=0; e<tsize; e++){
			int temp1=tree[e].i; int temp2=tree[e].j;	
			if (temp1 > n) {M[temp1-n][mem[temp1-n]]=temp2; mem[temp1-n]++;}
			if (temp2 > n) {M[temp2-n][mem[temp2-n]]=temp1; mem[temp2-n]++;}
		}
		cout<<"yyyyy"<<tree.size()<<endl;
		register int CurrentNode,VisitedNode; 
		double len=0;
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
								 if(VinhHaeseler) len+=dist[i][VisitedNode]/precomputedpowVH[Tau[i][VisitedNode]];              //pow(2.0,(double)Tau[i][VisitedNode]/n);
								 else len+=dist[i][VisitedNode]*precomputedpow[n-Tau[i][VisitedNode]]; 
								  	
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
	    return len;	     	
	}

    void ComputeD3Hierarchy(vector <struct EDGE>tree){
            cout<<"\nProcessing optimal solution for visual representation...";
            ComputeAdjacents(tree);
             for(int i=1; i<=n-2; i++){
                cout<<i+n<<":\t";
                for(int j=1; j<=3; j++) cout<<M[i][j]<<"\t"; cout<<endl;
             }
            RecursiveTraversal(2*n-2,-1,0);
        }


	void OutputPrimalBoundList(vector <vector <struct EDGE> > tree_list, string namefile){
		stringstream outputnamefile; 
		outputnamefile << namefile << "_primal_bound_list_" << n <<".txt";
		out.open((outputnamefile.str()).c_str());

		for(int k=1; k<tree_list.size(); k++){
			vector <struct EDGE> T = tree_list[k]; 
			cout<<"Printing primal solution "<<k<<"..."<<endl;
			for(register unsigned int i=0; i<(int)T.size(); i++){
				out<<T[i].i<<"\t"<<T[i].j<<endl;
			}
			cout<<"Printing the relative topological matrix..."<<endl;
			double val=ComputeTopologicalMatrix(T,n);
			cout<<"rrrr"<<endl;
			for(int i=1; i<=n; i++){
				for(int j=1; j<=n; j++) out << Tau[i][j] <<" ";
				out<<endl;
			}

			out<<"length: "<<std::setprecision(PRECISION)<<val<<endl;
			out<<endl;			
		}
		out.close();
	}	

	void OutputSpectralAnalysis(vector <vector <struct EDGE> > tree_list, double **dist){
		// cout<<setprecision(10)<<endl<<endl;		
		// cout<<"Printing data for spectral analysis..."<<endl;
		// cout<<endl;
		// cout<<"n = "<<n<<";"<<endl;
		// cout<<"dist"<<n<<"={"<<endl;
		for(int i=1; i<=n; i++){
			// cout<<"{";
			for(int j=1; j<=n-1; j++) cout<<dist[i][j]<<",";
			cout<<dist[i][n]<<"}";
			if(i<n) cout<<","<<endl; 	
		}
		cout<<"};"<<endl<<endl;


		for(int k=1; k<tree_list.size(); k++){
			vector <struct EDGE> T = tree_list[k]; 
			double val=ComputeTopologicalMatrix(T,n);
			cout<<"T"<<n<<"n"<<k<<"={"<<endl;
			for(int i=1; i<=n; i++){
				cout<<"{";
				for(int j=1; j<=n-1; j++) cout<<Tau[i][j]<<",";
				cout<<Tau[i][n]<<"}";
				if(i<n) cout<<","<<endl; 	
			}
			cout<<"};"<<endl<<endl;		
		}		
	}	


};


/*Stat rosa pristina nomine 
  Nomina nuda tenemus */