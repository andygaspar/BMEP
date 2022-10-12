#include "Solver.cpp"

class InputHandler{
private:
	ifstream in;
	bool distort;
	GlobalVars vars;

public:
	InputHandler(GlobalVars v){

	vars = v;

	}
	~InputHandler(){}
	// void DoubleStochasticRescaling(int n, double **, double** &);
	void ReadDistanceMatrix(char * const, string &, int, double** &, int &,bool);
	void Print(int **,int,int);
	void Print(double **,int,int);
	void PrintForMathematica(double **,int,int);
	void CallMosel(string, string, double** &, int);
	void CallConcorde(string, string, double** &, int);
	void InputSettler(int, char * const [], int &, bool &, int &, int &);
	void HowTo();
	void StabilityChecker(int, double **);
	void MachineEpsilon();
	void EntropyAnalysis(int, double ** &, bool);
	void DistanceDistorter(int, double ** &);
	double** set_d(double*, int);
};


void InputHandler::MachineEpsilon(){while ((1+vars.EPS) != 1){vars.TOLL = vars.EPS; vars.EPS /=2;} cout << "Machine epsilon: "<< vars.TOLL << endl;}
void InputHandler::StabilityChecker(int n, double **d){
	double m = INF;
	int acc=0;
	double smallest=INF;
	for(int i=1; i<=n-1; i++){
		for(int j=i+1; j<=n; j++){
			if(m > log2(d[i][j]/vars.EPS) && log2(d[i][j]/vars.EPS) >0){
				m=log2(d[i][j]/vars.EPS);
				// cout<<"d("<<i<<","<<j<<")="<<d[i][j]<<"\t"<<"EPS: "<<EPS<<"\tValue: "<<m<<endl;
			}
			if(d[i][j] < 1/pow(10.0,10.0)) acc++;
			if(smallest > d[i][j]) smallest=d[i][j];
		}
	}
	cout<<"The smallest entry in the matrix has value "<<smallest<<"."<<endl;
	cout<<"The matrix contains "<<acc<<" number of entries having value smaller than 10e-10."<<endl;
	cout<<"Numerical stability issues for the considered instance concerns path-lengths longer than or equal to "<<floor(m)<<"."<<endl;
}

void InputHandler::DistanceDistorter(int n, double ** &D){
	if(n <= 6){cout<<"less than 7 taxa, performing no distorsion..."<<endl; return;}
	cout<<"Priting D before distorsion..."<<endl;
	PrintForMathematica(D,n,n);
	cout<<endl;
	//finding the maximum distance in the matrix...
	double max_dist=-1;
	for(int i=1; i<=n-1; i++) for(int j=i+1; j<=n; j++) if(D[i][j]>max_dist) max_dist=D[i][j];

	//Taxon per taxon, we now set the last 5 largest distances to the maximum distance
	struct PAIR{int taxon; double distance;};
	for(int i=1; i<=n; i++){
		//We use vector v as an auxiliary variable to sort the entries of row i of D
		vector<PAIR> v;
		for(int j=1; j<=n; j++) if(j!=i){ PAIR p; p.taxon=j; p.distance=D[i][j]; v.push_back(p); }

		// cout<<"printing v vector..."<<endl;
		// for(int j=0; j<n-1; j++){
		// 	cout<<v[j].taxon<<"\t"<<v[j].distance<<endl;
		// }

		//cout<<"sorting v..."<<endl;
		//Sorting the entries of row i of D
		for(int r=0; r<n-2; r++){
			for(int s=r+1; s<n-1; s++){
				if(v[s].distance < v[r].distance){
					int t = v[r].taxon; double val = v[r].distance;
					v[r].taxon = v[s].taxon; v[r].distance = v[s].distance;
					v[s].taxon = t; v[s].distance = val;
				}
			}
		}
		//cout<<"done!"<<endl;
		//cout<<"applying distortion to D..."<<endl;
     	//Now we set the relative entries to max distance
     	max_dist=10000;
		for(int k=v.size()-1; k>=6; k--){
			int t = v[k].taxon;
			D[t][i]= D[i][t] = max_dist;
		}
		//cout<<"done!"<<endl;
		v.clear();
	}
	cout<<"Priting D after distorsion..."<<endl;
	PrintForMathematica(D,n,n);
	cout<<endl;
}


void InputHandler::EntropyAnalysis(int n, double ** &D, bool rescaling){
	double **d=new double*[n+1]; for(register unsigned int i=0; i<=n; i++) d[i] = new double[n+1];
	for(int i=1; i<=n; i++) for(int j=1; j<=n; j++) d[i][j]=D[i][j];

	//PUTTING d in double stochastic form
	for(register unsigned int i=1;i<=n;++i)for(register unsigned int j=1;j<=n;++j)if(i!=j)d[i][j]=pow(2.0,-d[i][j]);
	double* rows=new double[n+1];
	double* cols=new double[n+1];
	for(register unsigned int i=1;i<=n;++i){rows[i]=0.0;cols[i]=0.0;for(register unsigned int j=1;j<=n;++j)if(i!=j){rows[i]+=d[i][j];cols[i]+=d[j][i];}}
	double mean=0.0;
	for(register unsigned int i=1;i<=n;++i)mean+=rows[i];mean/=n;
	double TOLL1 = 0.0000000001;
	while(true){
		int p=1,q=1;for(register unsigned int i=1;i<=n;i++){if(abs(rows[i]-mean)>abs(rows[p]-mean))p=i;if(abs(cols[i]-mean)>abs(cols[q]-mean))q=i;}
		double ident=rows[1];for(register unsigned int i=1;i<=n;i++)if(abs(ident-rows[i])>TOLL1||abs(ident-cols[i])>TOLL1){ident=-1;break;}
		if(abs(rows[p]-ident)<TOLL1&&abs(cols[q]-ident)<TOLL1)break;
		else if(abs(cols[q]-mean)>abs(rows[p]-mean)){
			double mean_c=0.0;for(register unsigned int i=1;i<=n;++i)if(i!=q)mean_c+=cols[i];mean_c/=(n-1)*cols[q];
			for(register unsigned int i=1;i<=n;++i)if(i!=q){rows[i]-=d[i][q];d[i][q]*=mean_c;rows[i]+=d[i][q];}
			cols[q]*=mean_c;
		}else{
			double mean_r=0.0;for(register unsigned int i=1;i<=n;++i)if(i!=p)mean_r+=rows[i];mean_r/=(n-1)*rows[p];
			for(register unsigned int i=1;i<=n;++i)if(i!=p){cols[i]-=d[p][i];d[p][i]*=mean_r;cols[i]+=d[p][i];}
			rows[p]*=mean_r;
		}
	}
	for(register unsigned int i=1;i<=n-1;i++){
		for(register unsigned int j=i+1;j<=n;j++){
			d[i][j]=(-log(d[i][j]/rows[1])/log(2));
			d[j][i]=d[i][j];
		}
	}
	//now d is in double stochastic form
	double third_eq = 0;
	for(int i=1; i<=n; i++){
		for(int j=1; j<=n; j++){
			if(i!=j){
				third_eq += d[i][j] / pow(2.0, d[i][j]);
			}
		}
	}

	double ratio = third_eq / (2*n-3);
	cout<<"\nThe entropic ratio for the given distance matrix is: "<<ratio<<endl<<endl;

	if(rescaling==true){
		cout<<"Rescaling distance matrix...";
		for(int i=0; i<=n; i++) for(int j=0; j<=n; j++) D[i][j]=d[i][j];
		cout<<"done."<<endl;
	}
	else cout<<"Assuming no rescaling of the input distance matrix. "<<endl;
	for(register unsigned int i=0; i<n+1;i++) delete [] d[i]; delete [] d;
	delete[] rows;
	delete[] cols;
	if(distort==true) DistanceDistorter(n,D);
	// exit(0);
}




void InputHandler::CallMosel(string namefile, string result, double** &dist, int n){
	 cout<<"\nPreparing data for Mosel..."<<endl;
	 ofstream myfile;
	 myfile.open(result+"Resources/MoselTSPSolver/input.txt");
  	 myfile << "N: "<<n<<endl;
  	 myfile << "dist:["<<endl;
  	 for(register unsigned int i=1; i<=n; i++){
  		for(register unsigned int j=1; j<=n; j++) myfile<<dist[i][j]<<" ";
  		myfile <<endl;
  	 }
  	 myfile << "]"<<endl;
 	 myfile.close();
  	 string cmd="mosel exec "+result+"Resources/MoselTSPSolver/TSP.mos DATAFILE="+result+"Resources/MoselTSPSolver/input.txt VERBOSE=1 SECPATH="+result+"Resources/MoselTSPSolver/";
  	 system(cmd.c_str());
	 ifstream in1; in1.open("TSPout.txt",std::ios::in);
	 if(in1.is_open()){
		cout<<"Reading taxa order...";
		double **d=new double*[n+1]; for(register unsigned int i=1; i<=n; i++) d[i] = new double[n+1];
	   	for(register unsigned int i=1;i<=n;i++)for(register unsigned int j=1; j<=n; j++) d[i][j]=dist[i][j];
	   	int *olist=new int[n+1];
	    int t=1;
	    while(in1 >> olist[t]) t++;
	    // cout<<"\nMosel optimum "<<endl;
	    // for(register unsigned int i=1;i<=n;i++) cout<< olist[i]<<endl;
	    // cout<<endl;
	    if(t-1 == n){
	   		// for(register unsigned int i=1;i<=n;i++) in >> olist[i];
			for(register unsigned int i=1;i<=n;i++)for(register unsigned int j=1; j<=n; j++) dist[i][j]=d[olist[i]][olist[j]];
		}
		else cout<<"Proceeding without circular order."<<endl;
		for(register unsigned int i=1; i<n+1;i++) delete [] d[i]; delete [] d; delete[] olist;
		system("rm -f TSPout.txt");
		in1.close();
		cout<<"done!"<<endl;
	}
	else{
	 	cout<<"Proceeding without circular order."<<endl;
	}
}

void InputHandler::CallConcorde(string namefile, string result, double** &dist, int n){
	 cout<<"\nPreparing data for Concorde..."<<endl;
	 ofstream myfile;
	 myfile.open (result+"Resources/MoselTSPSolver/input_concorde.txt");
  	 myfile<<"NAME: "<<namefile<<endl;
  	 myfile<<"TYPE: TSP"<<endl;
   	 myfile<<"DIMENSION: "<<n<<endl;
   	 myfile<<"EDGE_WEIGHT_TYPE: EXPLICIT"<<endl;
   	 myfile<<"EDGE_WEIGHT_FORMAT: FULL_MATRIX"<<endl;
   	 myfile<<"EDGE_WEIGHT_SECTION"<<endl;
   	 int max_precision=0;
  	 for(register unsigned int i=1; i<=n; i++){
  		for(register unsigned int j=1; j<=n; j++){
  			stringstream s;
  			s<<dist[i][j];
  			string s1 = s.str();
  			size_t pos = s1.find_last_of(".");
  			string s2 = s1.substr(pos+1);
  			int len = s2.length();
  			if(len > max_precision) max_precision=len;
  		}
  	 }
  	 cout<<"Max precision: "<<max_precision<<endl;
  	 for(register unsigned int i=1; i<=n; i++){
  		for(register unsigned int j=1; j<=n; j++){
  			int num = round(dist[i][j]*pow(10,max_precision));
  			myfile<< num <<"\t";
  		}
  		myfile <<endl;
  	 }
	 myfile<<"EOF"<<endl;
	 myfile.close();
  	 string cmd="concorde "+result+"Resources/MoselTSPSolver/input_concorde.txt";
	 system(cmd.c_str());
	 system("rm -f Oinput_concorde.mas");
	 system("rm -f Oinput_concorde.pul");
	 system("rm -f Oinput_concorde.sav");
	 system("rm -f input_concorde.mas");
	 system("rm -f input_concorde.pul");
	 system("rm -f input_concorde.sav");
	 ifstream in1; in1.open("input_concorde.sol",std::ios::in);
	 if(in1.is_open()){
		cout<<"Reading taxa order...";
		double **d=new double*[n+1]; for(register unsigned int i=1; i<=n; i++) d[i] = new double[n+1];
	   	for(register unsigned int i=1;i<=n;i++)for(register unsigned int j=1; j<=n; j++) d[i][j]=dist[i][j];
	   	int *olist=new int[n+1];
	    int t=0;
	    while(in1 >> olist[t]){olist[t]++; t++;}
	    // cout<<"\nConcorde optimum "<<endl;
	    // for(register unsigned int i=1;i<=n;i++) cout<< olist[i]<<endl;
	    // cout<<endl;
	    if(t-1 == n){
			for(register unsigned int i=1;i<=n;i++)for(register unsigned int j=1; j<=n; j++) dist[i][j]=d[olist[i]][olist[j]];
		}
		else cout<<"Proceeding without circular order."<<endl;
		for(register unsigned int i=1; i<n+1;i++) delete [] d[i]; delete [] d; delete[] olist;
		// system("rm -f input_concorde.sol");
		in1.close();
		cout<<"done!"<<endl;
	}
	else{
	 	cout<<"Proceeding without circular order."<<endl;
	}
}


void InputHandler::ReadDistanceMatrix(char * const fullnamefile, string &namefile, int val, double** &dist, int &n, bool rescaling){
	if(n < 4){cout<<"\nWarning: the number of taxa must be a positive integer greater than or equal to 4. Please, try again."<<endl; exit(0);}
	in.open(fullnamefile,std::ios::in);
	if(in.is_open()==false){cout << "Warning: Unable to open file. Please, try again."; exit(0);}
	string fullpath_namefile=fullnamefile;
	size_t pos = fullpath_namefile.find_last_of("/\\");
	namefile = fullpath_namefile.substr(pos+1);

	cout<<"Preocessing file "<<"\x1b[92m"<<namefile<<"\x1b[0m...";

	int N; in >> N;
	if(val==MAX_INT) n=N; else{if(n > N) n=N; else n=val;}
	if(n<=N && n >=4){
		cout<<"\nTaxa in the current instance: \x1b[91m"<<n<<"\x1b[0m"<<endl;
		double **d=new double*[N]; for(register unsigned int i=0; i<N;i++) d[i] = new double[N];
		dist=new double*[n+1]; for(register unsigned int i=0; i<=n;i++) dist[i] = new double[n+1];
		for(register unsigned int i=0;i<N;i++)for(register unsigned int j=0; j<N; j++) in >> d[i][j];
		for(register unsigned int i=1;i<=n;i++)for(register unsigned int j=1; j<=n; j++) dist[i][j]=d[i-1][j-1];
		for(register unsigned int i=0; i<N;i++) delete [] d[i]; delete [] d;
	}
	else{
		 cout<<"\nWarning!\nThe specified input number of taxa must be a positive integer greater than or equal to 4.\nIf this number exceeds the number of taxa contained in the input file, it will be automatically set to the number of taxa in such a file.\nPlease, try again by keeping in mind this constaint."<<endl; exit(0);
	}
	in.close();

	//trying to reading the associate order
	stringstream sm2; sm2<<namefile; string sm22=sm2.str();
	sm22.erase(sm22.end()-4, sm22.end());
	string sm33=sm22;
	sm22.erase(sm22.begin(), sm22.begin()+3);
	stringstream sm3; sm3<<fullpath_namefile.substr(0,pos+1)<<sm33<<n<<"order.dat"; sm22=sm3.str();
	cout<<"Searching for taxa order file "<<sm22<<"...";
	in.open(sm22.c_str(),std::ios::in);
	if(in.is_open()){
		cout<<"File found! Reading taxa order...";
		double **d=new double*[n+1]; for(register unsigned int i=1; i<=n; i++) d[i] = new double[n+1];
    	for(register unsigned int i=1;i<=n;i++)for(register unsigned int j=1; j<=n; j++) d[i][j]=dist[i][j];
    	int *olist=new int[n+1];
    	for(register unsigned int i=1;i<=n;i++) in >> olist[i];
		for(register unsigned int i=1;i<=n;i++)for(register unsigned int j=1; j<=n; j++) dist[i][j]=d[olist[i]][olist[j]];
		for(register unsigned int i=1; i<n+1;i++) delete [] d[i]; delete [] d; delete[] olist;
		in.close();
		cout<<"done!"<<endl;
	}
	else{
		 cout<<"No taxa order found."<<endl;
		 if(vars.order > 0){
			 cout<<"Calling TSP Solver...";
		     std::array<char, 128> buffer;
	    	 std::string result;
	    	 std::unique_ptr<FILE, decltype(&pclose)> pipe(popen("pwd", "r"), pclose);
	    	 if (!pipe){throw std::runtime_error("Absolute path finder failed!"); exit(1);}
	    	 while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) result += buffer.data();
	    	 result.erase(remove(result.begin(), result.end(), '\n'), result.end());
	    	 result+="/";
	    	 //result contains the absolute path
	   		 switch(vars.order){
	   		 	case 1: CallMosel(namefile,result,dist,n); break;
	   		 	case 2: CallConcorde(namefile,result,dist,n); break;
	  		 }
	  	}
	}
	cout<<"The processing of "<<"\x1b[92m"<<namefile<<"\x1b[0m is completed."<<endl<<endl;
  	 // for(register unsigned int i=1; i<=n; i++){
  		// for(register unsigned int j=1; j<=n; j++) dist[i][j]=1;
  	 // }
}

void InputHandler::Print(int **C,int row,int col){cout<<endl;for(register unsigned int i=1;i<=row;i++){for(register unsigned int j=1; j<=col;j++) cout << C[i][j]<< "\t"; cout <<endl;}cout<<endl;}
void InputHandler::Print(double **C,int row,int col){cout<<endl; for(register unsigned int i=1;i<=row;i++){for(register unsigned int j=1; j<=col;j++) cout << C[i][j]<< "\t"; cout <<endl;}cout<<endl;}

void InputHandler::PrintForMathematica(double **C,int row,int col){
	cout<<"Printing distance matrix in Mathematica format..."<<endl;
	cout<<"Dist={";
	for(register unsigned int i=1;i<=row;i++){
		cout<<"{";
		for(register unsigned int j=1; j<=col;j++){
			if(j==col) cout << C[i][j]<< "}";
			else cout<<C[i][j]<<",";
		}
		if(i<row) cout<<","<<endl;
	}
	cout<<"};"<<endl;
}


void InputHandler::InputSettler(int argc, char * const argv[], int & taxa_to_consider, bool &rescaling, int &cores_to_use, int &boundtype){
	distort=false;
	if(argc>=2){
		for(register unsigned int i=2; i<argc; i++){
			string s=argv[i];

			std::size_t found=s.find("-taxa=");
			if(found != std::string::npos){
				if(s.length()<=6){
			 		cout<<"Syntax error with the parameter -taxa"<<endl<<endl;
			  		HowTo();
			  	}
				try{taxa_to_consider = stoi(s.substr(found+6));}
				catch(std::exception& e){std::cout << "Error converting taxa value. Aborting." << endl; exit(0);}
			}
		    else{
		    	found=s.find("-boundtype=");
		    	if(found!=std::string::npos){
			    	if(s.length()<=11){
					   	cout<<"Syntax error with the parameter -boundtype"<<endl<<endl;
					   	HowTo();
					}
					else if(s.substr(found+11)=="Pardi") vars.boundtype=1;
						else if(s.substr(found+11)=="VH") vars.VinhHaeseler=true;
							 else{cout<<"Bad value for parameter -boundtype. Aborting."<<endl; exit(0);}
				}
		    	else{
			    	found=s.find("-rescaling=");
			    	if(found!=std::string::npos){
			    		if(s.length()<=11){
					    	cout<<"Syntax error with the parameter -rescaling"<<endl<<endl;
					    	HowTo();
					    }
						if(s.substr(found+11)=="true") rescaling=true;
						else{cout<<"Bad value for parameter -rescaling. Aborting."<<endl; exit(0);}
					}
	        	 	else{
				    	found=s.find("-cores=");
				    	if(found!=std::string::npos){
					    	if(s.length()<=7){
					    		cout<<"Syntax error with the parameter -cores"<<endl<<endl;
					    		HowTo();
					    	}
		    				try{cores_to_use = stoi(s.substr(found+7));}
							catch(std::exception& e){std::cout << "Error converting cores value. Aborting." << endl; exit(0);}
				    		if(cores_to_use<=0) cores_to_use=MAX_INT;
						}
						else{
					    	found=s.find("-order=");
					    	if(found!=std::string::npos){
					    		if(s.length()<=7){
					    			cout<<"Syntax error with the parameter -order"<<endl<<endl;
					    			HowTo();
					    		}
			    				try{vars.order = stoi(s.substr(found+7));}
								catch(std::exception& e){std::cout << "Error converting order value. Aborting." << endl; exit(0);}
							}
		        	 		else{
		        	 			if(strcmp(argv[i],"-p")==0){
		        	 				vars.output_for_experiment=true;
		        	 			}
		        	 			else{
		        	 				found=s.find("-distort=");
							    	if(found!=std::string::npos){
							    		if(s.length()<=9){
									    	cout<<"Syntax error with the parameter -distort"<<endl<<endl;
									    	HowTo();
									    }
										if(s.substr(found+9)=="true") distort=true;
										else{cout<<"Bad value for parameter -rescaling. Aborting."<<endl; exit(0);}
									}
		        	 			}
		        	 		}
	        	 		}
	        	 	}
	        	}
	    	}
	    }
	}
}

void::InputHandler::HowTo(){
	cout<<"Synopsis: BME-Solver distance_file [optional parameters]"<<endl<<endl;
	cout<<"List of optional parameters"<<endl;
	cout<<"-taxa=[an integer K]"
		<<left<<setw(17)<<"\t"<<"Enables the processing of the first K taxa in an instance of N >= K taxa.\n"
		<<left<<setw(33)<<"\t"<<"For example, the option -taxa 10 consider the first 10 taxa in the considered instance;\n"
		<<left<<setw(33)<<"\t"<<"The option requires that K>=4; if K exceeds N then the solver sets automatically K=N.\n"
		<<left<<setw(33)<<"\t"<<"Default value: all taxa in the given instance."
		<<endl
		<<endl
	    <<"-boundtype=[Pardi, LP/Lagrangian, VH]"
	    <<left<<setw(1) <<"\t"<<"Specifies the type of lower bound to use during the search.\n"
	    <<left<<setw(33)<<"\t"<<"For example, the option -boundtype=Pardi enables the use of Pardi's bound.\n"
	    <<left<<setw(33)<<"\t"<<"The option -boundtype=LP/Lagrangian enables the use of the linear programming relaxation.\n"
	    <<left<<setw(33)<<"\t"<<"The option -boundtype=VH enable the use of the linear programming with Vinh-von Haeseler rescaling.\n"
	    <<left<<setw(33)<<"\t"<<"Default value: LP/Lagrangian."
	    <<endl
	    <<endl
	    <<"-rescaling=[true or false]"
	    <<left<<setw(9) <<"\t"<<"Enables or disables the double-stochastic rescaling of the distance matrix.\n"
	    <<left<<setw(33)<<"\t"<<"For example, the option -rescaling=true activates the rescaling; -rescaling=false deactivates it.\n"
	    <<left<<setw(33)<<"\t"<<"Default value: false."
	    <<endl
	    <<endl
	    <<"-cores=[a positive integer]"
	    <<left<<setw(9) <<"\t"<<"Specifies the number of computing cores to use.\n"
	    <<left<<setw(33)<<"\t"<<"For example, the option -cores 8 enables allows the use of 8 parallel threads.\n"
	    <<left<<setw(33)<<"\t"<<"Default value: all of the computing cores available."
	    <<endl
	    <<endl
	    <<"-order=[0, 1, 2]"
	    <<left<<setw(17) <<"\t"<<"Specifies the type of circular order to use.\n"
	    <<left<<setw(33)<<"\t"<<"0 is none; 1 - fico xpress mosel; 2 - concorde TSP.\n"
	    <<left<<setw(33)<<"\t"<<"Default value: 0."
	    <<endl
	    <<endl
	    <<"-distort=[true or false]"
	    <<left<<setw(9) <<"\t"<<"Enables or disables the distortion of the distance matrix.\n"
	    <<left<<setw(33)<<"\t"<<"For example, the option -distort=true activates the distortion; -rescaling=false deactivates it.\n"
	    <<left<<setw(33)<<"\t"<<"Default value: false."
	    // <<"-p"		  <<left<<setw(15)<<"\t"<<"print output for experiments";
	    <<endl
	    <<endl;
		exit(0);
}
/*Stat rosa pristina nomine
  Nomina nuda tenemus */