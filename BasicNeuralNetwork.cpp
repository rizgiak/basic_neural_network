//-------------------------------------------------------//
// Author : Aulia Khilmi Rizgi							 //
// Source : BasicNeuralNetwork.cpp						 //
//-------------------------------------------------------//

// Build Instruction : g++ -o BasicNeuralNetwork BasicNeuralNetwork.cpp -lboost_iostreams -lboost_system -lboost_filesystem

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <sstream>

#include "gnuplot-iostream.h"
#include <map>

#define ITERATION 1000
#define MAX_WEIGHT 22
#define MAX_BIAS 3
#define MAX_HIDDEN 7
#define LEARNING_RATE 0.5

using namespace std;

class BasicNeuralNetwork{
public:
	BasicNeuralNetwork(const string filename);
	bool isEof(void)
	{
		return m_trainingDataFile.eof();
	}
	void FeedForward(vector<double> i, vector<double> b, vector<double> w, vector<double> &h, double &o);
	void BackProp(vector<double> &d, vector<double> w, vector<double> h, vector<double> i, double o, double to);
	unsigned GenerateValue(vector<double> &weight, int var);
	void UpdateWeight(vector<double> &w, vector<double> d);
	unsigned NextInput(vector<double> &inputVals);
	unsigned TargetOutput(vector<double> &targetOutputVals);
private:
	ifstream m_trainingDataFile;
};

BasicNeuralNetwork::BasicNeuralNetwork(const string filename){
	m_trainingDataFile.open(filename.c_str());
}

void BasicNeuralNetwork::UpdateWeight(vector<double> &w, vector<double> d){
	for(unsigned i=0; i<MAX_WEIGHT; i++)
		w[i]=w[i]-(LEARNING_RATE*d[i]);
}

void BasicNeuralNetwork::BackProp(vector<double> &d, vector<double> w, vector<double> h, vector<double> i, double o, double to){
	vector<double> dh;
	GenerateValue(dh,MAX_HIDDEN);

	//Stage 1
	d[18]=-(to-o)*(o*(1-o)*h[3]);
	d[19]=-(to-o)*(o*(1-o)*h[4]);
	d[20]=-(to-o)*(o*(1-o)*h[5]);
	d[21]=-(to-o)*(o*(1-o)*h[6]);

	//Stage 2
	dh[3]=((-(to-o))*(o*(1-o))*w[18]);
	dh[4]=((-(to-o))*(o*(1-o))*w[19]);
	dh[5]=((-(to-o))*(o*(1-o))*w[20]);
	dh[6]=((-(to-o))*(o*(1-o))*w[21]);

	d[6] =(dh[3]*(h[3]*(1-h[3])*h[0]));
	d[7] =(dh[4]*(h[4]*(1-h[4])*h[0]));
	d[8] =(dh[5]*(h[5]*(1-h[5])*h[0]));
	d[9] =(dh[6]*(h[6]*(1-h[6])*h[0]));

	d[10]=(dh[3]*(h[3]*(1-h[3])*h[1]));
	d[11]=(dh[4]*(h[4]*(1-h[4])*h[1]));
	d[12]=(dh[5]*(h[5]*(1-h[5])*h[1]));
	d[13]=(dh[6]*(h[6]*(1-h[6])*h[1]));

	d[14]=(dh[3]*(h[3]*(1-h[3])*h[2]));
	d[15]=(dh[4]*(h[4]*(1-h[4])*h[2]));
	d[16]=(dh[5]*(h[5]*(1-h[5])*h[2]));
	d[17]=(dh[6]*(h[6]*(1-h[6])*h[2]));

	//Stage 3
	dh[0]=(dh[3]*(h[3]*(1-h[3]))*w[6]);
	dh[1]=(dh[3]*(h[3]*(1-h[3]))*w[10]);
	dh[2]=(dh[3]*(h[3]*(1-h[3]))*w[14]);

	d[0]=(dh[0]*(h[0]*(1-h[0]))*i[0]);
	d[1]=(dh[1]*(h[1]*(1-h[1]))*i[0]);
	d[2]=(dh[2]*(h[2]*(1-h[2]))*i[0]);

	d[3]=(dh[0]*(h[0]*(1-h[0]))*i[1]);
	d[4]=(dh[1]*(h[1]*(1-h[1]))*i[1]);
	d[5]=(dh[2]*(h[2]*(1-h[2]))*i[1]);
}

void BasicNeuralNetwork::FeedForward(vector<double> i, vector<double> b, vector<double> w, vector<double> &h, double &o){
	h[0]=1/(1+exp(-1*(w[0]*i[0]+w[3]*i[1]+b[0])));
	h[1]=1/(1+exp(-1*(w[1]*i[0]+w[4]*i[1]+b[0])));
	h[2]=1/(1+exp(-1*(w[2]*i[0]+w[5]*i[1]+b[0])));

	h[3]=1/(1+exp(-1*(w[6]*h[0]+w[10]*h[1]+w[14]*h[2]+b[1])));
	h[4]=1/(1+exp(-1*(w[7]*h[0]+w[11]*h[1]+w[15]*h[2]+b[1])));
	h[5]=1/(1+exp(-1*(w[8]*h[0]+w[12]*h[1]+w[16]*h[2]+b[1])));
	h[6]=1/(1+exp(-1*(w[9]*h[0]+w[13]*h[1]+w[17]*h[2]+b[1])));

	o=1/(1+exp(-1*(w[18]*h[3]+w[19]*h[4]+w[20]*h[5]+w[21]*h[6]+b[2])));
}

unsigned BasicNeuralNetwork::GenerateValue(vector<double> &weight, int var){
	for(unsigned i=0; i<var; i++)
		weight.push_back(rand() / double(RAND_MAX));
}

unsigned BasicNeuralNetwork::NextInput(vector<double> &inputVals){
    inputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss >> label;
    if (label.compare("in:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            inputVals.push_back(oneValue);
        }
    }

    return inputVals.size();	
}

unsigned BasicNeuralNetwork::TargetOutput(vector<double> &targetOutputVals)
{
    targetOutputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss>> label;
    if (label.compare("out:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            targetOutputVals.push_back(oneValue);
        }
    }
    return targetOutputVals.size();
}

void showVectorVals(string label, vector<double> &v){
	cout << label << " ";
	for(unsigned i = 0; i < v.size(); ++i)
	{
		cout << v[i] << " ";
	}
	cout << endl;
}

int main(){
	Gnuplot gp;
	std::vector<std::pair<double, double> > xy_point;

	BasicNeuralNetwork nn("trainingData.txt");
	vector<double> inputVals,outputVals,weight,bias,hidden;
	vector<double> de_dw;
	double output,outputTarget,error_;

	nn.GenerateValue(weight,MAX_WEIGHT);
	nn.GenerateValue(de_dw,MAX_WEIGHT);
	nn.GenerateValue(bias,MAX_BIAS);
	nn.GenerateValue(hidden,MAX_HIDDEN);

	int trainingPass = 0;
	while(!nn.isEof()){
		
		nn.NextInput(inputVals);
		showVectorVals("Input :", inputVals);
		
		if(nn.TargetOutput(outputVals)==1)outputTarget=outputVals[0];
		else break;
		showVectorVals("Output :", outputVals);

		for(unsigned i=0; i<ITERATION;i++){
			nn.FeedForward(inputVals,bias,weight,hidden,output);
			nn.BackProp(de_dw, weight, hidden, inputVals, output, outputTarget);
			nn.UpdateWeight(weight, de_dw);
		}

		error_=0.5*(outputTarget-output)*(outputTarget-output);

		cout << "Out NN : " << output << endl;
		cout << "Train  : " << trainingPass << endl;
		cout << "Error %: " << error_ << endl;

		xy_point.push_back(std::make_pair(trainingPass, error_));
		++trainingPass;
	}
	gp << "set xrange [0:100]\nset yrange [0:0.0004]\n";
	gp << "plot" << gp.file1d(xy_point) << "with lines title 'error'"<< std::endl;
}
