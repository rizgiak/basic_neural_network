#include <iostream>
#include <cmath>
#include <cstdlib>

using namespace std;

int main()
{
	for(int i = 100; i >= 0; --i){
		double n1 = (rand() / double(RAND_MAX));
		double n2 = (rand() / double(RAND_MAX));
		double t = log((log(n1)*sin(n2)*exp(n1))/(exp(n2)*log(n2)*cos(n1)))*n1*n2/exp(log(n1+n2));
		t=abs(t);
		if(t>1)t=sqrt(n1);

		cout << "in: " << n1 << " " << n2 << " " << endl;
		cout << "out: " << t << ""; 
		if(i>0) cout << endl;
	}
}
