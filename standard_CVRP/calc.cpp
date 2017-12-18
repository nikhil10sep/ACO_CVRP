#include <iostream>
#include <cmath>

using namespace std;

int main() {
	float r1, r2;
	cin >> r1 >> r2;
	float dist = 0.00;
	while(true) {
		float d1, d2;
		cin >> d1 >> d2;
		dist += sqrt((r1 - d1) * (r1 - d1) +  (r2 - d2) * (r2 - d2));
		r1 = d1;
		r2 = d2;
		cout << dist << endl;
	}
	return 0;
}