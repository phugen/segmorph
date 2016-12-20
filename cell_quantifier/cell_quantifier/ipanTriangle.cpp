#include "ipanTriangle.hpp"

// Creates a new triangle - specified by the three points - 
// and stores the angle at p.
IpanTriangle::IpanTriangle(IpanPoint *p, IpanPoint *p_minus, IpanPoint *p_plus, double angle)
{
	this->p = p;
	this->p_minus = p_minus;
	this->p_plus = p_plus;
	this->angle = angle;
}

// return the opening angle at the triangle point p.
double IpanTriangle::getAngle()
{
	return this->angle;
}

// Overload << operator to output three points and the angle at p
std::ostream& operator<<(std::ostream &strm, const IpanTriangle &t)
{
	return strm << "t = (p_minus: " << t.p_minus << " " << " p: " << t.p << " p_plus: " << t.p_plus << " angle: " << t.angle << ")" << "\n";
}