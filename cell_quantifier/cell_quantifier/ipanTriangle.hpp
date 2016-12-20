#pragma once

/*
	Header file for IPAN triangles. Those are defined by a point p, the "middle" point,
	and two others, p_minus (left of p) and p_plus (right of p).
	Also stores the opening angle of the triangle, which is the angle between the
	vectors p->p_minus and p->p_plus.
*/

// IPAN headers
#include "ipanPoint.hpp"


class IpanTriangle
{
private:
	IpanPoint *p, *p_minus, *p_plus; // corner points of triangle
	double angle; // angle at p
	friend std::ostream& operator<<(std::ostream&, const IpanTriangle&);

public:
	// Constructors
	IpanTriangle() = delete; // standard constructor forbidden
	IpanTriangle(IpanPoint *p, IpanPoint *p_minus, IpanPoint *p_plus, double angle);

	double getAngle();
};