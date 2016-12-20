/*
	Provides an implementation of the IPAN corner detection algorithm.
*/

// STDlib headers
#include <limits> // for double_max constant
#include <algorithm> // for min/max functions
#include <iostream>

// IPAN headers
#include "ipan.hpp"
#include "ipanPoint.hpp"
#include "ipanTriangle.hpp"

// misc headers
#include "auxiliary.hpp"


// Calculates sharpness values for each of the corners by trying
// to inscribe a rectangle into the curve, using each point p as a center
// point and choosing two different points p_minus and p_plus to its left and right by
// varying the opening angle of p_alpha.
// Each triangle must satisfy the constraints set by dmin, dmax and alphamax:
//
// dmin^2 <= dist(p, p_plus)^2 <= dmax^2
// dmin^2 <= dist(p, p_minus)^2 <= dmax^2
// 0 <= |alpha| <= alphamax
//
// with alphamax in [-pi, pi]. If one of the conditions is violated,
// the search stops.
//
// The valid triangle with the smallest opening angle alpha(p) is then chosen
// and the corner sharpness is set to pi - alpha(p).
void Ipan::calcStrengths(std::vector<IpanPoint*> curvePoints, double dmin, double dmax, double alphamax)
{
	IpanPoint *p = NULL, *p_minus = NULL, *p_plus = NULL;


	// TODO: how to deal with non-closed curves that would intersect the
	// triangle if wrap-around is used?



	// don't inspect curve if it has less than three points
	// because at least one "middle" point is needed
	// TODO: Think about this again
	if (curvePoints.size() < 3)
	{
		return;
	}

	std::vector<IpanTriangle> triangles; // list of valid triangles for current point p

	// Inspect every curve point (except first and last points)
	for (auto iter = curvePoints.begin(); iter != curvePoints.end(); iter++)
	{
		IpanPoint *pp_minus, *pp_plus; // vectors for angle calculation
		double plusDist, minusDist, dotProduct, normProduct, alpha, absAlpha; // misc variables
		int p_minus_index, p_plus_index; // index of neighbors points in list of points

		// set current point p
		// and set neighbor indices to p's index initially
		p = *iter;
		p->setStrength(-1); // set dummy value for sharpness

		p_minus_index = iter - curvePoints.begin();
		p_plus_index = iter - curvePoints.begin();

		// initialize left neighbor
		//p_minus_index = mod(--p_minus_index, curvePoints.size()); // wrap-around index if needed
		//p_minus = &curvePoints.at(p_minus_index); // move p_minus one slot to the left based on new index

		// initialize right neighbor
		//p_plus_index = mod(++p_plus_index, curvePoints.size());
		//p_plus = &curvePoints.at(p_plus_index); // move p_plus one slot to the right based on new index
			

		// TODO: don't let neighbor pointers "cross"


		// reset triangle list
		triangles.clear();

		// keep inscribing triangles as long as no rule violation occurs
		// TODO: left and right alternatingly or both at once per step?
		while (true)
		{
			// if moving left would make p_minus and p_plus
			// the same point, end search.
			if (p_minus_index - 1 == p_plus_index)
			{
				break;
			}

			// If not, move p_minus one point further to the left ...
			p_minus_index = mod(--p_minus_index, curvePoints.size());
			p_minus = curvePoints.at(p_minus_index);
			

			minusDist = sqrt(pow((p->x - p_minus->x), 2) + pow((p->y - p_minus->y), 2)); // distance between p and p_minus

			// ... and check if distance to left neighbor valid.
			if (!(dmin*dmin <= minusDist && minusDist <= dmax*dmax))
			{
				break;
			}



			// if moving right would make p_minus and p_plus
			// the same point...
			if (p_plus_index + 1 == p_minus_index)
			{
				break;
			}

			// move p_plus one point further to the right
			p_plus_index = mod(++p_plus_index, curvePoints.size());
			p_plus = curvePoints.at(p_plus_index);

			plusDist = sqrt(pow((p->x - p_plus->x), 2) + pow((p->y - p_plus->y), 2)); // distance between p and p_plus

			// ... and check if distance to right neighbor still valid 
			if (!(dmin*dmin <= plusDist && plusDist <= dmax*dmax)) 
			{
				break;
			}


			// calculate opening angle at p with cosine formula:
			// acos((pp_minus <dot> pp_plus) / (||pp_minus|| ||pp_plus||))
			pp_minus = new IpanPoint(p_minus->x - p->x, p_minus->y - p->y); // create p --> p_minus direction vector
			pp_plus = new IpanPoint(p_plus->x - p->x, p_plus->y - p->x); // create p --> p_plus direction vector

			dotProduct = pp_minus->x * pp_plus->x + pp_minus->y * pp_plus->y; // pp_minus <dot> pp_plus
			normProduct = sqrt(pow(pp_minus->x, 2) + pow(pp_minus->y, 2)) * sqrt(pow(pp_plus->x, 2) + pow((pp_plus->y), 2)); // ||pp_minus|| ||pp_plus||

			alpha = acos(dotProduct / normProduct); // get opening angle in radians
			absAlpha = abs(alpha);

			// check if resulting triangle has valid opening angle
			if(!(0 <= absAlpha && absAlpha <= alphamax))
			{
				break;
			}

			// triangle is valid, so add triangle
			// to list of possible triangles with center p
			triangles.emplace(triangles.begin(), IpanTriangle(p, p_minus, p_plus, absAlpha));
		}


		// TODO: Triangle list not needed. Simply save sharpness value and check on-the-fly!


		// all valid triangles found, find sharpest among
		// them and set its sharpness and p's sharpness
		if (!triangles.empty()) // if there is at least one valid triangle for the point
		{
			double sharpestAngle = std::numeric_limits<double>::max();
			for (auto triangle = triangles.begin(); triangle != triangles.end(); triangle++)
			{
				sharpestAngle = std::min(sharpestAngle, abs(triangle->getAngle()));
			}

			p->setStrength(pi - abs(sharpestAngle)); // set sharpest angle as sharpness of p
		}

		else
		{
			std::cout << "No valid triangles for p = " << p << "\n";
		}
	}	
}

// Calculates sharpness values for all points and then filter out those
// which are not valid candidates for corner points.
std::vector<IpanPoint*> Ipan::getCorners(std::vector<IpanPoint*> curvePoints, double dmin, double dmax, double alphamax)
{
	std::vector<IpanPoint*> corners; // final list of valid corner points

	// First pass: Calculate corner strengths for each point of the curve
	Ipan::calcStrengths(curvePoints, dmin, dmax, alphamax);

	// Second pass: A point is discarded if it has a valid neighbor p_v
	// which has a higher sharpness than p.
	// "Valid" neighbors satisfy ||p - p_v||^2 <= d_max^2 .
	for (auto point = curvePoints.begin(); point != curvePoints.end(); point++)
	{
		// skip points that didn't have any admissable triangles
		if ((*point)->getStrength() == -1)
		{
			continue;
		}

		int left_index = mod(curvePoints.begin() - point - 1, curvePoints.size());
		int right_index = mod(curvePoints.begin() - point + 1, curvePoints.size());

		IpanPoint* left = curvePoints.at(left_index);
		IpanPoint* right = curvePoints.at(right_index);

		// if both neighbor points of the current point are invalid,
		// or are valid but not sharper than the current point,
		// designate the current point a corner.
		if (!(
			left->getStrength() != -1 && // left neighbor assigned a sharpness value?
			left->getStrength() < (*point)->getStrength() && // sharper than p?
			sqrt(pow((*point)->x - left->x, 2) + pow((*point)->y - left->y, 2) <= dmax*dmax))) // valid?
		{
			if (!(
				right->getStrength() != -1 && // right neighbor assigned a sharpness value?
				right->getStrength() < (*point)->getStrength() && // sharper than p?
				sqrt(pow((*point)->x - right->x, 2) + pow((*point)->y - right->y, 2) <= dmax*dmax))) //  valid?
			{
				corners.emplace(corners.begin(), *point); // p is a corner.
			}
		}
	}

	return corners;
}