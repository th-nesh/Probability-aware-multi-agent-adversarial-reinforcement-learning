/*
 * DigitalOutputArrayROS.cpp
 *
 *  Created on: 09.12.2011
 *      Author: indorewala@servicerobotics.eu
 */

#include "DigitalOutputArrayROS.h"
#include <iostream>
#include <sstream>


DigitalOutputArrayROS::DigitalOutputArrayROS()
{
	digital_sub_ = nh_.subscribe("set_digital_values", 1,
			&DigitalOutputArrayROS::setDigitalValuesCallback, this);
}

DigitalOutputArrayROS::~DigitalOutputArrayROS()
{
	digital_sub_.shutdown();
}

void DigitalOutputArrayROS::setDigitalValuesCallback( const robotino_msgs::DigitalReadingsConstPtr& msg)
{
	int numValues = msg->values.size();
	if( numValues > 0 )
	{
		int values[numValues];
		for(int i=0; i <numValues; i++)
		{
			values[i] = (int)msg->values.data()[i];
			//std::cout<< values[i]<< std::endl;
		}


		//std::cout<< values << std::endl;
		//std::cout<< msg->stamp << std::endl;
		//memcpy( values, msg->values.data(), numValues * sizeof(bool) );
		//std::cout<< values << std::endl;
		setValues( values, numValues );
	}
}
