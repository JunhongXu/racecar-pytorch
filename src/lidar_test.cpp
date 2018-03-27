#include "ros/ros.h"
#include "sensor_msgs/LaserScan.h"
#include "iostream"
#include "ackermann_msgs/AckermannDrive.h"
#include "ackermann_msgs/AckermannDriveStamped.h"

#define DISTANCE_T  0.5


class LidarController{
	public:
		// default constructor
		LidarController();
	// private variables
	private:
		// ros related objects
		ros::NodeHandle nodeHandle;
		ros::Publisher actionPub;
	       	ros::Subscriber lidarSub;
		
		void lidarCallback(const sensor_msgs::LaserScan::ConstPtr& msg);
		void sendAction(const ackermann_msgs::AckermannDriveStamped& msg);
		ackermann_msgs::AckermannDriveStamped calculateAction(unsigned long angle, float distance);
};


LidarController::LidarController(){
	actionPub = nodeHandle.advertise<ackermann_msgs::AckermannDriveStamped>("lidar_cmd", 1, true);
	lidarSub = nodeHandle.subscribe("scan", 10, &LidarController::lidarCallback, this);
}

ackermann_msgs::AckermannDriveStamped LidarController::calculateAction(unsigned long angle, float distance){
	ackermann_msgs::AckermannDriveStamped msg;
	
	if(0<= angle && angle<= 60){
		msg.drive.steering_angle = -1;
		ROS_INFO("GO RIGHT");
		ROS_INFO("Distance, %.4f", distance);
	}else if(300<=angle && angle<=360){
		msg.drive.steering_angle = 1;
		ROS_INFO("GO LEFT");
		ROS_INFO("Distance, %.4f", distance);
	}
	sendAction(msg);
	return msg;	
}

void LidarController::sendAction(const ackermann_msgs::AckermannDriveStamped& msg){
	actionPub.publish(msg);	
}

void LidarController::lidarCallback(const sensor_msgs::LaserScan::ConstPtr& msg){
	// read lidar messages
	for(unsigned long i=0; i < msg->ranges.size(); i++){
		float distance = msg->ranges[i];
		if (distance < DISTANCE_T){
			calculateAction(i, distance);
		}
	}
}	



int main(int argc, char **argv){
	ros::init(argc, argv, "lidar_controller");
	ROS_INFO("Initializing lidar");
	LidarController controller;
	ros::spin();
	return 0;  
}
