#include "ros/ros.h"
#include "iostream"
#include "ackermann_msgs/AckermannDriveStamped.h"


class Controller{
	public:
		// default constructor
		Controller();

	private:
		// node handle
		ros::NodeHandle nodeHandle;
		// human sub
		ros::Subscriber humanSub;
		// lidar action sub
		ros::Subscriber lidarSub;
		// neural network sub
		ros::Subscriber nnSub;

		// publisher
		ros::Publisher actionPub;

		// published msg
		ackermann_msgs::AckermannDriveStamped::ConstPtr humanMsg;
		ackermann_msgs::AckermannDriveStamped::ConstPtr lidarMsg;
		ackermann_msgs::AckermannDriveStamped::ConstPtr nnMsg;

		// subscriber functions
		void humanCallBack(const ackermann_msgs::AckermannDriveStamped::ConstPtr& msg);
		void lidarCallBack(const ackermann_msgs::AckermannDriveStamped::ConstPtr& msg);
		void nnCallBack(const ackermann_msgs::AckermannDriveStamped::ConstPtr& msg);
		void pubAction();
};

Controller::Controller(){
	actionPub = nodeHandle.advertise<ackermann_msgs::AckermannDriveStamped>("ackermann_cmd", 1, true);
	humanSub = nodeHandle.subscribe("human_cmd", 10, &Controller::humanCallBack, this);
	lidarSub = nodeHandle.subscribe("lidar_cmd", 10, &Controller::lidarCallBack, this);
	nnSub = nodeHandle.subscribe("nn_cmd", 10, &Controller::nnCallBack, this);
}

void Controller::pubAction(){
	if(humanMsg){
		actionPub.publish(*humanMsg);
	}else if(lidarMsg){
		actionPub.publish(*lidarMsg);
		
	}else if(nnMsg){
		actionPub.publish(*nnMsg);

	}else{
		ackermann_msgs::AckermannDriveStamped msg;
		actionPub.publish(msg);
	}
}

void Controller::humanCallBack(const ackermann_msgs::AckermannDriveStamped::ConstPtr& msg){
	humanMsg = msg;
	pubAction();
}


void Controller::lidarCallBack(const ackermann_msgs::AckermannDriveStamped::ConstPtr& msg){
	lidarMsg = msg;
	pubAction();
}


void Controller::nnCallBack(const ackermann_msgs::AckermannDriveStamped::ConstPtr& msg){
	nnMsg = msg;
	pubAction();
}



int main(int argc, char **argv){
	ros::init(argc, argv, "controller");
	ROS_INFO("INITIALIZING CONTROLLER");

	Controller controller;
	ros::spin();
	return 0;
}
