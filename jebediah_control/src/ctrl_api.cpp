#include "ros/ros.h"
#include <control_msgs/JointControllerState.h>
#include <jcontrol_msgs/State.h>
#include <jcontrol_msgs/Action.h>
#include <gazebo_msgs/ContactsState.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Float64.h>
#include <gazebo_msgs/ModelStates.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Twist.h>
#include <iostream>
#include <random>
using namespace std;

class controller {
  private:
  // Variables
    // gaussian noise generator
    std::default_random_engine generator;
    double P = 0.034;  // distribution (80%) within 1 degree
    double U = 0;     // center
    // Auxiliary variables
    static const int pubhz = 50;
    ros::NodeHandle n;
    jcontrol_msgs::State state;
    float angles[12];
    bool ground[4];
    bool selfCollide[4];
    double ang_vel;
    // Joint position subscribers: joint_leg/joint
    ros::Subscriber joint_00, joint_10 , joint_20 , joint_30;
    ros::Subscriber joint_01, joint_11 , joint_21 , joint_31;
    ros::Subscriber joint_02, joint_12 , joint_22 , joint_32;
    // Robot position
    ros::Subscriber position_sub;
    geometry_msgs::Point position;
    geometry_msgs::Twist twist;
    // Toouch sensor topic
    ros::Subscriber touch_sensor_1, touch_sensor_2, touch_sensor_3, touch_sensor_4;
    // IMU
    ros::Subscriber IMU;
    sensor_msgs::Imu imu;
    // Action
    ros::Subscriber action_sub;
    ros::Publisher action_pub[12];
    std_msgs::Float64 action_val;
  // FUNCTIONS
    // State publisher
    ros::Publisher pub;
    void joint_callback_00(const control_msgs::JointControllerState::ConstPtr& joint){
      std::normal_distribution<double> distribution(U,P);
      angles[0] = joint->process_value + distribution(generator);
    }
    void joint_callback_01(const control_msgs::JointControllerState::ConstPtr& joint){
      std::normal_distribution<double> distribution(U,P);
      angles[1] = -1*joint->process_value + distribution(generator);
    }
    void joint_callback_02(const control_msgs::JointControllerState::ConstPtr& joint){
      std::normal_distribution<double> distribution(U,P);
      angles[2] = -1*joint->process_value + distribution(generator);
    }
    void joint_callback_10(const control_msgs::JointControllerState::ConstPtr& joint){
      std::normal_distribution<double> distribution(U,P);
      angles[3] = joint->process_value + distribution(generator);
    }
    void joint_callback_11(const control_msgs::JointControllerState::ConstPtr& joint){
      std::normal_distribution<double> distribution(U,P);
      angles[4] = joint->process_value + distribution(generator);
    }
    void joint_callback_12(const control_msgs::JointControllerState::ConstPtr& joint){
      std::normal_distribution<double> distribution(U,P);
      angles[5] = joint->process_value + distribution(generator);
    }
    void joint_callback_20(const control_msgs::JointControllerState::ConstPtr& joint){
      std::normal_distribution<double> distribution(U,P);
      angles[6] = joint->process_value + distribution(generator);
    }
    void joint_callback_21(const control_msgs::JointControllerState::ConstPtr& joint){
      std::normal_distribution<double> distribution(U,P);
      angles[7] = joint->process_value + distribution(generator);
    }
    void joint_callback_22(const control_msgs::JointControllerState::ConstPtr& joint){
      std::normal_distribution<double> distribution(U,P);
      angles[8] = joint->process_value + distribution(generator);
    }
    void joint_callback_30(const control_msgs::JointControllerState::ConstPtr& joint){
      std::normal_distribution<double> distribution(U,P);
      angles[9] = joint->process_value + distribution(generator);
    }
    void joint_callback_31(const control_msgs::JointControllerState::ConstPtr& joint){
      std::normal_distribution<double> distribution(U,P);
      angles[10] = -1*joint->process_value + distribution(generator);
    }
    void joint_callback_32(const control_msgs::JointControllerState::ConstPtr& joint){
      std::normal_distribution<double> distribution(U,P);
      angles[11] = -1*joint->process_value + distribution(generator);
    }
    void touch_callback(const gazebo_msgs::ContactsState::ConstPtr& contact){
      int tibia = int(contact->header.frame_id[6]) - 48;
      int gnd = 0;
      int self = 0;
      for (int i = 0; i < contact->states.size(); i++){
        if(contact->states[i].collision2_name.find("ground") < contact->states[i].collision2_name.npos){
          if (contact->states[i].total_wrench.force.z > contact->states[i].total_wrench.force.x && contact->states[i].total_wrench.force.z > contact->states[i].total_wrench.force.y){
            gnd = 1;
          }
        }
        else if (contact->states[i].collision2_name.find("jebediah") < contact->states[i].collision2_name.npos){
          self = 1;
        }
      }
      ground[tibia-1] = gnd;
      selfCollide[tibia-1] = self;
    }
  void IMU_callback(const sensor_msgs::Imu::ConstPtr& data){
    imu = *data;
  }
  void position_callback(const gazebo_msgs::ModelStates::ConstPtr& model_state){
    position = model_state->pose[1].position;
    twist = model_state->twist[1];
  }
  void action_callback(const jcontrol_msgs::Action::ConstPtr& act){
    int j = 1;
    for(int i = 0; i < 12; i++){
      if(i == 1 || i == 2 || i == 10 || i == 11) j = -1;
      else j = 1;
      action_val.data = j*act->Actions[i];
      action_pub[i].publish(action_val);
    }
  }
  public:
    controller(){
      // Random gaussian noise generator
      //const int nrolls=10000;  // number of experiments
      //const int nstars=200;    // maximum number of stars to distribute
      // std::normal_distribution<double> distribution(U,P);

      // int p[20]={};

      // for (int i=0; i<nrolls; ++i) {
      //   double number = distribution(generator);
      //   float c = 10+20*number/P; // normalize and spread trhough plot range
      //   cout << c <<  endl;
      //   if ((c>=0.0)&&(c<20.0)) ++p[int(c)];
      // }

      // std::cout << "normal_distribution ("<< P << ","<< U <<"):" << std::endl;

      // for (int i=0; i<20; ++i) {
      //   std::cout << (float)i/10 << "-" << (float)(i+1)/10 << ": ";
      //   std::cout << std::string(p[i]*nstars/nrolls,'*') << std::endl;
      // }

      ros::service::waitForService("gazebo/reset_simulation", -1);
      //Publisher
      ROS_INFO("Setting publisher");
      pub = n.advertise<jcontrol_msgs::State> ("jebediah/State", 1);
      ROS_INFO("Subscribing to topics");
      // Subscribe to joint angles topics
      joint_00 = n.subscribe("/jebediah/coxa_1_position_controller/state", 1, &controller::joint_callback_00, this);
      joint_10 = n.subscribe("/jebediah/coxa_2_position_controller/state", 1, &controller::joint_callback_10, this);
      joint_20 = n.subscribe("/jebediah/coxa_3_position_controller/state", 1, &controller::joint_callback_20, this);
      joint_30 = n.subscribe("/jebediah/coxa_4_position_controller/state", 1, &controller::joint_callback_30, this);
      joint_01 = n.subscribe("/jebediah/femur_1_position_controller/state", 1, &controller::joint_callback_01, this);
      joint_11 = n.subscribe("/jebediah/femur_2_position_controller/state", 1, &controller::joint_callback_11, this);
      joint_21 = n.subscribe("/jebediah/femur_3_position_controller/state", 1, &controller::joint_callback_21, this);
      joint_31 = n.subscribe("/jebediah/femur_4_position_controller/state", 1, &controller::joint_callback_31, this);
      joint_02 = n.subscribe("/jebediah/tibia_1_position_controller/state", 1, &controller::joint_callback_02, this);
      joint_12 = n.subscribe("/jebediah/tibia_2_position_controller/state", 1, &controller::joint_callback_12, this);
      joint_22 = n.subscribe("/jebediah/tibia_3_position_controller/state", 1, &controller::joint_callback_22, this);
      joint_32 = n.subscribe("/jebediah/tibia_4_position_controller/state", 1, &controller::joint_callback_32, this);
      // Position sensor
      position_sub = n.subscribe("gazebo/model_states", 1, &controller::position_callback, this);
      // Touch sensor
      touch_sensor_1 = n.subscribe("/jebediah/tibia_1_conctact_sensor", 1, &controller::touch_callback, this);
      touch_sensor_2 = n.subscribe("/jebediah/tibia_2_conctact_sensor", 1, &controller::touch_callback, this);
      touch_sensor_3 = n.subscribe("/jebediah/tibia_3_conctact_sensor", 1, &controller::touch_callback, this);
      touch_sensor_4 = n.subscribe("/jebediah/tibia_4_conctact_sensor", 1, &controller::touch_callback, this);
      // IMU sensor
      IMU = n.subscribe("/jebediah/imu_data", 1, &controller::IMU_callback, this);
      // Action
      action_sub = n.subscribe("jebediah/Action", 1, &controller::action_callback, this);
      action_pub[0] = n.advertise<std_msgs::Float64> ("jebediah/coxa_1_position_controller/command", 1);
      action_pub[1] = n.advertise<std_msgs::Float64> ("jebediah/femur_1_position_controller/command", 1);
      action_pub[2] = n.advertise<std_msgs::Float64> ("jebediah/tibia_1_position_controller/command", 1);
      action_pub[3] = n.advertise<std_msgs::Float64> ("jebediah/coxa_2_position_controller/command", 1);
      action_pub[4] = n.advertise<std_msgs::Float64> ("jebediah/femur_2_position_controller/command", 1);
      action_pub[5] = n.advertise<std_msgs::Float64> ("jebediah/tibia_2_position_controller/command", 1);
      action_pub[6] = n.advertise<std_msgs::Float64> ("jebediah/coxa_3_position_controller/command", 1);
      action_pub[7] = n.advertise<std_msgs::Float64> ("jebediah/femur_3_position_controller/command", 1);
      action_pub[8] = n.advertise<std_msgs::Float64> ("jebediah/tibia_3_position_controller/command", 1);
      action_pub[9] = n.advertise<std_msgs::Float64> ("jebediah/coxa_4_position_controller/command", 1);
      action_pub[10] = n.advertise<std_msgs::Float64> ("jebediah/femur_4_position_controller/command", 1);
      action_pub[11] = n.advertise<std_msgs::Float64> ("jebediah/tibia_4_position_controller/command", 1);
      // Publisher loop
      ROS_INFO("Publishing at %dhz.", pubhz);
      ros::Rate loop_rate(pubhz);
      while(ros::ok()){
        for(int i = 0; i < 12; i++)
          state.Angles[i] = angles[i];
        for(int i = 0; i < 4; i++){
          state.Self_Collision[i] = selfCollide[i];
          state.Ground_Collision[i] = ground[i];
        }
        state.IMU = imu;
        state.Position = position;
        state.Twist = twist;
        pub.publish(state);
        ros::spinOnce();
        loop_rate.sleep();
      }
    }
};

int main (int argc, char* argv[]) {
  ros::init(argc, argv, "jebediah_ctrl_api");
  controller c;
  ros::spin();
  return 0;
}