#include "ros/ros.h"
#include <control_msgs/JointControllerState.h>
#include <jcontrol_msgs/State.h>
#include <gazebo_msgs/ContactsState.h>


#include <iostream>

using namespace std;

class controller {
  private:
    static const int pubhz = 50;
    ros::NodeHandle n;
    jcontrol_msgs::State state; 
    float angles[12];
    bool ground[4];
    bool selfCollide[4];
    // Joint position subscribers: joint_leg/joint
    ros::Subscriber joint_00, joint_10 , joint_20 , joint_30;
    ros::Subscriber joint_01, joint_11 , joint_21 , joint_31;
    ros::Subscriber joint_02, joint_12 , joint_22 , joint_32;
    // Toouch sensor topic
    ros::Subscriber touch_sensor_1, touch_sensor_2, touch_sensor_3, touch_sensor_4;
    // State publisher
    ros::Publisher pub;

  public:
    void joint_callback_00(const control_msgs::JointControllerState::ConstPtr& joint){
      angles[0] = joint->process_value;
    }
    void joint_callback_01(const control_msgs::JointControllerState::ConstPtr& joint){
      angles[1] = -1*joint->process_value;
    }
    void joint_callback_02(const control_msgs::JointControllerState::ConstPtr& joint){
      angles[2] = -1*joint->process_value;
    }
    void joint_callback_10(const control_msgs::JointControllerState::ConstPtr& joint){
      angles[3] = joint->process_value;
    }
    void joint_callback_11(const control_msgs::JointControllerState::ConstPtr& joint){
      angles[4] = joint->process_value;
    }
    void joint_callback_12(const control_msgs::JointControllerState::ConstPtr& joint){
      angles[5] = joint->process_value;
    }
    void joint_callback_20(const control_msgs::JointControllerState::ConstPtr& joint){
      angles[6] = joint->process_value;
    }
    void joint_callback_21(const control_msgs::JointControllerState::ConstPtr& joint){
      angles[7] = joint->process_value;
    }
    void joint_callback_22(const control_msgs::JointControllerState::ConstPtr& joint){
      angles[8] = joint->process_value;
    }
    void joint_callback_30(const control_msgs::JointControllerState::ConstPtr& joint){
      angles[9] = joint->process_value;
    }
    void joint_callback_31(const control_msgs::JointControllerState::ConstPtr& joint){
      angles[10] = -1*joint->process_value;
    }
    void joint_callback_32(const control_msgs::JointControllerState::ConstPtr& joint){
      angles[11] = -1*joint->process_value;
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
    controller(){
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
      // Touch sensor
      touch_sensor_1 = n.subscribe("/jebediah/tibia_1_conctact_sensor", 1, &controller::touch_callback, this);
      touch_sensor_2 = n.subscribe("/jebediah/tibia_2_conctact_sensor", 1, &controller::touch_callback, this);
      touch_sensor_3 = n.subscribe("/jebediah/tibia_3_conctact_sensor", 1, &controller::touch_callback, this);
      touch_sensor_4 = n.subscribe("/jebediah/tibia_4_conctact_sensor", 1, &controller::touch_callback, this);
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