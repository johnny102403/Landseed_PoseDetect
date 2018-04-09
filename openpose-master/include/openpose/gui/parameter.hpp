#pragma once
#ifndef GUI_PARAMETER_HPP
#define GUI_PARAMETER_HPP

#include <stdio.h>

extern double shoulder_height_differnce, hand_height_difference;
extern double LElbow_Angle, RElbow_Angle, LShouder_Angle, RShouder_Angle, LKnee_Angle, RKnee_Angle;
extern bool is_start;
extern std::string shouder_vec ,hand_vec ,distance_vec ,knee_distance_vec ,back_vec,warn_str;
double get_shoulder_height_differnce();
double get_hand_height_difference();


#endif