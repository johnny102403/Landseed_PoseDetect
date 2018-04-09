#include <opencv2/opencv.hpp> // cv::imshow, cv::waitKey, cv::namedWindow, cv::setWindowProperty
#include <opencv2/highgui/highgui.hpp> // cv::imshow, cv::waitKey, cv::namedWindow, cv::setWindowProperty
#include <openpose/gui/frameDisplayer.hpp>
#include <string>
#include <cvui.h>
#include <openpose/gui/parameter.hpp>
#include "openpose/gui/putText.hpp"

bool is_start = false;
bool checked = false;
bool checked2 = true;
int count = 0.0;
double countFloat = 0.0;
double trackbarValue = 0.0;
double LElbow_Angle = 0.0, RElbow_Angle = 0.0, LShouder_Angle = 0.0, RShouder_Angle = 0.0, LKnee_Angle = 0.0, RKnee_Angle = 0.0;

double shoulder_height_differnce = 0.5;
double hand_height_difference = 15.0;
using namespace cv;
using namespace std;

std::string str("Start");

cv::Mat show_frame = cv::Mat(cv::Size(1280, 960), 0);

double get_shoulder_height_differnce(void) {
	return hand_height_difference;

}

double get_hand_height_difference(void) {
	return hand_height_difference;
}

namespace op
{
	FrameDisplayer::FrameDisplayer(const std::string& windowedName, const Point<int>& initialWindowedSize, const bool fullScreen) :
		mWindowName{ windowedName },
		mWindowedSize{ initialWindowedSize },
		mGuiDisplayMode{ (fullScreen ? GuiDisplayMode::FullScreen : GuiDisplayMode::Windowed) }
	{
		try
		{
			// If initial window size = 0 --> initialize to 640x480
			if (mWindowedSize.x <= 0 || mWindowedSize.y <= 0)
				mWindowedSize = Point<int>{ 640, 480 };
		}
		catch (const std::exception& e)
		{
			error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		}
	}

	void FrameDisplayer::initializationOnThread()
	{
		try
		{
			setGuiDisplayMode(mGuiDisplayMode);

			const cv::Mat blackFrame(mWindowedSize.y, mWindowedSize.x, CV_32FC3, { 0,0,0 });


			FrameDisplayer::displayFrame(blackFrame);
			cv::waitKey(1); // This one will show most probably a white image (I guess the program does not have time to render in 1 msec)
							// cv::waitKey(1000); // This one will show the desired black image
		}
		catch (const std::exception& e)
		{
			error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		}
	}

	void FrameDisplayer::setGuiDisplayMode(const GuiDisplayMode displayMode)
	{
		try
		{
			mGuiDisplayMode = displayMode;

			// Setting output resolution
			cv::namedWindow(mWindowName, CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);





			if (mGuiDisplayMode == GuiDisplayMode::FullScreen)
				cv::setWindowProperty(mWindowName, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
			else if (mGuiDisplayMode == GuiDisplayMode::Windowed)
			{
				cv::resizeWindow(mWindowName, mWindowedSize.x, mWindowedSize.y);
				cv::setWindowProperty(mWindowName, CV_WND_PROP_FULLSCREEN, CV_WINDOW_NORMAL);
			}
			else
				error("Unknown GuiDisplayMode", __LINE__, __FUNCTION__, __FILE__);
		}
		catch (const std::exception& e)
		{
			error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		}
	}

	void FrameDisplayer::switchGuiDisplayMode()
	{
		try
		{
			if (mGuiDisplayMode == GuiDisplayMode::FullScreen)
				setGuiDisplayMode(GuiDisplayMode::Windowed);
			else if (mGuiDisplayMode == GuiDisplayMode::Windowed)
				setGuiDisplayMode(GuiDisplayMode::FullScreen);
			else
				error("Unknown GuiDisplayMode", __LINE__, __FUNCTION__, __FILE__);
		}
		catch (const std::exception& e)
		{
			error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		}
	}

	void FrameDisplayer::displayFrame(const cv::Mat& frame, const int waitKeyValue)
	{
		//parameter initialize


		try
		{
			// If frame > window size --> Resize window
			if (mWindowedSize.x < frame.cols || mWindowedSize.y < frame.rows)
			{
				mWindowedSize.x = max(mWindowedSize.x, frame.cols);
				mWindowedSize.y = max(mWindowedSize.y, frame.rows);
				cv::resizeWindow(mWindowName, mWindowedSize.x, mWindowedSize.y);
				cv::waitKey(1); // This one will show most probably a white image (I guess the program does not have time to render in 1 msec)
			}
			frame.copyTo(show_frame);



			//cvui
			cvui::init(mWindowName);

			cvui::window(show_frame, 512, 0, 512, 768, "Setting");
			if (cvui::button(show_frame, 550, 50, str)) {
				if (str == "Start") {
					str = "Stop";
					is_start = true;
				}
				else {
					str = "Start";
					is_start = false;
				}
				std::cout << "Button clicked" << std::endl;
			}

			Mat img = imread("src.jpg");
			putTextZH(show_frame, "¥ª¤â¨y¨¤«×:", cv::Point(550, 115), Scalar(255, 255, 255), 15, "Arial");
			//cvui::text(show_frame, 550, 120, "LElbow_Angle:");

			cvui::text(show_frame, 650, 120, std::to_string(LElbow_Angle));

			putTextZH(show_frame, "¥k¤â¨y¨¤«×:", cv::Point(768, 115), Scalar(255, 255, 255), 15, "Arial");
			//cvui::text(show_frame, 768, 120, "RElbow_Angle:");
			cvui::text(show_frame, 868, 120, std::to_string(RElbow_Angle));

			putTextZH(show_frame, "¥ªªÓ»H¨¤«×:", cv::Point(550, 185), Scalar(255, 255, 255), 15, "Arial");
			//cvui::text(show_frame, 550, 200, "LShouder_Angle:");
			cvui::text(show_frame, 660, 190, std::to_string(LShouder_Angle));

			putTextZH(show_frame, "¥kªÓ»H¨¤«×:", cv::Point(768, 185), Scalar(255, 255, 255), 15, "Arial");
			//cvui::text(show_frame, 768, 200, "RShouder_Angle:");
			cvui::text(show_frame, 878, 190, std::to_string(RShouder_Angle));

			putTextZH(show_frame, "¥ª½¥¨¤«×:", cv::Point(550, 255), Scalar(255, 255, 255), 15, "Arial");
			//cvui::text(show_frame, 550, 280, "LKnee_Angle:");
			cvui::text(show_frame, 650, 260, std::to_string(LKnee_Angle));

			putTextZH(show_frame, "¥k½¥¨¤«×:", cv::Point(768, 255), Scalar(255, 255, 255), 15, "Arial");
			//cvui::text(show_frame, 768, 280, "RKnee_Angle:");
			cvui::text(show_frame, 868, 260, std::to_string(RKnee_Angle));

			putTextZH(show_frame, "¨âªÓ°¾±×¨¤«×¥¿­tªùÂe­È:", cv::Point(550, 320), Scalar(255, 255, 255), 13, "Arial");
			cvui::trackbar(show_frame, 550, 360, 250, &hand_height_difference, 0., 30.);

			putTextZH(show_frame, "«e³¼¤ñ¨Ò:", cv::Point(550, 430), Scalar(255, 255, 255), 13, "Arial");
			cvui::trackbar(show_frame, 550, 470, 250, &shoulder_height_differnce, 0., 1.);


			cvui::counter(show_frame, 840, 380, &hand_height_difference, 0.1, "%.1f");
			cvui::counter(show_frame, 840, 490, &shoulder_height_differnce, 0.1, "%.1f");




			cvui::window(show_frame, 560, 550, 400, 200, "Warn!");

			if (warn_str != "") {
				char *cstr = &warn_str[0u];
				putTextZH(show_frame, cstr, cv::Point(580, 580), Scalar(255, 255, 255), 13, "Arial");
			}

			cvui::update();

			//cvui::trackbar(show_frame, 550, 240, 150, &hand_height_difference, 0., 1.);


			//cvui::checkbox(show_frame, 550, 280, "Checkbox", &checked);
			//cvui::checkbox(show_frame, 550, 320, "A checked checkbox", &checked2);

			cvui::update();


			//cv::Mat frame_set = cv::Mat(show_frame, cv::Rect(0, 0, 1024, 768));
			//cv::Mat frame_control_set = cv::Mat(show_frame, cv::Rect(1024, 0, 512, 768));

			//resize(frame, frame, cv::Size(width/2, height), 0, 0, CV_INTER_LINEAR);
			//resize(frame_control, frame_control, cv::Size(width/2, height), 0, 0, CV_INTER_LINEAR);


			//frame.copyTo(frame_set);
			//frame_control.copyTo(frame_control_set);


			cv::imshow(mWindowName, show_frame);
			if (waitKeyValue != -1)
				cv::waitKey(waitKeyValue);
		}
		catch (const std::exception& e)
		{
			error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		}
	}
}
