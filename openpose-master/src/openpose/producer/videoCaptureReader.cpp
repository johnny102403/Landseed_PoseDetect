#include <iostream>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/string.hpp>
#include <openpose/producer/videoCaptureReader.hpp>
#include <opencv2\imgproc\imgproc.hpp>  

namespace op
{
	VideoCaptureReader::VideoCaptureReader(const int index, const bool throwExceptionIfNoOpened) :
		Producer{ ProducerType::Webcam },
		mVideoCapture{ 0 },
		mVideoCapture2{ 1 }
	{
		try
		{
			// Make sure video capture was opened
			if (throwExceptionIfNoOpened && !isOpened())
				error("VideoCapture (webcam) could not be opened.", __LINE__, __FUNCTION__, __FILE__);
			if (throwExceptionIfNoOpened && !isOpened2())
				error("VideoCapture (webcam2) could not be opened.", __LINE__, __FUNCTION__, __FILE__);
		}
		catch (const std::exception& e)
		{
			error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		}
	}

	VideoCaptureReader::VideoCaptureReader(const std::string& path, const ProducerType producerType) :
		Producer{ producerType },
		mVideoCapture{ path }
	{
		try
		{
			// Make sure only video or IP camera
			if (producerType != ProducerType::IPCamera && producerType != ProducerType::Video)
				error("VideoCapture with an input path must be IP camera or video.",
					__LINE__, __FUNCTION__, __FILE__);
			// Make sure video capture was opened
			if (!isOpened())
				error("VideoCapture (IP camera/video) could not be opened for path: '" + path + "'. If"
					" it is a video path, is the path correct?", __LINE__, __FUNCTION__, __FILE__);
		}
		catch (const std::exception& e)
		{
			error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		}
	}

	VideoCaptureReader::~VideoCaptureReader()
	{
		try
		{
			release();
		}
		catch (const std::exception& e)
		{
			error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		}
	}

	std::string VideoCaptureReader::getFrameName()
	{
		try
		{
			const auto stringLength = 12u;
			return toFixedLengthString(fastMax(0ll, longLongRound(get(CV_CAP_PROP_POS_FRAMES))), stringLength);
		}
		catch (const std::exception& e)
		{
			error(e.what(), __LINE__, __FUNCTION__, __FILE__);
			return "";
		}
	}

	cv::Mat VideoCaptureReader::getRawFrame()
	{
		try
		{
			int width, height;
			cv::Mat frame, frame1;

			mVideoCapture >> frame1;
			mVideoCapture2 >> frame;


			//width = frame.cols;
			//height = frame.rows;

			cv::Mat new_img(960, 1280, CV_8UC3);
			cv::Mat webcam1 = cv::Mat(new_img, cv::Rect(0, 0, 640, 480));
			cv::Mat webcam2 = cv::Mat(new_img, cv::Rect(0, 480, 640, 480));


			resize(frame, frame, cv::Size(640, 480), 0, 0, CV_INTER_LINEAR);
			resize(frame1, frame1, cv::Size(640, 480), 0, 0, CV_INTER_LINEAR);

			frame.copyTo(webcam1);
			frame1.copyTo(webcam2);

			return new_img;

		}
		catch (const std::exception& e)
		{
			error(e.what(), __LINE__, __FUNCTION__, __FILE__);
			return cv::Mat();
		}
	}

	void VideoCaptureReader::release()
	{
		try
		{
			if (mVideoCapture.isOpened())
			{
				mVideoCapture.release();
				log("cv::VideoCapture released.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
			}
			if (mVideoCapture2.isOpened())
			{
				mVideoCapture2.release();
				log("cv::VideoCapture2 released.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
			}
		}
		catch (const std::exception& e)
		{
			error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		}
	}

	double VideoCaptureReader::get(const int capProperty)
	{
		try
		{
			// Specific cases
			// If rotated 90 or 270 degrees, then width and height is exchanged
			if ((capProperty == CV_CAP_PROP_FRAME_WIDTH || capProperty == CV_CAP_PROP_FRAME_HEIGHT) && (get(ProducerProperty::Rotation) != 0. && get(ProducerProperty::Rotation) != 180.))
			{
				if (capProperty == CV_CAP_PROP_FRAME_WIDTH)
					return mVideoCapture.get(CV_CAP_PROP_FRAME_HEIGHT);
				else
					return mVideoCapture.get(CV_CAP_PROP_FRAME_WIDTH);
			}

			// Generic cases
			return mVideoCapture.get(capProperty);
		}
		catch (const std::exception& e)
		{
			error(e.what(), __LINE__, __FUNCTION__, __FILE__);
			return 0.;
		}
	}

	void VideoCaptureReader::set(const int capProperty, const double value)
	{
		try
		{
			mVideoCapture.set(capProperty, value);
		}
		catch (const std::exception& e)
		{
			error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		}
	}
}
