/* SPDX-License-Identifier: LGPL-2.1-or-later */
/*
 * Copyright (C) 2026, kervel
 *
 * Pipeline handler for Intel IPU6 cameras using libcamhal
 *
 * This pipeline handler wraps Intel's Camera HAL (libcamhal) to provide
 * hardware ISP processing for IPU6-based laptop cameras. It delegates
 * all image processing (3A, noise reduction, color correction, etc.) to
 * the Intel PSYS hardware via libcamhal, producing high-quality NV12 output.
 */

#include <algorithm>
#include <climits>
#include <cstring>
#include <map>
#include <memory>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/mman.h>
#include <unistd.h>

#include <libcamera/base/log.h>
#include <libcamera/base/utils.h>

#include <libcamera/camera.h>
#include <libcamera/control_ids.h>
#include <libcamera/controls.h>
#include <libcamera/formats.h>
#include <libcamera/framebuffer.h>
#include <libcamera/property_ids.h>
#include <libcamera/request.h>
#include <libcamera/stream.h>

#include "libcamera/internal/camera.h"
#include "libcamera/internal/device_enumerator.h"
#include "libcamera/internal/dma_buf_allocator.h"
#include "libcamera/internal/framebuffer.h"
#include "libcamera/internal/mapped_framebuffer.h"
#include "libcamera/internal/media_device.h"
#include "libcamera/internal/pipeline_handler.h"
#include "libcamera/internal/request.h"

/* libcamhal C API */
#include <ICamera.h>

namespace libcamera {

LOG_DEFINE_CATEGORY(IPU6)

/* Supported output resolutions for OV01A10 (from HAL XML config) */
static const std::vector<Size> kSupportedSizes = {
	{ 1280, 720 },
	{ 640, 480 },
	{ 640, 360 },
};

class IPU6CameraData : public Camera::Private
{
public:
	IPU6CameraData(PipelineHandler *pipe)
		: Camera::Private(pipe), cameraId_(-1),
		  deviceOpened_(false), streaming_(false)
	{
	}

	~IPU6CameraData()
	{
		stopStreaming();
		if (deviceOpened_) {
			icamera::camera_device_close(cameraId_);
			deviceOpened_ = false;
		}
		free(halBufferAddr_);
		halBufferAddr_ = nullptr;
	}

	int init(int cameraId);
	int configureHalStream(const Size &size);
	int startStreaming();
	void stopStreaming();

	void workerThread();

	int cameraId_;
	std::string sensorName_;
	bool deviceOpened_;
	bool streaming_;

	Stream stream_;
	Size configuredSize_;

	/* Worker thread continuously dqbufs from HAL and delivers */
	std::thread worker_;
	std::mutex queueMutex_;
	std::condition_variable requestCv_;
	std::queue<Request *> pendingRequests_;
	bool workerStop_ = false;
	uint32_t frameSequence_ = 0;

	void deliverFrame(Request *request, void *src, size_t srcSize);

	/* HAL stream info filled by config_streams */
	icamera::stream_t halStream_;
	int halStreamId_ = -1;

	/* Single HAL buffer */
	void *halBufferAddr_ = nullptr;
	int halBufferSize_ = 0;
	icamera::camera_buffer_t halBuffer_;
};

class IPU6CameraConfiguration : public CameraConfiguration
{
public:
	IPU6CameraConfiguration(IPU6CameraData *data);
	Status validate() override;

private:
	IPU6CameraData *data_;
};

class PipelineHandlerIPU6 : public PipelineHandler
{
public:
	PipelineHandlerIPU6(CameraManager *manager);
	~PipelineHandlerIPU6();

	std::unique_ptr<CameraConfiguration> generateConfiguration(Camera *camera,
								     Span<const StreamRole> roles) override;
	int configure(Camera *camera, CameraConfiguration *config) override;

	int exportFrameBuffers(Camera *camera, Stream *stream,
			       std::vector<std::unique_ptr<FrameBuffer>> *buffers) override;

	int start(Camera *camera, const ControlList *controls) override;
	void stopDevice(Camera *camera) override;

	int queueRequestDevice(Camera *camera, Request *request) override;

	bool match(DeviceEnumerator *enumerator) override;

private:
	IPU6CameraData *cameraData(Camera *camera)
	{
		return static_cast<IPU6CameraData *>(camera->_d());
	}

	DmaBufAllocator dmaBufAllocator_;
	bool halInitialized_ = false;
};

/* --- IPU6CameraData implementation --- */

int IPU6CameraData::init(int cameraId)
{
	cameraId_ = cameraId;

	icamera::camera_info_t info;
	int ret = icamera::get_camera_info(cameraId_, info);
	if (ret < 0) {
		LOG(IPU6, Error) << "Failed to get camera info for camera " << cameraId_;
		return ret;
	}

	sensorName_ = info.name ? info.name : "unknown";
	LOG(IPU6, Info) << "Found IPU6 camera " << cameraId_
			<< ": " << sensorName_
			<< " (facing=" << info.facing << ")";

	/* Set camera properties */
	properties_.set(properties::Model, sensorName_);

	/*
	 * The HAL uses FACING_BACK=0, FACING_FRONT=1, but the config
	 * for ov01a10-uf (user-facing) incorrectly sets facing=0.
	 * Use the sensor name suffix as a more reliable indicator:
	 * "-uf" = user-facing (front), "-wf" = world-facing (back).
	 */
	if (sensorName_.find("-uf") != std::string::npos ||
	    info.facing == 1)
		properties_.set(properties::Location, properties::CameraLocationFront);
	else
		properties_.set(properties::Location, properties::CameraLocationBack);

	/* Use the largest supported resolution as the pixel array size */
	Size maxSize = kSupportedSizes.front();
	for (const auto &s : kSupportedSizes) {
		if (s.width * s.height > maxSize.width * maxSize.height)
			maxSize = s;
	}
	properties_.set(properties::PixelArraySize, maxSize);
	properties_.set(properties::PixelArrayActiveAreas,
			{ Rectangle(maxSize) });

	return 0;
}

int IPU6CameraData::configureHalStream(const Size &size)
{
	int ret;

	if (!deviceOpened_) {
		LOG(IPU6, Info) << "Opening camera device " << cameraId_;
		ret = icamera::camera_device_open(cameraId_);
		if (ret < 0) {
			LOG(IPU6, Error) << "Failed to open camera device " << cameraId_
					 << ": " << ret << " (errno=" << errno << ")";
			return ret;
		}
		deviceOpened_ = true;
		LOG(IPU6, Info) << "Camera device " << cameraId_ << " opened successfully";
	}

	/* Configure the output stream */
	memset(&halStream_, 0, sizeof(halStream_));
	halStream_.format = V4L2_PIX_FMT_NV12;
	halStream_.width = size.width;
	halStream_.height = size.height;
	halStream_.memType = V4L2_MEMORY_USERPTR;
	halStream_.field = V4L2_FIELD_ANY;
	halStream_.stride = size.width;
	halStream_.size = size.width * size.height * 3 / 2; /* NV12 */

	icamera::stream_config_t streamConfig;
	streamConfig.num_streams = 1;
	streamConfig.streams = &halStream_;
	streamConfig.operation_mode = 2; /* CAMERA_STREAM_CONFIGURATION_MODE_AUTO */

	ret = icamera::camera_device_config_streams(cameraId_, &streamConfig);
	if (ret < 0) {
		LOG(IPU6, Error) << "Failed to configure HAL streams: " << ret;
		return ret;
	}

	/* After config_streams, the stream id is filled in halStream_.id */
	halStreamId_ = halStream_.id;
	configuredSize_ = size;
	/* Update size/stride from what HAL returned (may have been adjusted) */
	int bufSize = halStream_.size ? halStream_.size
				      : size.width * size.height * 3 / 2;

	LOG(IPU6, Info) << "Configured HAL stream: " << size.width << "x"
			<< size.height << " NV12, stream_id=" << halStreamId_
			<< " size=" << bufSize;

	halBufferSize_ = bufSize;

	/* Allocate internal buffer for HAL frames */
	if (halBufferAddr_) {
		free(halBufferAddr_);
		halBufferAddr_ = nullptr;
	}

	if (posix_memalign(&halBufferAddr_, getpagesize(), halBufferSize_) != 0) {
		LOG(IPU6, Error) << "Failed to allocate HAL buffer";
		return -ENOMEM;
	}

	return 0;
}

int IPU6CameraData::startStreaming()
{
	if (streaming_)
		return 0;

	/* Prepare and queue the HAL buffer before start */
	memset(&halBuffer_, 0, sizeof(halBuffer_));
	halBuffer_.s = halStream_;
	halBuffer_.addr = halBufferAddr_;
	halBuffer_.flags = 0;

	icamera::camera_buffer_t *bufPtr = &halBuffer_;
	int qret = icamera::camera_stream_qbuf(cameraId_, &bufPtr, 1, nullptr);
	if (qret < 0) {
		LOG(IPU6, Error) << "Failed to queue initial HAL buffer: " << qret;
		return qret;
	}

	int ret = icamera::camera_device_start(cameraId_);
	if (ret < 0) {
		LOG(IPU6, Error) << "Failed to start camera device: " << ret;
		return ret;
	}

	streaming_ = true;
	workerStop_ = false;

	/* Start worker thread */
	worker_ = std::thread(&IPU6CameraData::workerThread, this);

	LOG(IPU6, Info) << "Streaming started for camera " << cameraId_;
	return 0;
}

void IPU6CameraData::stopStreaming()
{
	if (!streaming_)
		return;

	/* Signal worker to stop */
	{
		std::lock_guard<std::mutex> lock(queueMutex_);
		workerStop_ = true;
	}
	requestCv_.notify_all();

	if (worker_.joinable())
		worker_.join();

	icamera::camera_device_stop(cameraId_);
	streaming_ = false;

	/* Drain pending requests */
	std::lock_guard<std::mutex> lock(queueMutex_);
	while (!pendingRequests_.empty())
		pendingRequests_.pop();
	frameSequence_ = 0;

	LOG(IPU6, Info) << "Streaming stopped for camera " << cameraId_;
}

void IPU6CameraData::deliverFrame(Request *request, void *src, size_t srcSize)
{
	FrameBuffer *buffer = request->findBuffer(&stream_);
	if (!buffer)
		return;

	MappedFrameBuffer mapped(buffer, MappedFrameBuffer::MapFlag::Write);
	if (!mapped.isValid())
		return;

	const auto &planes = mapped.planes();
	size_t totalSize = 0;
	for (const auto &plane : planes)
		totalSize += plane.size();

	size_t copySize = std::min(totalSize, srcSize);
	uint8_t *srcPtr = static_cast<uint8_t *>(src);
	size_t offset = 0;
	for (const auto &plane : planes) {
		size_t planeBytes = std::min(plane.size(), copySize - offset);
		memcpy(plane.data(), srcPtr + offset, planeBytes);
		offset += planeBytes;
		if (offset >= copySize)
			break;
	}

	FrameMetadata &metadata = buffer->_d()->metadata();
	metadata.status = FrameMetadata::FrameSuccess;
	metadata.sequence = frameSequence_++;
	auto now = std::chrono::steady_clock::now();
	metadata.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
		now.time_since_epoch()).count();

	unsigned int yPlaneSize = configuredSize_.width * configuredSize_.height;
	unsigned int uvPlaneSize = yPlaneSize / 2;
	if (metadata.planes().size() >= 2) {
		metadata.planes()[0].bytesused = yPlaneSize;
		metadata.planes()[1].bytesused = uvPlaneSize;
	} else if (metadata.planes().size() == 1) {
		metadata.planes()[0].bytesused = yPlaneSize + uvPlaneSize;
	}

	request->_d()->metadata().set(controls::SensorTimestamp, metadata.timestamp);
	pipe()->completeBuffer(request, buffer);
	pipe()->completeRequest(request);
}

void IPU6CameraData::workerThread()
{
	LOG(IPU6, Info) << "Worker thread started";

	while (true) {
		/* Dqbuf: get the frame the HAL filled (blocks until ready) */
		icamera::camera_buffer_t *dqBuf = nullptr;
		LOG(IPU6, Info) << "Worker: dqbuf...";
		int ret = icamera::camera_stream_dqbuf(cameraId_, halStreamId_, &dqBuf);
		if (ret < 0) {
			std::lock_guard<std::mutex> lock(queueMutex_);
			if (workerStop_) {
				LOG(IPU6, Info) << "Worker: dqbuf interrupted by stop";
				break;
			}
			LOG(IPU6, Error) << "Worker: dqbuf failed: " << ret;
			break;
		}
		LOG(IPU6, Info) << "Worker: dqbuf ok";

		/* Wait for a request to deliver this frame to */
		Request *request = nullptr;
		{
			std::unique_lock<std::mutex> lock(queueMutex_);
			LOG(IPU6, Info) << "Worker: waiting for request, pending=" << pendingRequests_.size();
			requestCv_.wait(lock, [this] {
				return workerStop_ || !pendingRequests_.empty();
			});
			if (workerStop_) {
				LOG(IPU6, Info) << "Worker: stop signaled";
				break;
			}
			request = pendingRequests_.front();
			pendingRequests_.pop();
		}

		/* Deliver the frame */
		LOG(IPU6, Info) << "Worker: delivering frame " << frameSequence_;
		if (dqBuf)
			deliverFrame(request, dqBuf->addr, dqBuf->s.size);
		LOG(IPU6, Info) << "Worker: frame delivered";

		/* Re-queue buffer for next frame */
		LOG(IPU6, Info) << "Worker: qbuf...";
		icamera::camera_buffer_t *bufPtr = &halBuffer_;
		ret = icamera::camera_stream_qbuf(cameraId_, &bufPtr, 1, nullptr);
		if (ret < 0) {
			LOG(IPU6, Error) << "Worker: qbuf failed: " << ret;
			break;
		}
		LOG(IPU6, Info) << "Worker: qbuf ok";
	}

	LOG(IPU6, Info) << "Worker thread exiting";
}

/* --- IPU6CameraConfiguration --- */

IPU6CameraConfiguration::IPU6CameraConfiguration(IPU6CameraData *data)
	: CameraConfiguration(), data_(data)
{
}

CameraConfiguration::Status IPU6CameraConfiguration::validate()
{
	Status status = Valid;

	if (config_.empty())
		return Invalid;

	if (config_.size() > 1) {
		config_.resize(1);
		status = Adjusted;
	}

	if (orientation != Orientation::Rotate0) {
		orientation = Orientation::Rotate0;
		status = Adjusted;
	}

	StreamConfiguration &cfg = config_[0];

	/* We only support NV12 output */
	if (cfg.pixelFormat != formats::NV12) {
		cfg.pixelFormat = formats::NV12;
		status = Adjusted;
	}

	/* Find the best matching size */
	Size requestedSize = cfg.size;
	Size bestSize = kSupportedSizes.front();
	unsigned int bestDist = UINT_MAX;

	for (const auto &s : kSupportedSizes) {
		unsigned int dist = abs(static_cast<int>(s.width) - static_cast<int>(requestedSize.width)) +
				    abs(static_cast<int>(s.height) - static_cast<int>(requestedSize.height));
		if (dist < bestDist) {
			bestDist = dist;
			bestSize = s;
		}
	}

	if (cfg.size != bestSize) {
		LOG(IPU6, Debug) << "Adjusting size from " << cfg.size
				 << " to " << bestSize;
		cfg.size = bestSize;
		status = Adjusted;
	}

	cfg.stride = cfg.size.width;
	cfg.frameSize = cfg.size.width * cfg.size.height * 3 / 2; /* NV12 */
	cfg.bufferCount = 4;
	cfg.colorSpace = ColorSpace::Sycc;

	return status;
}

/* --- PipelineHandlerIPU6 --- */

PipelineHandlerIPU6::PipelineHandlerIPU6(CameraManager *manager)
	: PipelineHandler(manager),
	  dmaBufAllocator_(DmaBufAllocator::DmaBufAllocatorFlag::CmaHeap |
			   DmaBufAllocator::DmaBufAllocatorFlag::SystemHeap |
			   DmaBufAllocator::DmaBufAllocatorFlag::UDmaBuf)
{
}

PipelineHandlerIPU6::~PipelineHandlerIPU6()
{
	if (halInitialized_) {
		icamera::camera_hal_deinit();
		halInitialized_ = false;
	}
}

std::unique_ptr<CameraConfiguration>
PipelineHandlerIPU6::generateConfiguration(Camera *camera,
					   Span<const StreamRole> roles)
{
	IPU6CameraData *data = cameraData(camera);
	auto config = std::make_unique<IPU6CameraConfiguration>(data);

	if (roles.empty())
		return config;

	/* We support a single stream */
	std::map<PixelFormat, std::vector<SizeRange>> formats;
	for (const auto &size : kSupportedSizes)
		formats[formats::NV12].push_back(SizeRange(size));

	StreamFormats streamFormats(formats);
	StreamConfiguration cfg(streamFormats);
	cfg.pixelFormat = formats::NV12;
	cfg.size = kSupportedSizes.front(); /* 1280x720 default */
	cfg.bufferCount = 4;

	config->addConfiguration(cfg);
	config->validate();

	return config;
}

int PipelineHandlerIPU6::configure(Camera *camera, CameraConfiguration *config)
{
	IPU6CameraData *data = cameraData(camera);
	const StreamConfiguration &cfg = config->at(0);

	int ret = data->configureHalStream(cfg.size);
	if (ret < 0)
		return ret;

	const_cast<StreamConfiguration &>(cfg).setStream(&data->stream_);

	return 0;
}

int PipelineHandlerIPU6::exportFrameBuffers(Camera *camera, Stream *stream,
					    std::vector<std::unique_ptr<FrameBuffer>> *buffers)
{
	IPU6CameraData *data = cameraData(camera);
	unsigned int count = stream->configuration().bufferCount;
	const Size &size = data->configuredSize_;

	/* NV12: Y plane + UV plane */
	unsigned int ySize = size.width * size.height;
	unsigned int uvSize = ySize / 2;

	return dmaBufAllocator_.exportBuffers(count, { ySize, uvSize }, buffers);
}

int PipelineHandlerIPU6::start(Camera *camera,
			       [[maybe_unused]] const ControlList *controls)
{
	IPU6CameraData *data = cameraData(camera);
	return data->startStreaming();
}

void PipelineHandlerIPU6::stopDevice(Camera *camera)
{
	IPU6CameraData *data = cameraData(camera);
	data->stopStreaming();
}

int PipelineHandlerIPU6::queueRequestDevice(Camera *camera, Request *request)
{
	IPU6CameraData *data = cameraData(camera);

	FrameBuffer *buffer = request->findBuffer(&data->stream_);
	if (!buffer) {
		LOG(IPU6, Error) << "Request has no buffer for stream";
		return -ENOENT;
	}

	{
		std::lock_guard<std::mutex> lock(data->queueMutex_);
		data->pendingRequests_.push(request);
	}
	data->requestCv_.notify_one();

	return 0;
}

bool PipelineHandlerIPU6::match(DeviceEnumerator *enumerator)
{
	/*
	 * First check if libcamhal can work before acquiring the media device.
	 * If libcamhal can't find cameras (missing config, firmware, etc.),
	 * we return false so the 'simple' pipeline handler can fall back
	 * to software ISP.
	 */
	if (!halInitialized_) {
		int ret = icamera::camera_hal_init();
		if (ret < 0) {
			LOG(IPU6, Debug) << "Failed to init camera HAL, falling back";
			return false;
		}
		halInitialized_ = true;
	}

	int numCameras = icamera::get_number_of_cameras();
	if (numCameras <= 0) {
		LOG(IPU6, Debug) << "libcamhal found no cameras, falling back";
		return false;
	}

	/* Look for Intel IPU6 media device */
	DeviceMatch dm("intel-ipu6");
	std::shared_ptr<MediaDevice> media = acquireMediaDevice(enumerator, dm);
	if (!media) {
		LOG(IPU6, Debug) << "No IPU6 media device found";
		return false;
	}

	if (!dmaBufAllocator_.isValid()) {
		LOG(IPU6, Error) << "DMA buffer allocator not available";
		return false;
	}

	LOG(IPU6, Info) << "Found " << numCameras << " IPU6 camera(s) via libcamhal";

	/* Register only the first camera (real sensor).
	 * The HAL may report additional "ghost" cameras (e.g., AR0234 USB
	 * devices) that are not physically present. Only camera 0 (the
	 * built-in OV01A10) is real on this platform.
	 * TODO: Detect real cameras more robustly.
	 */
	int registerCount = std::min(numCameras, 1);
	for (int i = 0; i < registerCount; i++) {
		auto data = std::make_unique<IPU6CameraData>(this);
		int ret = data->init(i);
		if (ret < 0) {
			LOG(IPU6, Error) << "Failed to init camera " << i;
			continue;
		}

		std::string id = "ipu6-" + data->sensorName_ + "-" + std::to_string(i);
		std::set<Stream *> streams{ &data->stream_ };
		auto camera = Camera::create(std::move(data), id, streams);
		registerCamera(std::move(camera));
	}

	return true;
}

REGISTER_PIPELINE_HANDLER(PipelineHandlerIPU6, "ipu6")

} /* namespace libcamera */
