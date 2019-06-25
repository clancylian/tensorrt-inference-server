#include "myplugin.h"
#include <assert.h>
#include <cuda_runtime_api.h>
#include <string.h>
#include "src/core/logging.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace std;

void calcPReLU(const float *input, float *output, const float* weights, int batchSize, int channels,
               int width, int height, cudaStream_t stream);

//构造函数：从caffe模型文件解析参数
PReLULayer::PReLULayer(const Weights *weights, int nbWeights)
{
    // since we want to deal with the case where there is no bias, we can't infer
    // the number of channels from the bias weights.
    assert(nbWeights == 1);
    mPReLuWeights = copyToDevice(weights[0].values, weights[0].count);
}

//构造函数：从TensorRT模型文件解析参数
PReLULayer::PReLULayer(const void* data, size_t length)
{
    const char* d = reinterpret_cast<const char*>(data), *a = d;
    width = read<int>(d);
    height = read<int>(d);
    channel = read<int>(d);
    mPReLuWeights = deserializeToDevice(d, channel);
    assert(d == a + length);
}

PReLULayer::~PReLULayer()
{
    cudaFree(const_cast<void*>(mPReLuWeights.values));
    cudaFree(deviceData);
}

//获取输出个数
int PReLULayer::getNbOutputs() const
{
    return 1;
}

Dims PReLULayer::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
    width = inputs[0].d[2];
    height = inputs[0].d[1];
    channel = inputs[0].d[0];
    //PReLu output dims the same as input dims
    return DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
}

bool PReLULayer::supportsFormat(nvinfer1::DataType type, PluginFormat format) const
{
    return (type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kHALF)
            && format == PluginFormat::kNCHW;
}

void PReLULayer::configureWithFormat(const Dims* inputDims, int nbInputs,
                                     const Dims* outputDims, int nbOutputs,
                                     nvinfer1::DataType type, PluginFormat format,
                                     int maxBatchSize)
{
    assert((type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kHALF)
           && format == PluginFormat::kNCHW);
    mDataType = type;
}

int PReLULayer::initialize()
{
    return 0;
}

void PReLULayer::terminate()
{
}

size_t PReLULayer::getWorkspaceSize(int) const
{
    return 0;
}

//推理函数调用CUDA函数进行推理
int PReLULayer::enqueue(int batchSize, const void*const * inputs, void** outputs, void*,
                        cudaStream_t stream)
{
    calcPReLU(reinterpret_cast<const float *>(inputs[0]), (float*)outputs[0],
            reinterpret_cast<const float*>(mPReLuWeights.values),
            batchSize, mPReLuWeights.count, width, height, stream);
    return 0;
}

//获取序列化的大小
size_t PReLULayer::getSerializationSize()
{
    return sizeof(int) * 3 + mPReLuWeights.count * sizeof(float);
}

//序列化操作
void PReLULayer::serialize(void* buffer)
{
    char* d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, width);
    write(d, height);
    write(d, channel);
    serializeFromDevice(d, mPReLuWeights);
    assert(d == a + getSerializationSize());
}

template<typename T> void PReLULayer::write(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

template<typename T> T PReLULayer::read(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

Weights PReLULayer::copyToDevice(const void* hostData, size_t count)
{
    (cudaMalloc(&deviceData, count * sizeof(float)));
    (cudaMemcpy(deviceData, hostData, count * sizeof(float), cudaMemcpyHostToDevice));
    return Weights{ nvinfer1::DataType::kFLOAT, deviceData, int64_t(count) };
}

void PReLULayer::serializeFromDevice(char*& hostBuffer, Weights deviceWeights)
{
    cudaMemcpy(hostBuffer, deviceWeights.values, deviceWeights.count * sizeof(float),
               cudaMemcpyDeviceToHost);
    hostBuffer += deviceWeights.count * sizeof(float);
}

Weights PReLULayer::deserializeToDevice(const char*& hostBuffer, size_t count)
{
    Weights w = copyToDevice(hostBuffer, count);
    hostBuffer += count * sizeof(float);
    return w;
}

bool AcrFacePluginFactory::isPlugin(const char *name)
{
    return isPluginExt(name);
}

bool AcrFacePluginFactory::isPluginExt(const char* name)
{
    if (strcmp(name, "relu0") == 0) {
        return true;
    }

    string layerName(name);
    string suffix("_relu1");
    size_t pos = layerName.find(suffix);
    if (pos == string::npos) {
        return false;
    }

    int stage, unit;
    int num = sscanf(name, "stage%d_unit%d", &stage, &unit);
    if (num != 2 || unit < 1) {
        return false;
    }

    if ((stage == 1 && unit <= Stage1PReLUNum) || (stage == 2 && unit <= Stage2PReLUNum)
            || (stage == 3 && unit <= Stage3PReLUNum) || (stage == 4 && unit <= Stage4PReLUNum)) {
        return true;
    }

    return false;
}

IPlugin *AcrFacePluginFactory::createPlugin(const char* layerName, const Weights* weights,
                                            int nbWeights)
{
    LOG_INFO << "Create TensorRT plugin layer " << layerName << ".";
    // there's no way to pass parameters through from the model definition, so we have to define it here explicitly
    assert(isPlugin(layerName) && nbWeights == 1 && weights[0].type == nvinfer1::DataType::kFLOAT);
    if (strcmp(layerName, "relu0") == 0) {
        assert(prelu0.get() == nullptr);
        prelu0 = std::unique_ptr<PReLULayer>(new PReLULayer(weights, nbWeights));
        return prelu0.get();
    }

    int stage, unit;
    sscanf(layerName, "stage%d_unit%d_relu1", &stage, &unit);
    std::unique_ptr<PReLULayer> *plugin;
    if (stage == 1) {
        plugin = &stage1[unit - 1];
    } else if (stage == 2) {
        plugin = &stage2[unit - 1];
    } else if (stage == 3) {
        plugin = &stage3[unit - 1];
    } else if (stage == 4) {
        plugin = &stage4[unit - 1];
    } else {
        assert(true);
    }

    assert(plugin->get() == nullptr);
    *plugin = std::unique_ptr<PReLULayer>(new PReLULayer(weights, nbWeights));
    return plugin->get();
}

// deserialization plugin implementation
IPlugin *AcrFacePluginFactory::createPlugin(const char* layerName, const void* serialData,
                                            size_t serialLength)
{
    assert(isPlugin(layerName));
    if (strcmp(layerName, "relu0") == 0) {
        assert(prelu0.get() == nullptr);
        prelu0 = std::unique_ptr<PReLULayer>(new PReLULayer(serialData, serialLength));
        return prelu0.get();
    }

    int stage, unit;
    sscanf(layerName, "stage%d_unit%d_relu1", &stage, &unit);
    std::unique_ptr<PReLULayer> *plugin;
    if (stage == 1) {
        plugin = &stage1[unit - 1];
    } else if (stage == 2) {
        plugin = &stage2[unit - 1];
    } else if (stage == 3) {
        plugin = &stage3[unit - 1];
    } else if (stage == 4) {
        plugin = &stage4[unit - 1];
    } else {
        assert(true);
    }

    assert(plugin->get() == nullptr);
    *plugin = std::unique_ptr<PReLULayer>(new PReLULayer(serialData, serialLength));
    return plugin->get();
}

// the application has to destroy the plugin when it knows it's safe to do so
void AcrFacePluginFactory::destroyPlugin()
{
    prelu0.release();
    for (int i = 0; i < Stage1PReLUNum; i++) {
        stage1[i].release();
    }
    for (int i = 0; i < Stage2PReLUNum; i++) {
        stage2[i].release();
    }
    for (int i = 0; i < Stage3PReLUNum; i++) {
        stage3[i].release();
    }
    for (int i = 0; i < Stage4PReLUNum; i++) {
        stage4[i].release();
    }
}
