#ifndef ACSCS_H
#define ACSCS_H

#include <NvInfer.h>
#include <NvCaffeParser.h>
#include <memory>

#define Stage1PReLUNum    3
#define Stage2PReLUNum    13
#define Stage3PReLUNum    30
#define Stage4PReLUNum    3

class PReLULayer : public nvinfer1::IPluginExt
{
public:
    //构造函数：从caffe模型文件解析参数
    PReLULayer(const nvinfer1::Weights *weights, int nbWeights);

    //构造函数：从TensorRT模型文件解析参数
    PReLULayer(const void* data, size_t length);

    ~PReLULayer();

    //获取输出个数
    int getNbOutputs() const override;

    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) override;

    bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const override;

    void configureWithFormat(const nvinfer1::Dims* inputDims, int nbInputs,
                             const nvinfer1::Dims* outputDims, int nbOutputs,
                             nvinfer1::DataType type, nvinfer1::PluginFormat format,
                             int maxBatchSize) override;

    int initialize() override;

    virtual void terminate() override;

    virtual size_t getWorkspaceSize(int) const override;

    //推理函数调用CUDA函数进行推理
    virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void*,
                        cudaStream_t stream) override;

    //获取序列化的大小
    virtual size_t getSerializationSize() override;

    //序列化操作
    virtual void serialize(void* buffer) override;

private:
    template<typename T> void write(char*& buffer, const T& val);
    template<typename T> T read(const char*& buffer);
    nvinfer1::Weights copyToDevice(const void* hostData, size_t count);
    void serializeFromDevice(char*& hostBuffer, nvinfer1::Weights deviceWeights);
    nvinfer1::Weights deserializeToDevice(const char*& hostBuffer, size_t count);

    void* deviceData;
    int width, height, channel;
    nvinfer1::Weights mPReLuWeights;
    nvinfer1::DataType mDataType{nvinfer1::DataType::kFLOAT};
};


class AcrFacePluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactoryExt
{
public:
    //解析名字是否为插件
    bool isPlugin(const char* name) override;

    bool isPluginExt(const char* name) override;

    virtual nvinfer1::IPlugin *createPlugin(const char* layerName, const nvinfer1::Weights* weights,
                                            int nbWeights) override;

    // deserialization plugin implementation
    nvinfer1::IPlugin *createPlugin(const char* layerName, const void* serialData, size_t serialLength) override;

    // the application has to destroy the plugin when it knows it's safe to do so
    void destroyPlugin();

private:
    std::unique_ptr<PReLULayer> prelu0 = { nullptr };
    std::unique_ptr<PReLULayer> stage1[Stage1PReLUNum] = { nullptr };
    std::unique_ptr<PReLULayer> stage2[Stage2PReLUNum] = { nullptr };
    std::unique_ptr<PReLULayer> stage3[Stage3PReLUNum] = { nullptr };
    std::unique_ptr<PReLULayer> stage4[Stage4PReLUNum] = { nullptr };
};

#endif // ACSCS_H
