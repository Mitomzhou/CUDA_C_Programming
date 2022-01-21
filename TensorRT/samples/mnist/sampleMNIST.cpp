/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// \file sampleMNIST.cpp
// \brief This file contains the implementation of the MNIST sample.
//
// It builds a TensorRT engine by importing a trained MNIST Caffe model. It uses the engine to run
// inference on an input image of a digit.
// It can be run with the following command line:
// Command: ./sample_mnist [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "NvCaffeParser.h"
#include "NvInfer.h"

#include <cassert>
#include <cmath>
#include <iostream>

const std::string gSampleName = "TensorRT.sample_mnist";

/**
 * The SampleMNIST class implements the MNIST sample.
 * It creates the network using a trained Caffe MNIST classification model
 */
class SampleMNIST
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    SampleMNIST(const samplesCommon::CaffeSampleParams& params): mParams(params){ }

    bool build();
    bool infer();
    bool teardown();

private:
    void constructNetwork(SampleUniquePtr<nvcaffeparser1::ICaffeParser>& parser, SampleUniquePtr<nvinfer1::INetworkDefinition>& network);
    bool processInput(const samplesCommon::BufferManager& buffers, const std::string& inputTensorName, int inputFileIdx) const;
    bool verifyOutput(const samplesCommon::BufferManager& buffers, const std::string& outputTensorName, int groundTruthDigit) const;

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr}; //< The TensorRT engine used to run the network
    samplesCommon::CaffeSampleParams mParams; //< The parameters for the sample.
    nvinfer1::Dims mInputDims; //< The dimensions of the input to the network.
    SampleUniquePtr<nvcaffeparser1::IBinaryProtoBlob> mMeanBlob; // the mean blob, which we need to keep around until build is done
};

/**
 * 通过解析caffe模型创建MNIST网络，并构建用于运行MNIST（mEngine）的引擎
 * @return
 */
bool SampleMNIST::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork());
    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    auto parser = SampleUniquePtr<nvcaffeparser1::ICaffeParser>(nvcaffeparser1::createCaffeParser());

    // 使用caffe解析器创建MNIST网络并标记输出层
    constructNetwork(parser, network);

    // batch大小
    builder->setMaxBatchSize(mParams.batchSize);
    // 最大显存
    config->setMaxWorkspaceSize(16_MiB);
    config->setFlag(BuilderFlag::kGPU_FALLBACK);
    config->setFlag(BuilderFlag::kSTRICT_TYPES);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
    }

    // LDA核心 NVIDIA硬件的一个平台,基本用不到
    // samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    // 返回一个初始化好的cuda推理引擎
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());

    if (!mEngine)
        return false;

    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    assert(mInputDims.nbDims == 3);

    return true;
}

/**
 * Reads the input and mean data, preprocesses, and stores the result in a managed buffer
 * @param buffers
 * @param inputTensorName
 * @param inputFileIdx
 * @return
 */
bool SampleMNIST::processInput(
        const samplesCommon::BufferManager& buffers, const std::string& inputTensorName, int inputFileIdx) const
{
    const int inputH = mInputDims.d[1];
    const int inputW = mInputDims.d[2];

    // Read a random digit file
    srand(unsigned(time(nullptr)));
    std::vector<uint8_t> fileData(inputH * inputW);
    readPGMFile(locateFile(std::to_string(inputFileIdx) + ".pgm", mParams.dataDirs), fileData.data(), inputH, inputW);

    // Print ASCII representation of digit
    gLogInfo << "Input:\n";
    for (int i = 0; i < inputH * inputW; i++)
    {
        gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");
    }
    gLogInfo << std::endl;

    float* hostInputBuffer = static_cast<float*>(buffers.getHostBuffer(inputTensorName));

    for (int i = 0; i < inputH * inputW; i++)
    {
        hostInputBuffer[i] = float(fileData[i]);
    }

    return true;
}

/**
 * Verifies that the output is correct and prints it
 * @param buffers
 * @param outputTensorName
 * @param groundTruthDigit
 * @return
 */
bool SampleMNIST::verifyOutput(
        const samplesCommon::BufferManager& buffers, const std::string& outputTensorName, int groundTruthDigit) const
{
    const float* prob = static_cast<const float*>(buffers.getHostBuffer(outputTensorName));

    // Print histogram of the output distribution
    gLogInfo << "Output:\n";
    float val{0.0f};
    int idx{0};
    const int kDIGITS = 10;

    for (int i = 0; i < kDIGITS; i++)
    {
        if (val < prob[i])
        {
            val = prob[i];
            idx = i;
        }

        gLogInfo << i << ": " << std::string(int(std::floor(prob[i] * 10 + 0.5f)), '*') << "\n";
    }
    gLogInfo << std::endl;

    return (idx == groundTruthDigit && val > 0.9f);
}

/**
 * 构建网络
 * @param parser
 * @param network
 */
void SampleMNIST::constructNetwork(
        SampleUniquePtr<nvcaffeparser1::ICaffeParser>& parser, SampleUniquePtr<nvinfer1::INetworkDefinition>& network)
{
    const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor = parser->parse(
            mParams.prototxtFileName.c_str(), mParams.weightsFileName.c_str(), *network, nvinfer1::DataType::kFLOAT);

    for (auto& s : mParams.outputTensorNames)
    {
        network->markOutput(*blobNameToTensor->find(s.c_str()));
    }

    // add mean subtraction to the beginning of the network
    nvinfer1::Dims inputDims = network->getInput(0)->getDimensions();
    // 读取均值文件数据
    mMeanBlob = SampleUniquePtr<nvcaffeparser1::IBinaryProtoBlob>(parser->parseBinaryProto(mParams.meanFileName.c_str()));
    nvinfer1::Weights meanWeights{nvinfer1::DataType::kFLOAT, mMeanBlob->getData(), inputDims.d[1] * inputDims.d[2]};
    // 数据的原始分布是[0,256], 减去均值后分布[-127, 127]
    float maxMean = samplesCommon::getMaxValue(static_cast<const float*>(meanWeights.values), samplesCommon::volume(inputDims));

    auto mean = network->addConstant(nvinfer1::Dims3(1, inputDims.d[1], inputDims.d[2]), meanWeights);
    mean->getOutput(0)->setDynamicRange(-maxMean, maxMean);
    network->getInput(0)->setDynamicRange(-maxMean, maxMean);
    // 减均值操作
    auto meanSub = network->addElementWise(*network->getInput(0), *mean->getOutput(0), ElementWiseOperation::kSUB);
    meanSub->getOutput(0)->setDynamicRange(-maxMean, maxMean);
    network->getLayer(0)->setInput(0, *meanSub->getOutput(0));
    // 缩放,输出[-1,1]
    samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
}

/**
 * 推理：配置缓冲区,设置输入,执行推理引擎并验证输出
 * @return
 */
bool SampleMNIST::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Pick a random digit to try to infer
    srand(time(NULL));
    const int digit = rand() % 10;

    // Read the input data into the managed buffers
    // There should be just 1 input tensor
    assert(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers, mParams.inputTensorNames[0], digit))
    {
        return false;
    }
    // Create CUDA stream for the execution of this inference.
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Asynchronously copy data from host input buffers to device input buffers
    buffers.copyInputToDeviceAsync(stream);

    // Asynchronously enqueue the inference work
    if (!context->enqueue(mParams.batchSize, buffers.getDeviceBindings().data(), stream, nullptr))
    {
        return false;
    }
    // Asynchronously copy data from device output buffers to host output buffers
    buffers.copyOutputToHostAsync(stream);

    // Wait for the work in the stream to complete
    cudaStreamSynchronize(stream);

    // Release stream
    cudaStreamDestroy(stream);

    // Check and print the output of the inference
    // There should be just one output tensor
    assert(mParams.outputTensorNames.size() == 1);
    bool outputCorrect = verifyOutput(buffers, mParams.outputTensorNames[0], digit);

    return outputCorrect;
}

/**
 * Used to clean up any state created in the sample class
 * @return
 */
bool SampleMNIST::teardown()
{
    // Clean up the libprotobuf files as the parsing is complete
    // note It is not safe to use any other part of the protocol buffers library after
    // ShutdownProtobufLibrary() has been called.
    nvcaffeparser1::shutdownProtobufLibrary();
    return true;
}

/**
 * 命令行参数初始化params结构成员
 * @param args
 * @return
 */
samplesCommon::CaffeSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
    samplesCommon::CaffeSampleParams params;
    // 默认路径
    if (args.dataDirs.empty())
    {
        params.dataDirs.push_back("data/mnist/");
        params.dataDirs.push_back("data/samples/mnist/");
    }
    else
    {
        params.dataDirs = args.dataDirs;
    }

    // 配置文件
    params.prototxtFileName = locateFile("mnist.prototxt", params.dataDirs);
    params.weightsFileName = locateFile("mnist.caffemodel", params.dataDirs);
    params.meanFileName = locateFile("mnist_mean.binaryproto", params.dataDirs); // 网络均值文件
    params.inputTensorNames.push_back("data"); // 输入Tensor
    params.batchSize = 1;
    params.outputTensorNames.push_back("prob"); // 输出Tensor
    params.dlaCore = args.useDLACore; // 是否使用DLA核心
    params.int8 = args.runInInt8;     // 以INT8的方式运行
    params.fp16 = args.runInFp16;     // 以FP16的方式运行

    return params;
}


int main(int argc, char** argv)
{
    samplesCommon::Args args;

    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);

    // 1.命令行参数初始化params结构成员
    samplesCommon::CaffeSampleParams params = initializeSampleParams(args);
    SampleMNIST sample(params);

    // 2.解析caffe模型创建MNIST网络，并构建用于运行MNINST（mEngine）的引擎
    sample.build();
    // 3.推理
    sample.infer();
    // 4.清除状态，释放内存
    sample.teardown();

    return gLogger.reportPass(sampleTest);
}
