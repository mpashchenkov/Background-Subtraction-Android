#include <jni.h>
#include <string>
#include <vector>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include <tensorflow/lite/version.h>
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"

// #include <android/log.h>
// #define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, "gandoulf", __VA_ARGS__))
// #define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, "gandoulf", __VA_ARGS__))

namespace tflite_operations {
namespace {

constexpr int kWeightsTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kDataInputTensor = 0;
constexpr int kOutputTensor = 0;

// These functions were copied from the following places:
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/reference/reference_ops.h
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/transpose_conv.cc

inline void TransposeConvBias(
    const ::tflite::ConvParams& params,
    const ::tflite::RuntimeShape& input_shape, const float* input_data,
    const ::tflite::RuntimeShape& filter_shape, const float* filter_data,
    const ::tflite::RuntimeShape& bias_shape, const float* bias_data,
    const ::tflite::RuntimeShape& output_shape, float* output_data,
    const ::tflite::RuntimeShape& im2col_shape, float* im2col_data) {
  // Start of copy from
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/reference/reference_ops.h
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;

  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(bias_shape.DimensionsCount(), 1);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  (void)im2col_data;   // only used in optimized code.
  (void)im2col_shape;  // only used in optimized code.

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);


  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; out_y++) {
      for (int out_x = 0; out_x < output_width; out_x++) {
        for (int out_channel = 0; out_channel < output_depth; out_channel++) {
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              bias_data[out_channel];
        }
      }
    }

    for (int in_y = 0; in_y < input_height; ++in_y) {
      for (int in_x = 0; in_x < input_width; ++in_x) {
        for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
          // Loop through the output elements it will influence
          const int out_x_origin = (in_x * stride_width) - pad_width;
          const int out_y_origin = (in_y * stride_height) - pad_height;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              for (int out_channel = 0; out_channel < output_depth;
                   ++out_channel) {
                // Compute output element location
                const int out_x = out_x_origin + filter_x;
                const int out_y = out_y_origin + filter_y;
                // We cannot accumulate out of bounds
                if ((out_x >= 0) && (out_x < output_width) && (out_y >= 0) &&
                    (out_y < output_height)) {
                  float input_value = input_data[Offset(
                      input_shape, batch, in_y, in_x, in_channel)];
                  float filter_value =
                      filter_data[Offset(filter_shape, out_channel, filter_y,
                                         filter_x, in_channel)];
                  output_data[Offset(output_shape, batch, out_y, out_x,
                                     out_channel)] +=
                      input_value * filter_value;
                }
              }
            }
          }
        }
      }
    }
  }
  // End of copy.
}

// Start of copy from
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/transpose_conv.cc
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, ::tflite::NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, ::tflite::NumOutputs(node), 1);

  const TfLiteTensor* weights =
      ::tflite::GetInput(context, node, kWeightsTensor);
  TF_LITE_ENSURE(context, weights != nullptr);
  const TfLiteTensor* bias = ::tflite::GetInput(context, node, kBiasTensor);
  TF_LITE_ENSURE(context, bias != nullptr);
  const TfLiteTensor* input =
      ::tflite::GetInput(context, node, kDataInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output = ::tflite::GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_EQ(context, ::tflite::NumDimensions(input), 4);
  TF_LITE_ENSURE_EQ(context, ::tflite::NumDimensions(weights), 4);
  TF_LITE_ENSURE_EQ(context, ::tflite::NumDimensions(bias), 1);

  TF_LITE_ENSURE_EQ(context, ::tflite::SizeOfDimension(weights, 0),
                    ::tflite::SizeOfDimension(bias, 0));

  // Currently only supports float32.
  const TfLiteType data_type = input->type;
  TF_LITE_ENSURE(context, data_type == kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, output->type, data_type);
  TF_LITE_ENSURE_EQ(context, weights->type, data_type);
  TF_LITE_ENSURE_EQ(context, bias->type, data_type);

  // Ensure that weights and inputs have the same channel dimension.
  // Note: TOCO will reorder weights in the following format: OHWI.
  TF_LITE_ENSURE_EQ(context, ::tflite::SizeOfDimension(input, 3),
                    ::tflite::SizeOfDimension(weights, 3));

  // Ensure that weights and bias have the same output channel dimension.
  TF_LITE_ENSURE_EQ(context, ::tflite::SizeOfDimension(weights, 0),
                    ::tflite::SizeOfDimension(bias, 0));

  const auto* params = reinterpret_cast<const TfLiteTransposeConvParams*>(
      node->custom_initial_data);
  const int filter_width = ::tflite::SizeOfDimension(weights, 2);
  const int filter_height = ::tflite::SizeOfDimension(weights, 1);
  const int stride_width = params->stride_width;
  const int stride_height = params->stride_height;
  const int in_width = ::tflite::SizeOfDimension(input, 2);
  const int in_height = ::tflite::SizeOfDimension(input, 1);

  // Get height and width of the output image.
  TfLiteIntArray* output_shape_array = TfLiteIntArrayCreate(4);
  output_shape_array->data[0] = ::tflite::SizeOfDimension(input, 0);
  output_shape_array->data[3] = ::tflite::SizeOfDimension(weights, 0);

  TfLitePaddingValues padding_size{0, 0};
  if (params->padding == kTfLitePaddingSame) {
    padding_size.height =
        std::max(0, filter_height - (in_height - 1) % stride_height - 1);
    padding_size.width =
        std::max(0, filter_width - (in_width - 1) % stride_width - 1);
  }
  output_shape_array->data[1] =
      stride_height * (in_height - 1) + filter_height - padding_size.height;
  output_shape_array->data[2] =
      stride_width * (in_width - 1) + filter_width - padding_size.width;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, output, output_shape_array));
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* weights =
      ::tflite::GetInput(context, node, kWeightsTensor);
  TF_LITE_ENSURE(context, weights != nullptr);
  const TfLiteTensor* bias = ::tflite::GetInput(context, node, kBiasTensor);
  TF_LITE_ENSURE(context, bias != nullptr);
  const TfLiteTensor* input =
      ::tflite::GetInput(context, node, kDataInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output = ::tflite::GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  const auto* params = reinterpret_cast<const TfLiteTransposeConvParams*>(
      node->custom_initial_data);

  const int filter_width = ::tflite::SizeOfDimension(weights, 2);
  const int filter_height = ::tflite::SizeOfDimension(weights, 1);
  const int stride_width = params->stride_width;
  const int stride_height = params->stride_height;
  const int in_width = ::tflite::SizeOfDimension(input, 2);
  const int in_height = ::tflite::SizeOfDimension(input, 1);

  TfLitePaddingValues padding_size{0, 0};
  if (params->padding == kTfLitePaddingSame) {
    padding_size.height =
        std::max(0, filter_height - (in_height - 1) % stride_height - 1);
    padding_size.width =
        std::max(0, filter_width - (in_width - 1) % stride_width - 1);
  }

  // Currently only support float32.
  switch (input->type) {
    case kTfLiteFloat32: {
      ::tflite::ConvParams op_params;
      op_params.padding_type = ::tflite::PaddingType::kSame;
      op_params.padding_values.width = padding_size.width / 2;
      op_params.padding_values.height = padding_size.height / 2;
      op_params.stride_width = stride_width;
      op_params.stride_height = stride_height;

      TransposeConvBias(
          op_params, ::tflite::GetTensorShape(input),
          ::tflite::GetTensorData<float>(input),
          ::tflite::GetTensorShape(weights),
          ::tflite::GetTensorData<float>(weights),
          ::tflite::GetTensorShape(bias), ::tflite::GetTensorData<float>(bias),
          ::tflite::GetTensorShape(output),
          ::tflite::GetTensorData<float>(output),
          // Last two args specify im2col which reference_ops ignores.
          // (Note this does not lead to a performance regression, as the
          // previous optimized version was just a copy of the reference code.)
          // TODO: Allocate im2col tensors and switch to
          // optimized_ops.
          ::tflite::GetTensorShape(output),
          ::tflite::GetTensorData<float>(output));
      break;
    }
    default:
      context->ReportError(context, "Type %d, not currently supported.",
                           input->type);
      return kTfLiteError;
  }

  return kTfLiteOk;
}
// End of copy.
}  // namespace

TfLiteRegistration* RegisterConvolution2DTransposeBias() {
    static TfLiteRegistration reg = {nullptr, nullptr, Prepare, Eval};
    return &reg;
}
} // namespace tflite_operations

//------------------------------------------------------------------------------

class State { // a. k. a. Model
public:
    State(const std::string& modelPath) {
        std::unique_ptr<tflite::FlatBufferModel> model =
            tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());

        tflite::ops::builtin::BuiltinOpResolver resolver;
        resolver.AddCustom("Convolution2DTransposeBias",
                           tflite_operations::RegisterConvolution2DTransposeBias());

        tflite::InterpreterBuilder builder(*model, resolver);

        builder(&interpreter);

        /*auto gpuOptions = TfLiteGpuDelegateOptionsV2Default();
        gpuOptions.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
        gpuOptions.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
        TfLiteDelegate* gpuDelegate = TfLiteGpuDelegateV2Create(&gpuOptions);
        if (interpreter->ModifyGraphWithDelegate(gpuDelegate) != kTfLiteOk) {
            throw std::runtime_error("Failed to apply GPU delegate.");
        }*/

        /*auto xnnpackOptions = TfLiteXNNPackDelegateOptionsDefault();
        xnnpackOptions.num_threads = 4;
        TfLiteDelegate* xnnpackDelegate = TfLiteXNNPackDelegateCreate(&xnnpackOptions);
        if (interpreter->ModifyGraphWithDelegate(xnnpackDelegate) != kTfLiteOk) {
            throw std::runtime_error("Failed to apply XNNPack delegate.");
        }*/

        interpreter->AllocateTensors();

        std::unique_ptr<TfLiteTensor> in_tensor;
        const std::vector<int> inputs = interpreter->inputs();
        in_tensor = std::make_unique<TfLiteTensor>(*interpreter->tensor(inputs.front()));
        input_dims = in_tensor->dims;
        int input_dims_size = in_tensor->dims->size;

        std::unique_ptr<TfLiteTensor> out_tensor;
        const std::vector<int> outputs = interpreter->outputs();
        out_tensor = std::make_unique<TfLiteTensor>(*interpreter->tensor(outputs.front()));
        output_dims = out_tensor->dims;
        int output_dims_size = out_tensor->dims->size;

        default_color = cv::Scalar(155, 255, 120);
    }

    std::unique_ptr<tflite::Interpreter> interpreter;
    TfLiteIntArray* input_dims;
    TfLiteIntArray* output_dims;
    cv::Scalar default_color;
};

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_bs_1sample_MainActivity_InitializeTFLite(JNIEnv *env, jobject, jstring str) {

    const char *utf_str = env->GetStringUTFChars(str, 0);
        std::string path(utf_str ? utf_str : "");
    State* tfliteInit = new State(path);

    return (jlong)tfliteInit;
}

/*extern "C" JNIEXPORT jintArray JNICALL
Java_com_example_bs_1sample_MainActivity_GetTensorSize(JNIEnv *env, jobject , jlong interpreterPTR) {
    auto state = ((State*)interpreterPTR);

    std::unique_ptr<TfLiteTensor> in_tensor;
    const std::vector<int> inputs = (state->interpreter)->inputs();
    TfLiteIntArray* input_dims;
    in_tensor = std::make_unique<TfLiteTensor>(*(state->interpreter)->tensor(inputs.front()));
    input_dims = in_tensor->dims;
    int input_dims_size = in_tensor->dims->size;


    std::unique_ptr<TfLiteTensor> out_tensor;
    const std::vector<int> outputs = state->interpreter->outputs();
    TfLiteIntArray* output_dims;
    out_tensor = std::make_unique<TfLiteTensor>(*(state->interpreter)->tensor(outputs.front()));
    output_dims = out_tensor->dims;
    int output_dims_size = out_tensor->dims->size;

    jintArray out_ints;

    out_ints = env->NewIntArray(input_dims_size + output_dims_size);
    env->SetIntArrayRegion(out_ints, 0, 2, &(input_dims->data[1])); // NHWC
    env->SetIntArrayRegion(out_ints, 3, 5, &(output_dims->data[1])); // HWC HWC
    // returns data_ints as a jintArray
    return out_ints;
}*/

extern "C" JNIEXPORT void JNICALL
Java_com_example_bs_1sample_MainActivity_infer(JNIEnv *env, jobject thiz, jlong inputaddr,
                                               jlong resultaddr, jlong interpreterPTR) {
    auto state = ((State*)interpreterPTR);
    cv::Mat& input = *(cv::Mat*)inputaddr;
    cv::Mat& output_res = *(cv::Mat*)resultaddr;
    // int* io_dims = (int*)arr;

    // output_res += input;
    // (state->interpreter)
    cv::Mat rsz;
    const int net_channels = state->input_dims->data[3]; // C
    const int net_in_width = state->input_dims->data[2]; // W
    const int net_in_heght = state->input_dims->data[1]; // H
    float* input_tensor = (state->interpreter)->typed_input_tensor<float>(0);

    cv::cvtColor (input, input, cv::COLOR_RGBA2BGR);

    cv::Size in_size = input.size();

    cv::Mat cvt(cv::Size(net_in_width, net_in_heght), CV_32FC3, input_tensor);
    cv::resize(input, rsz, cv::Size(net_in_width, net_in_heght));
    rsz.convertTo(cvt, CV_32F, 1 / 255.f);
    cv::cvtColor(cvt, cvt, cv::COLOR_BGR2RGB);

    // float* cvt_ptr = cvt.ptr<float>();

    // for (int32_t i = 0; i < net_in_width * net_in_heght; i++) {
    //     for (int32_t c = 0; c < net_channels; c++) {
    //         input_tensor[i * net_channels + c] = (cvt_ptr[i * net_channels + c]);
    //     }
    // }

    (state->interpreter)->Invoke();
    float* output = (state->interpreter)->typed_output_tensor<float>(0);

        const int net_out_width = state->output_dims->data[2]; // W
        const int net_out_heght = state->output_dims->data[1]; // H
        cv::Mat mask(std::vector<int>{net_out_width, net_out_heght}, CV_32F, output);

        mask.convertTo(mask, CV_8U);

        cv::Mat rsz_mask, mask3ch, rsz_mask3ch, contour_mask, res, blured_res;
        auto to3ch = [](const cv::Mat& src, cv::Mat& dst) {
                        auto channels = std::vector<cv::Mat>{src, src, src};
                        cv::merge(channels, dst);
                     };

        cv::resize(mask * 255, rsz_mask, in_size);

        to3ch(rsz_mask, mask3ch);

        cv::Mat background(in_size, CV_8UC3, state->default_color);
        res = (background & ~mask3ch) + (input & mask3ch);

        const float contour_width = in_size.width * 0.008;
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(rsz_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::Mat contour(in_size, CV_8U, cv::Scalar(0, 0, 0));
        cv::drawContours(contour, contours, -1, cv::Scalar(255, 255, 255), contour_width);
        to3ch(contour, contour_mask);

        cv::blur(res, blured_res, cv::Size(10, 10));
        auto blured_contour = blured_res & contour_mask;
        cv::drawContours(res, contours, -1, cv::Scalar(0, 0, 0), contour_width);

        cv::Mat tmp = res + blured_contour;
        cv::cvtColor(tmp, output_res, cv::COLOR_BGR2RGBA);
}