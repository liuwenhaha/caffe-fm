#include <vector>

#include "caffe/layers/normalized_sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
__global__ void FilterIgnoreLabel(const int n, Dtype* diff, const Dtype* target) {
  CUDA_KERNEL_LOOP(index, n) {
    diff[index] = (target[index] == -1) ? 0 : diff[index];
  }
}

template <typename Dtype>
void NormalizedSigmoidCrossEntropyLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) { LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs."; }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const int unit_count = count / num;

    const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy(count, sigmoid_output_data, bottom_diff);
    caffe_gpu_axpy(count, Dtype(-1), target, bottom_diff);
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    FilterIgnoreLabel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_diff, target);
    caffe_gpu_scal(count, loss_weight / unit_count, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_BACKWARD(NormalizedSigmoidCrossEntropyLossLayer);


}  // namespace caffe
