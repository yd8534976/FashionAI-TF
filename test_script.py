from fashion_stacked import FashionAI
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    fashionAI = FashionAI()
    fashionAI.test(model_dir='model/fashion_stacked/fashionai.ckpt',
                   output_dir="outputs/results_test_bb.csv")
