from fashion_stacked import FashionAI
import datetime
import os
import argparse


def main(a):
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    i = datetime.datetime.now()
    output_dir = "outputs/yrd{:02}{:02}_{}_{}.csv".format(i.month, i.day, a.id, a.data_dir)

    fashionAI = FashionAI()
    fashionAI.test(model_dir=a.model_dir,
                   data_dir=a.data_dir,
                   output_dir=output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='test_b')
    parser.add_argument("--model_dir", default='model/ai0708_cpn/fashionai.ckpt')
    parser.add_argument("--id", default="stack")
    parser.add_argument("--gpu", default='0')
    a = parser.parse_args()
    main(a)
