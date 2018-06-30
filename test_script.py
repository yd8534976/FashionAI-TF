from fashion_stacked import FashionAI
import datetime
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if __name__ == '__main__':
    fashionAI = FashionAI()

    i = datetime.datetime.now()
    output_dir = "outputs/yrd{:02}{:02}_final.csv".format(i.month, i.day)
    fashionAI.test(model_dir='model/ai0528_stack/fashionai.ckpt',
                   data_dir='final',
                   output_dir=output_dir)

