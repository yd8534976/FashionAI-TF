from fashion_stacked import FashionAI
import os
import argparse


def main(a):
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu

    fashionAI = FashionAI(is_training=True)
    fashionAI.train(max_epochs=a.max_epochs,
                    batch_size=a.batch_size,
                    write_summary=a.write_summary,
                    freq_summary=a.freq_summary,
                    dataset_dir=a.dataset_dir,
                    model_dir=a.model_dir,
                    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default='train_set')
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("max_epochs", default=20)
    parser.add_argument("batch_size", default=5)
    parser.add_argument("write_summary", default=True)
    parser.add_argument("freq_summary", default=20)
    parser.add_argument("--gpu", default='0')
    a = parser.parse_args()
    main(a)
