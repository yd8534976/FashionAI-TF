from fashion_stacked import FashionAI
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    fashionAI = FashionAI()
    fashionAI.train(max_epochs=20, batch_size=10,
                    write_summary=True, freq_summary=10,
                    model_dir='model/ai0515/fashionai.ckpt',
                    )
