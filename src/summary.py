from torchsummary import torchsummary
import ml.models as ml_models


def combinedv2():
    batch_size = 400
    print("Summary")
    for depth in [0, 1, 2, 3, 4]:
        print('depth: ' + str(depth))
        distmap_model = ml_models.CNN2DModel(feature_channels=8, depth=depth, kernel_size=5, dropout=0.7,
                                             max_length=540)
        torchsummary.summary(distmap_model, input_size=(2, 540, 540), batch_size=batch_size)

        emb_model = ml_models.CNN1DModel(feature_channels=128, kernel_size=5, dropout=0.7, in_channels=1024)
        torchsummary.summary(emb_model, input_size=(1024, 540), batch_size=batch_size)

        in_chan = 128 + 8 * (2 ^ depth)
        bce_model = ml_models.BCEModel(kernel_size=5, in_channels=in_chan)
        torchsummary.summary(bce_model, input_size=(in_chan, 540), batch_size=batch_size)


if __name__ == '__main__':
    combinedv2()
