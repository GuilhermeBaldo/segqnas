from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from cnn.cells import DownscalingCell, NonscalingCell, UpscalingCell
from cnn.loss import gen_dice_coef_loss
from cnn.metric import gen_dice_coef


def build_net(input_shape, num_classes, fn_dict, net_list, is_train=True):

    print(fn_dict)
    print(net_list)

    inputs = Input(input_shape, name="input")    
    x = inputs

    filters = [32, 64, 128, 256, 512, 256, 128, 64, 32]
    kernel_size = 3
    block = 'VGGBlock'

    skips = []
    for cell in net_list[0:4]:
        block = fn_dict[cell]['block']
        kernel_size = fn_dict[cell]['params']['kernel']
        filters = fn_dict[cell]['params']['filters']
        x = DownscalingCell(block, kernel_size, filters)
        skips.append(x)

    cell = net_list[4]
    block = fn_dict[cell]['block']
    kernel_size = fn_dict[cell]['params']['kernel']
    filters = fn_dict[cell]['params']['filters']
    x = NonscalingCell(block, kernel_size, filters)(x)

    for cell in net_list[5:]:
        block = fn_dict[cell]['block']
        kernel_size = fn_dict[cell]['params']['kernel']
        filters = fn_dict[cell]['params']['filters']
        x = UpscalingCell(block, kernel_size, filters)([x, skips.pop()])

    # final conv
    prediction = Conv2D(name="prediction_mask",
                                filters=num_classes, kernel_size=(1, 1),
                                activation="sigmoid")(x)
    
    model = Model(inputs=[inputs], outputs=[
                               prediction], name="net")

    model.compile(optimizer=Adam(learning_rate=1e-4), loss=gen_dice_coef_loss, metrics=[gen_dice_coef])

    return model
